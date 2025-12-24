"""
Zero-Expert Self-Training Pipeline v2 for Stellar Flare Detection.

This module implements an active learning pipeline that iteratively trains
a classifier to detect stellar flares in ZTF light curves using only
a small set of known flares and a large unlabeled dataset.

Key features:
- Asymmetric pseudo-labeling (aggressive for negatives, conservative for positives)
- Bootstrap consensus for pseudo-label validation
- Three-way data split: training / validation (biased) / held-out (honest)
- Validation set for training decisions (rollback, model selection)
- Truly honest held-out evaluation (only computed once at the end)
- Automatic rollback on degradation
- Adaptive thresholds based on model stability

Decision Criteria:
    Model quality is evaluated using a combined score:
        score = ICE * (1 - w) + tracked_error * w

    where:
        - ICE = Integrated Calibration Error from validation set (lower is better)
        - tracked_error = average of (1 - pos_prob) and neg_prob for tracked samples
        - w = tracked_sample_decision_weight config parameter (default 0.3)

    Tracked samples are two specific row indices (one expected positive, one expected
    negative) whose predictions are monitored across iterations. If the tracked
    positive's probability drops or the tracked negative's probability rises beyond
    configured thresholds, rollback is triggered even if ICE is stable. This provides
    a secondary sanity-check against pseudo-label drift.
"""

from __future__ import annotations

import json
import logging
import psutil
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from os.path import join
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset as HFDataset

import joblib
import numpy as np
import polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from pyutilz.system import tqdmu, clean_ram

# Optional mlframe integration
try:
    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.training.extractors import FeaturesAndTargetsExtractor
    from mlframe.training.configs import TargetTypes
    from mlframe.training.utils import get_pandas_view_of_polars_df

    MLFRAME_AVAILABLE = True
except ImportError:
    MLFRAME_AVAILABLE = False
    FeaturesAndTargetsExtractor = object  # Fallback for class inheritance
    get_pandas_view_of_polars_df = None

# report_model_perf integration
from mlframe.training_old import report_model_perf
from mlframe.training.evaluation import plot_model_feature_importances

# Optional astro_flares integration for plotting
try:
    from astro_flares import view_series

    VIEW_SERIES_AVAILABLE = True
except ImportError:
    VIEW_SERIES_AVAILABLE = False
    view_series = None

# Jupyter detection
try:
    from pyutilz.pythonlib import is_jupyter_notebook
except ImportError:

    def is_jupyter_notebook():
        return False


# Numba JIT compilation for hot numerical loops
try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Provide no-op decorators as fallback
    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator if not args or callable(args[0]) else decorator

    def prange(*args):
        return range(*args)


logger = logging.getLogger(__name__)

# =============================================================================
# Module Constants
# =============================================================================

# Stratification for train/val/test splits
# 5 bins with 10+ samples each ensures balanced representation of flare types
DEFAULT_STRATIFICATION_BINS = 5
MIN_SAMPLES_PER_STRATIFICATION_BIN = 10

# Bootstrap consensus requires at least 2 models to agree, even if n_bootstrap is small
MIN_BOOTSTRAP_CONSENSUS_MODELS = 2

# Safety limit for time-based stopping; prevents infinite loops if time check fails
FALLBACK_MAX_ITERATIONS = 100_000

# Threshold bounds prevent thresholds from becoming too extreme during adaptation
MAX_PSEUDO_POS_THRESHOLD_BOUND = 0.999  # Pos threshold can't exceed this (too strict = no candidates)
MIN_PSEUDO_NEG_THRESHOLD_BOUND = 0.01  # Neg threshold can't go below this (too loose = contamination)
DEFAULT_CLASSIFICATION_THRESHOLD = 0.5

# Pseudo-labeling candidate selection multipliers
# These control the candidate pool size before final selection:
# - SUBSAMPLE: For negatives, we subsample to 2x the max we'll actually add. Higher values
#   explore more candidates but increase compute. 2x empirically balances diversity vs. cost.
# - TOP_K: For positives, we take the top 50x candidates by probability before consensus check.
#   Higher values increase chance of finding high-consensus candidates.
PSEUDO_NEG_SUBSAMPLE_MULTIPLIER = 2
PSEUDO_POS_TOP_K_MULTIPLIER = 50

# Positive sample weights grow slower than negatives (0.5x increment rate) because
# false positives are more damaging than false negatives in rare event detection
PSEUDO_POS_WEIGHT_INCREMENT_SCALE = 0.5

# Bootstrap consensus: fraction of models that must agree for negative pseudo-labels
# 0.5 = majority consensus. Lower values increase recall but risk contamination.
BOOTSTRAP_CONSENSUS_FRACTION = 0.5

# Freaky held-out set: minimum negatives to include for statistical validity
FREAKY_NEG_MIN_COUNT = 10

# Curriculum learning thresholds define three training phases:
# Phase 1 (early): Only use high-confidence samples (>95%) until recall reaches 50%
# Phase 2 (middle): Include medium-confidence (>85%) until recall reaches 65%
# Phase 3 (mature): Include lower-confidence (>70%) samples
# This prevents the model from learning noise early when it's unstable.
DEFAULT_CURRICULUM_PHASE1_RECALL = 0.5
DEFAULT_CURRICULUM_PHASE2_RECALL = 0.65
DEFAULT_CURRICULUM_PHASE1_CONF = 0.95
DEFAULT_CURRICULUM_PHASE2_CONF = 0.85
DEFAULT_CURRICULUM_PHASE3_CONF = 0.70
DEFAULT_CURRICULUM_PHASE2_WEIGHT = 0.7  # Weight applied to phase 2 samples


# =============================================================================
# Numba JIT-Compiled Functions
# =============================================================================


@njit(cache=True, parallel=True)
def _compute_curriculum_weights_numba(
    confidences: np.ndarray,
    current_recall: float,
    phase1_recall: float,
    phase2_recall: float,
    phase1_conf: float,
    phase2_conf: float,
    phase3_conf: float,
    phase2_weight: float,
) -> np.ndarray:
    """
    JIT-compiled curriculum weight computation.

    Computes sample weights based on confidence and model maturity (current recall).
    Uses parallel loops for large arrays.
    """
    n = len(confidences)
    weights = np.empty(n, dtype=np.float32)

    if current_recall < phase1_recall:
        # Phase 1: Strict - only very confident pseudo-labels
        for i in prange(n):
            weights[i] = 1.0 if confidences[i] > phase1_conf else 0.0
    elif current_recall < phase2_recall:
        # Phase 2: Medium - expand with reduced weights
        for i in prange(n):
            if confidences[i] > phase1_conf:
                weights[i] = 1.0
            elif confidences[i] > phase2_conf:
                weights[i] = phase2_weight
            else:
                weights[i] = 0.0
    else:
        # Phase 3: Mature - use gradient weights
        for i in prange(n):
            if confidences[i] > phase1_conf:
                weights[i] = 1.0
            elif confidences[i] > phase3_conf:
                weights[i] = confidences[i]
            else:
                weights[i] = 0.0
    return weights


@njit(cache=True, parallel=True)
def _count_low_prob_consensus_numba(
    bootstrap_preds: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Count models with prob < threshold per candidate (JIT-compiled).

    Parameters
    ----------
    bootstrap_preds : np.ndarray
        Shape (n_models, n_candidates).
    threshold : float
        Probability threshold.

    Returns
    -------
    np.ndarray
        Count of models below threshold for each candidate.
    """
    n_models, n_candidates = bootstrap_preds.shape
    counts = np.zeros(n_candidates, dtype=np.int32)
    for i in prange(n_candidates):
        for j in range(n_models):
            if bootstrap_preds[j, i] < threshold:
                counts[i] += 1
    return counts


@njit(cache=True, parallel=True)
def _check_all_high_consensus_numba(
    bootstrap_preds: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Check if ALL models have prob > threshold per candidate (JIT-compiled).

    Parameters
    ----------
    bootstrap_preds : np.ndarray
        Shape (n_models, n_candidates).
    threshold : float
        Probability threshold.

    Returns
    -------
    np.ndarray
        Boolean array: True if all models exceed threshold.
    """
    n_models, n_candidates = bootstrap_preds.shape
    result = np.ones(n_candidates, dtype=np.bool_)
    for i in prange(n_candidates):
        for j in range(n_models):
            if bootstrap_preds[j, i] <= threshold:
                result[i] = False
                break
    return result


@njit(cache=True, parallel=True)
def _compute_bootstrap_stats_batch_numba(
    bootstrap_preds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean, std, and consensus score for all candidates at once (JIT-compiled).

    Parameters
    ----------
    bootstrap_preds : np.ndarray
        Shape (n_models, n_candidates).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (means, stds, consensus_scores) each of shape (n_candidates,).
    """
    n_models, n_candidates = bootstrap_preds.shape
    means = np.empty(n_candidates, dtype=np.float32)
    stds = np.empty(n_candidates, dtype=np.float32)
    consensus = np.empty(n_candidates, dtype=np.float32)

    for i in prange(n_candidates):
        # Compute mean
        total = 0.0
        for j in range(n_models):
            total += bootstrap_preds[j, i]
        mean = total / n_models
        means[i] = mean

        # Compute std
        sq_diff_sum = 0.0
        for j in range(n_models):
            diff = bootstrap_preds[j, i] - mean
            sq_diff_sum += diff * diff
        std = np.sqrt(sq_diff_sum / n_models)
        stds[i] = std
        consensus[i] = 1.0 - std

    return means, stds, consensus


def compute_bootstrap_stats(bootstrap_preds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean, std, and consensus from bootstrap predictions.

    Wrapper that uses numba JIT version when available, otherwise falls back
    to numpy implementation.

    Parameters
    ----------
    bootstrap_preds : np.ndarray
        Shape (n_models, n_candidates).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (means, stds, consensus_scores) each of shape (n_candidates,).
    """
    if NUMBA_AVAILABLE:
        return _compute_bootstrap_stats_batch_numba(bootstrap_preds)
    return (
        bootstrap_preds.mean(axis=0).astype(np.float32),
        bootstrap_preds.std(axis=0).astype(np.float32),
        (1.0 - bootstrap_preds.std(axis=0)).astype(np.float32),
    )


# =============================================================================
# Enums
# =============================================================================


class SampleSource(str, Enum):
    """Source type for labeled samples in the training set.

    Inherits from str to maintain JSON serialization compatibility.
    """

    SEED = "seed"
    PSEUDO_POS = "pseudo_pos"
    PSEUDO_NEG = "pseudo_neg"
    FORCED_POS = "forced_pos"  # User-specified positive from unlabeled pool
    FORCED_NEG = "forced_neg"  # User-specified negative from unlabeled pool


# =============================================================================
# Helper Functions for Improved Pipeline
# =============================================================================


def _random_split(
    all_idx: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random fallback split for stratified_flare_split."""
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(all_idx)  # Shuffle the actual indices, not 0..n
    n_samples = len(all_idx)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    return (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_val],
        shuffled[n_train + n_val :],
    )


def stratified_flare_split(
    known_flares: pl.DataFrame,
    train_ratio: float = 0.10,
    val_ratio: float = 0.40,
    held_out_ratio: float = 0.50,
    stratify_cols: list[str] | None = None,
    random_state: int = 42,
    exclude_indices: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform stratified split of known flares to ensure representative subsets.

    In active learning, the quality of the initial train/val/held-out split
    significantly impacts convergence. Random splitting may create subsets
    that don't represent the full diversity of flare characteristics.

    This function stratifies by flare properties (amplitude, duration) to ensure
    each subset contains examples from different regions of the feature space.
    This is particularly important for rare event detection where the positive
    class has high internal variance.

    Parameters
    ----------
    known_flares : pl.DataFrame
        DataFrame of known flares with feature columns.
    train_ratio : float, default 0.10
        Fraction of flares for initial training seed.
    val_ratio : float, default 0.40
        Fraction for validation pool (used for hard example mining).
    held_out_ratio : float, default 0.50
        Fraction for held-out evaluation (never touched during training).
    stratify_cols : list[str], optional
        Columns to use for stratification. If None, attempts to use
        'norm_amplitude_sigma' and 'npoints' if available.
    random_state : int, default 42
        Random seed for reproducibility.
    exclude_indices : list[int], optional
        Row indices to exclude from the split (e.g., freaky samples to report separately).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (train_indices, val_indices, held_out_indices) as numpy arrays of row indices.

    Notes
    -----
    If stratification fails (e.g., too few samples per stratum), falls back
    to random splitting with a warning.

    References
    ----------
    - Settles, B. (2012). Active Learning. Morgan & Claypool.
      Chapter on query strategies and sample selection.
    """
    n_samples = len(known_flares)
    all_idx = np.arange(n_samples)

    # Exclude specified indices from splitting
    if exclude_indices:
        exclude_set = set(exclude_indices)
        all_idx = np.array([i for i in all_idx if i not in exclude_set])
        logger.info(f"Excluded {len(exclude_indices)} indices from split, {len(all_idx)} remaining")

    # Determine stratification columns
    if stratify_cols is None:
        stratify_cols = []
        if "norm_amplitude_sigma" in known_flares.columns:
            stratify_cols.append("norm_amplitude_sigma")
        if "npoints" in known_flares.columns:
            stratify_cols.append("npoints")

    # If no stratification possible, use random split
    if not stratify_cols:
        logger.warning("No stratification columns found, using random split")
        return _random_split(all_idx, train_ratio, val_ratio, random_state)

    # Build stratification labels
    try:
        # Extract each column separately from polars to avoid shape issues
        # IMPORTANT: Only extract for samples in all_idx (after exclusion filtering)
        strat_arrays = []
        for col in stratify_cols:
            col_data = known_flares[col].to_numpy()
            # Ensure 1D and float
            col_data = np.asarray(col_data, dtype=np.float64).flatten()
            # Filter to only include samples in all_idx
            col_data = col_data[all_idx]
            strat_arrays.append(col_data)

        # Stack into 2D array (n_filtered_samples, n_cols)
        if len(strat_arrays) == 1:
            strat_data = strat_arrays[0].reshape(-1, 1)
        else:
            strat_data = np.column_stack(strat_arrays)

        # Use quantile binning to create strata
        n_filtered = len(all_idx)
        n_bins = min(DEFAULT_STRATIFICATION_BINS, n_filtered // MIN_SAMPLES_PER_STRATIFICATION_BIN)
        if n_bins < 2:
            raise ValueError("Too few samples for stratification")

        binner = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
        strata = np.asarray(binner.fit_transform(strat_data))

        # Combine multiple columns into single stratum label
        # strata_labels now has same length as all_idx
        if strata.shape[1] > 1:
            strata_labels = (strata[:, 0] * n_bins + strata[:, 1]).astype(int)
        else:
            strata_labels = strata.ravel().astype(int)

        # First split: train vs rest
        # Note: all_idx contains the original row indices, strata_labels is aligned with all_idx
        train_idx, rest_idx, _, rest_strata = train_test_split(
            all_idx,
            strata_labels,
            train_size=train_ratio,
            stratify=strata_labels,
            random_state=random_state,
        )

        # Second split: val vs held-out from rest
        # rest_strata was returned from train_test_split, already aligned with rest_idx
        if held_out_ratio == 0:
            # All rest goes to validation, no held-out
            val_idx = rest_idx
            held_out_idx = np.array([], dtype=np.int64)
        else:
            val_frac = val_ratio / (val_ratio + held_out_ratio)
            val_idx, held_out_idx = train_test_split(
                rest_idx,
                train_size=val_frac,
                stratify=rest_strata,
                random_state=random_state,
            )

        logger.info(f"Stratified split: train={len(train_idx)}, val={len(val_idx)}, held_out={len(held_out_idx)}")

        return train_idx, val_idx, held_out_idx

    except ValueError as e:
        logger.warning(f"Stratified split failed ({e}), falling back to random")
        return _random_split(all_idx, train_ratio, val_ratio, random_state)


def get_adaptive_curriculum_weights(
    confidences: np.ndarray,
    current_recall: float,
    config: PipelineConfig | None = None,
) -> np.ndarray:
    """
    Compute sample weights using adaptive curriculum learning (vectorized).

    Curriculum learning gradually introduces harder examples as the model
    improves. This function determines sample weights based on both the
    confidence of the pseudo-label AND the current model quality (recall).

    The key insight: Early in training (low recall), the model makes many
    mistakes. Adding low-confidence pseudo-labels amplifies these errors.
    As the model matures (high recall), it can benefit from harder examples.

    Phases
    ------
    Phase 1 (recall < phase1_recall):
        Only accept high-confidence pseudo-labels (>phase1_conf).
        The model is still learning basic patterns.

    Phase 2 (phase1_recall <= recall < phase2_recall):
        Accept medium-confidence labels (>phase2_conf) with reduced weight.
        The model is improving and can handle some uncertainty.

    Phase 3 (recall >= phase2_recall):
        Accept lower-confidence labels (>phase3_conf) with weight proportional
        to confidence. The model is mature enough to learn from edge cases.

    Parameters
    ----------
    confidences : np.ndarray
        Array of pseudo-label confidences in [0, 1].
    current_recall : float
        Current held-out recall in [0, 1]. Measures model maturity.
    config : PipelineConfig, optional
        Configuration with curriculum thresholds. If None, uses defaults.

    Returns
    -------
    np.ndarray
        Sample weights in [0, 1]. Weight of 0 means "don't include this sample".

    References
    ----------
    - Bengio et al. (2009). Curriculum Learning. ICML.
    - Kumar et al. (2010). Self-Paced Learning for Latent Variable Models.
    """
    # Use config values or module-level defaults
    if config is not None:
        phase1_recall = config.curriculum.phase1_recall
        phase2_recall = config.curriculum.phase2_recall
        phase1_conf = config.curriculum.phase1_conf
        phase2_conf = config.curriculum.phase2_conf
        phase3_conf = config.curriculum.phase3_conf
        phase2_weight = config.curriculum.phase2_weight
    else:
        # Use module-level defaults (single source of truth with CurriculumConfig)
        phase1_recall = DEFAULT_CURRICULUM_PHASE1_RECALL
        phase2_recall = DEFAULT_CURRICULUM_PHASE2_RECALL
        phase1_conf = DEFAULT_CURRICULUM_PHASE1_CONF
        phase2_conf = DEFAULT_CURRICULUM_PHASE2_CONF
        phase3_conf = DEFAULT_CURRICULUM_PHASE3_CONF
        phase2_weight = DEFAULT_CURRICULUM_PHASE2_WEIGHT

    # Use numba JIT version for performance when available
    if NUMBA_AVAILABLE:
        return _compute_curriculum_weights_numba(
            confidences.astype(np.float32),
            current_recall,
            phase1_recall,
            phase2_recall,
            phase1_conf,
            phase2_conf,
            phase3_conf,
            phase2_weight,
        )

    # Fallback to numpy vectorized implementation
    if current_recall < phase1_recall:
        return np.where(confidences > phase1_conf, 1.0, 0.0).astype(np.float32)
    elif current_recall < phase2_recall:
        return np.where(confidences > phase1_conf, 1.0, np.where(confidences > phase2_conf, phase2_weight, 0.0)).astype(np.float32)
    else:
        return np.where(confidences > phase1_conf, 1.0, np.where(confidences > phase3_conf, confidences, 0.0)).astype(np.float32)


def compute_oob_metrics(
    bootstrap_models: list[CatBoostClassifier],
    features: np.ndarray,
    labels: np.ndarray,
    bootstrap_indices_list: list[np.ndarray],
) -> dict:
    """
    Compute Out-of-Bag (OOB) predictions and metrics.

    OOB evaluation provides an unbiased estimate of model performance without
    requiring a separate validation set. Each sample is evaluated only by
    bootstrap models that did NOT include it in their training set.

    For bootstrap samples, approximately 37% of the original data is left out
    (OOB) for each model. With multiple bootstrap models, most samples have
    at least one OOB prediction available.

    OOB metrics serve as a "free" validation signal that can detect:
    - Overfitting (OOB metrics << training metrics)
    - Instability (high variance in OOB predictions)
    - Divergence from held-out (if OOB and held-out disagree significantly)

    Parameters
    ----------
    bootstrap_models : list
        List of trained bootstrap models with predict_proba method.
    features : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    labels : np.ndarray
        True labels of shape (n_samples,).
    bootstrap_indices_list : list[np.ndarray]
        List of arrays, each containing indices used to train corresponding
        bootstrap model.

    Returns
    -------
    dict
        Dictionary with:
        - oob_proba: np.ndarray of OOB predicted probabilities
        - oob_recall: Recall computed on OOB predictions
        - oob_precision: Precision computed on OOB predictions
        - oob_coverage: Fraction of samples with OOB predictions

    References
    ----------
    - Breiman, L. (1996). Out-of-bag estimation. Technical report.
    - Hastie et al. (2009). Elements of Statistical Learning, Ch. 15.
    """
    n_samples = len(labels)
    n_models = len(bootstrap_models)

    # Accumulate OOB predictions
    oob_predictions = np.zeros((n_samples, n_models), dtype=np.float32)
    oob_counts = np.zeros(n_samples, dtype=np.int32)

    for i, (model, train_indices) in enumerate(zip(bootstrap_models, bootstrap_indices_list)):
        # OOB mask: samples NOT in this bootstrap's training set
        oob_mask = np.ones(n_samples, dtype=bool)
        oob_mask[train_indices] = False

        if np.any(oob_mask):
            oob_features = features[oob_mask]
            oob_predictions[oob_mask, i] = model.predict_proba(oob_features)[:, 1]
            oob_counts[oob_mask] += 1

    # Compute average OOB prediction for each sample
    valid_mask = oob_counts > 0
    oob_proba = np.zeros(n_samples, dtype=np.float32)
    oob_proba[valid_mask] = oob_predictions[valid_mask].sum(axis=1) / oob_counts[valid_mask]

    # Compute metrics on samples with OOB coverage
    if valid_mask.sum() == 0:
        return {
            "oob_proba": oob_proba,
            "oob_recall": 0.0,
            "oob_precision": 0.0,
            "oob_coverage": 0.0,
        }

    oob_preds = (oob_proba[valid_mask] > 0.5).astype(int)
    valid_labels = labels[valid_mask]

    return {
        "oob_proba": oob_proba,
        "oob_recall": recall_score(valid_labels, oob_preds, zero_division=0),
        "oob_precision": precision_score(valid_labels, oob_preds, zero_division=0),
        "oob_coverage": float(valid_mask.mean()),
    }


# =============================================================================
# mlframe Integration (Optional)
# =============================================================================


class ActiveLearningFeaturesExtractor(FeaturesAndTargetsExtractor):
    """
    Adapter for integrating active learning pipeline with mlframe.

    This class bridges the labeled_samples from pseudo-labeling with mlframe's
    training infrastructure. It provides:
    1. Feature extraction from labeled samples
    2. Target extraction with proper types
    3. Curriculum-based sample weights

    The architecture allows using mlframe's full power (calibration, feature
    importance, early stopping) without duplicating preprocessing/training code.

    Parameters
    ----------
    labeled_samples : list[LabeledSample]
        Labeled training samples from active learning iterations.
    unlabeled_samples : pl.DataFrame
        Large unlabeled dataset (for pseudo-labeled negatives/positives).
    known_flares : pl.DataFrame
        Known flares dataset (for seed positives).
    feature_cols : list[str]
        Feature column names to use.
    current_recall : float
        Current validation recall for adaptive curriculum weighting.
    verbose : int
        Verbosity level.

    Notes
    -----
    mlframe will receive a DataFrame containing only the labeled samples,
    which is more memory-efficient than passing the full dataset.
    """

    def __init__(
        self,
        labeled_samples: list,  # list[LabeledSample]
        unlabeled_samples: pl.DataFrame,
        known_flares: pl.DataFrame,
        feature_cols: list[str],
        current_recall: float = 0.0,
        config: PipelineConfig | None = None,
        verbose: int = 1,
    ):
        if not MLFRAME_AVAILABLE:
            raise ImportError("mlframe is required for this feature. Install it first.")

        super().__init__(
            columns_to_drop={"_label", "_weight", "_source", "_confidence"},
            verbose=verbose,
        )
        self.labeled_samples = labeled_samples
        self.unlabeled_samples = unlabeled_samples
        self.known_flares = known_flares
        self.feature_cols = feature_cols
        self.current_recall = current_recall
        self.config = config
        self._prepared_df: pl.DataFrame | None = None

    def _build_rows_for_batch(
        self,
        samples: list[tuple[int, LabeledSample]],
        source_df: pl.DataFrame,
        rows: list[dict | None],
    ) -> None:
        """
        Build feature rows for a batch of samples from a single source.

        Parameters
        ----------
        samples : list[tuple[int, LabeledSample]]
            List of (output_index, sample) tuples.
        source_df : pl.DataFrame
            Source DataFrame (known_flares or unlabeled_samples).
        rows : list[dict | None]
            Output list to populate (mutated in place).
        """
        if not samples:
            return

        indices = [s.index for _, s in samples]
        data = source_df[indices].select(self.feature_cols).to_dicts()

        # Vectorized curriculum weight computation
        confidences = np.array([s.confidence for _, s in samples], dtype=np.float32)
        sample_weights = np.array([s.weight for _, s in samples], dtype=np.float32)
        curriculum_weights = get_adaptive_curriculum_weights(confidences, self.current_recall, self.config)
        final_weights = sample_weights * curriculum_weights

        for j, (i, sample) in enumerate(samples):
            row_data = data[j].copy()
            row_data["_label"] = sample.label
            row_data["_confidence"] = sample.confidence
            row_data["_source"] = sample.source
            row_data["_weight"] = float(final_weights[j])
            rows[i] = row_data

    def add_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Build DataFrame from labeled samples.

        This method constructs the training DataFrame by extracting features
        from the appropriate source (known_flares or unlabeled_samples) for each
        labeled sample. Uses batched extraction for efficiency.

        Parameters
        ----------
        df : pl.DataFrame
            Dummy DataFrame (ignored, we build our own).

        Returns
        -------
        pl.DataFrame
            DataFrame with features and metadata columns.
        """
        if self._prepared_df is not None:
            return self._prepared_df

        # Group samples by source for batched extraction
        oos_samples = [(i, s) for i, s in enumerate(self.labeled_samples) if s.from_known_flares]
        big_samples = [(i, s) for i, s in enumerate(self.labeled_samples) if not s.from_known_flares]

        # Preallocate rows list
        rows: list[dict | None] = [None] * len(self.labeled_samples)

        # Batch extract from both sources using helper
        self._build_rows_for_batch(oos_samples, self.known_flares, rows)
        self._build_rows_for_batch(big_samples, self.unlabeled_samples, rows)

        self._prepared_df = pl.DataFrame(rows)
        logger.info(f"ActiveLearningFeaturesExtractor: built {len(rows)} samples")
        return self._prepared_df

    def build_targets(self, df: pl.DataFrame) -> dict:
        """
        Extract targets from the prepared DataFrame.

        Returns
        -------
        dict
            Dictionary mapping TargetTypes.BINARY_CLASSIFICATION to target dict.
        """
        targets = {"flare": df["_label"].cast(pl.Int8).to_numpy()}
        return {TargetTypes.BINARY_CLASSIFICATION: targets}

    def get_sample_weights(
        self,
        df: pl.DataFrame,
        timestamps=None,
    ) -> dict[str, np.ndarray]:
        """
        Return curriculum-based sample weights.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary with "curriculum" weights.
        """
        weights = df["_weight"].to_numpy().astype(np.float32)

        # Filter out zero-weight samples
        nonzero_count = np.sum(weights > 0)
        logger.info(f"Curriculum weights: {nonzero_count}/{len(weights)} samples with non-zero weight")

        return {"curriculum": weights}


def train_model_via_mlframe(
    labeled_samples: list,  # list[LabeledSample]
    unlabeled_samples: pl.DataFrame,
    known_flares: pl.DataFrame,
    feature_cols: list[str],
    current_recall: float,
    iteration: int,
    output_dir: Path,
    config: PipelineConfig | None = None,
    mlframe_models: list[str] | None = None,
) -> tuple[dict, dict]:
    """
    Train ensemble using mlframe.

    mlframe automatically handles:
    - Train/val split within labeled_samples
    - Early stopping based on ICE metric
    - Calibration curves
    - Feature importance
    - Multiple model types (CatBoost, LightGBM, XGBoost)

    Parameters
    ----------
    labeled_samples : list[LabeledSample]
        Labeled training samples.
    unlabeled_samples : pl.DataFrame
        Large dataset for feature lookup.
    known_flares : pl.DataFrame
        Known flares dataset for feature lookup.
    feature_cols : list[str]
        Feature column names.
    current_recall : float
        Current held-out recall for curriculum weighting.
    iteration : int
        Current iteration number.
    config : PipelineConfig, optional
        Pipeline configuration for curriculum learning thresholds.
    output_dir : Path
        Output directory for models.
    mlframe_models : list[str], optional
        Model types to train. Default: ["cb"] (CatBoost only).

    Returns
    -------
    tuple[dict, dict]
        (models_dict, metadata_dict)
    """
    if not MLFRAME_AVAILABLE:
        raise ImportError("mlframe is required. Install it or use train_model() instead.")

    if mlframe_models is None:
        mlframe_models = ["cb"]  # CatBoost only by default

    extractor = ActiveLearningFeaturesExtractor(
        labeled_samples=labeled_samples,
        unlabeled_samples=unlabeled_samples,
        known_flares=known_flares,
        feature_cols=feature_cols,
        current_recall=current_recall,
        config=config,
    )

    # Dummy DataFrame - extractor.add_features() will build the real one
    dummy_df = pl.DataFrame({"dummy": [0]})

    models, metadata = train_mlframe_models_suite(
        df=dummy_df,
        target_name="flare_detection",
        model_name=f"active_learning_iter_{iteration:03d}",
        features_and_targets_extractor=extractor,
        mlframe_models=mlframe_models,
        use_mlframe_ensembles=len(mlframe_models) > 1,
        data_dir=str(output_dir),
        models_dir="mlframe_models",
        verbose=1,
    )

    return models, metadata


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DataSplitConfig:
    """Configuration for train/validation/held-out data splits."""

    n_train_flares: int = 62  # Initial training flares (was 50, +12 from held-out)
    n_train_neg_init: int = 3500  # Initial training negatives (was 1000, +2500 from held-out)
    n_validation_flares: int = 37  # Validation set flares (was 25, +12 from held-out)
    n_validation_neg: int = 7500  # Validation set negatives (was 5000, +2500 from held-out)
    n_held_out_flares: int = 0  # Held-out flares (moved to train/validation)
    n_held_out_neg: int = 0  # Held-out negatives (moved to train/validation)


@dataclass
class CatBoostConfig:
    """CatBoost model hyperparameters."""

    iterations: int = 2000
    depth: int = 6  # Deeper trees for richer feature interactions
    learning_rate: float = 0.1
    verbose: bool = False
    use_gpu: bool = True
    eval_fraction: float = 0.1  # Fraction for auto early stopping
    early_stopping_rounds: int = 150
    plot: bool = True  # Plot training progress
    loss_function: str = "Logloss"
    eval_metric: str = "Logloss"
    # Regularization parameters
    l2_leaf_reg: float = 3.0  # L2 regularization on leaf values
    min_data_in_leaf: int = 20  # Minimum samples per leaf to prevent overfitting
    random_strength: float = 0.5  # Moderate randomness in split selection


@dataclass
class CurriculumConfig:
    """Curriculum learning phase thresholds."""

    phase1_recall: float = DEFAULT_CURRICULUM_PHASE1_RECALL  # Below this: strict phase
    phase2_recall: float = DEFAULT_CURRICULUM_PHASE2_RECALL  # Below this: medium phase
    phase1_conf: float = DEFAULT_CURRICULUM_PHASE1_CONF  # Required confidence in phase 1
    phase2_conf: float = DEFAULT_CURRICULUM_PHASE2_CONF  # Required confidence in phase 2
    phase3_conf: float = DEFAULT_CURRICULUM_PHASE3_CONF  # Required confidence in phase 3
    phase2_weight: float = DEFAULT_CURRICULUM_PHASE2_WEIGHT  # Weight for medium-confidence samples in phase 2


@dataclass
class ThresholdConfig:
    """Pseudo-labeling thresholds and adaptive adjustment settings."""

    # Initial thresholds
    pseudo_pos_threshold: float = 0.998
    pseudo_neg_threshold: float = 0.002
    consensus_threshold: float = 0.99

    # Limits per iteration (frugal: 10x less than before for slower, more careful learning)
    max_pseudo_pos_per_iter: int = 200  # Increased to speed up positive class expansion
    max_pseudo_neg_per_iter: int = 10000

    # Adaptive adjustments
    enable_adaptive_thresholds: bool = False  # Set True to enable threshold relaxing/tightening
    relax_successful_iters: int = 3  # Iters before relaxing
    relax_pos_delta: float = 0.005
    relax_neg_delta: float = 0.01
    relax_max_pos_delta: int = 1
    tighten_pos_delta: float = 0.01
    tighten_neg_delta: float = 0.02
    tighten_max_pos_delta: int = 3

    # Bounds
    min_pseudo_pos_threshold: float = 0.99
    max_pseudo_neg_threshold: float = 0.01
    min_pseudo_pos_per_iter: int = 3
    max_pseudo_pos_cap: int = 200  # Must be >= max_pseudo_pos_per_iter initial value

    # Review thresholds (more aggressive to actually trigger removals)
    pseudo_pos_removal_prob: float = 0.9  # was 0.3 - pseudo_pos removed if prob drops below this
    pseudo_neg_promotion_prob: float = 0.1  # was 0.7 - pseudo_neg removed if prob rises above this
    bootstrap_instability_std: float = 0.2
    bootstrap_instability_mean: float = 0.7
    seed_neg_removal_prob: float = 0.2
    seed_neg_removal_bootstrap_mean: float = 0.85
    neg_consensus_min_low_prob: float = 0.1

    # Bootstrap ensemble for uncertainty estimation
    n_bootstrap_models: int = 3  # Increased for better uncertainty estimation
    bootstrap_iterations: int = 1000  # Reduced per model, compensated by more models
    bootstrap_early_stopping_rounds: int = 150  # Same as main model
    bootstrap_variance_threshold: float = 0.03

    # Ban list (prevents re-adding samples after rollback)
    ban_iterations: int = 10  # How many iterations banned samples stay banned after rollback

    # Sample weights for pseudo-labels
    initial_pseudo_pos_weight: float = 0.2
    initial_pseudo_neg_weight: float = 0.8
    weight_increment: float = 0.1
    max_pseudo_pos_weight: float = 0.8


@dataclass
class StoppingConfig:
    """Stopping criteria and plateau detection settings."""

    # Enrichment calculation
    top_k_candidates: int = 10000  # Number of top candidates for enrichment

    # Candidate saving
    top_k_candidates_save: int = 5000  # Number of candidates to save per iteration

    # Plateau detection
    plateau_window: int = 20  # Iterations to check for plateau
    plateau_recall_range: float = 0.01  # Min recall variation for plateau
    plateau_ice_range: float = 0.01  # Min ICE variation for plateau
    plateau_min_zero_pos_iters: int = 5  # Iterations with 0 pseudo_pos for plateau

    # Instability detection
    max_rollbacks_for_instability: int = 10  # Max rollbacks in window before stopping
    rollback_window: int = 10  # Window size for counting rollbacks


@dataclass
class PipelineConfig:
    """Configuration for the active learning pipeline.

    Groups related settings into nested dataclasses for better organization:
    - data: Train/validation/held-out split sizes
    - catboost: Model hyperparameters
    - curriculum: Curriculum learning phases
    - thresholds: Pseudo-labeling thresholds and adjustments
    - stopping: Stopping criteria and plateau detection

    Decision criteria settings (rollback and best model selection):
    - ice_increase_threshold: Maximum ICE increase before triggering rollback
    - tracked_sample_decision_weight: Weight for tracked samples in combined score (0-1)
    - tracked_pos_degradation_threshold: Max prob drop for tracked positive before rollback
    - tracked_neg_degradation_threshold: Max prob rise for tracked negative before rollback
    """

    # Nested configuration groups
    data: DataSplitConfig = field(default_factory=DataSplitConfig)
    catboost: CatBoostConfig = field(default_factory=CatBoostConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    stopping: StoppingConfig = field(default_factory=StoppingConfig)

    # Stopping criteria
    max_iters: int | None = None
    max_time_hours: float | None = None
    target_ice: float | None = None

    # Success targets
    target_recall: float = 1.00
    target_precision: float = 1.00
    target_enrichment: float = 999.0
    min_successful_iters_for_success: int = 50

    # Class balancing
    class_imbalance_threshold: float = 20.0
    class_weight_divisor: float = 10.0

    # Rollback / Decision criteria
    ice_increase_threshold: float = 0.10
    oob_divergence_warning: float = 0.15
    # Tracked sample weights in combined decision score (0 = ignore, higher = more influence)
    tracked_sample_decision_weight: float = 0.3  # Weight for tracked samples vs ICE in combined score
    tracked_pos_degradation_threshold: float = 0.30  # Trigger rollback if tracked pos prob drops by this much
    tracked_neg_degradation_threshold: float = 0.30  # Trigger rollback if tracked neg prob rises by this much

    # Enrichment
    assumed_prevalence: float = 0.001
    enrichment_every_n_iters: int = 5

    # Prediction
    prediction_batch_size: int = 100_000_000

    # Plotting
    plot_samples: bool = True
    plot_singlepoint_min_outlying_factor: float = 10.0
    display_perf_charts: bool = False  # Show matplotlib windows for performance charts (default OFF)
    display_sample_plots: bool = False  # Show matplotlib windows for sample plots (default OFF)

    # Misc
    review_seed_negatives: bool = True
    tracked_positive_rowid: int | None = 55554273  # Row index in unlabeled_samples to track (expected positive)
    tracked_negative_rowid: int | None = 83251304  # Row index in unlabeled_samples to track (expected negative)
    exclude_columns: set[str] = field(default_factory=lambda: {"id", "class", "ts", "index"})

    # mlframe integration
    use_mlframe: bool = False
    mlframe_models: list[str] = field(default_factory=lambda: ["cb"])

    # Feature curriculum (gradual feature introduction)
    feature_warmup_enabled: bool = True  # Enable gradual feature introduction
    feature_warmup_initial_prefix: str = "wv_"  # Start with features matching this prefix
    feature_warmup_expansion_rate: float = 0.10  # Add 10% of remaining features per step
    feature_warmup_expansion_interval: int = 20  # Expand feature set every N iterations
    feature_warmup_full_features_iter: int | None = None  # Force all features after this iter (None = gradual until done)

    # Forced samples from unlabeled pool (user-verified samples to inject into training)
    # These get weight=1.0 (same as known_flares), are protected from relabeling and banning
    forced_positive_indices: list[int] = field(default_factory=list)  # Row indices in unlabeled_samples (verified positives)
    forced_negative_indices: list[int] = field(default_factory=list)  # Row indices in unlabeled_samples (verified negatives)


@dataclass(slots=True)
class LabeledSample:
    """A single labeled sample in the training set."""

    index: int  # Index in the source DataFrame (unlabeled_samples or known_flares)
    label: int  # 0 or 1
    weight: float  # Sample weight for training
    source: SampleSource  # Origin of the sample (seed, pseudo_pos, pseudo_neg)
    added_iter: int  # Iteration when added
    confidence: float  # P(label) when added
    consensus_score: float  # Bootstrap consensus [0, 1]
    from_known_flares: bool = False  # True if from known_flares, False if from unlabeled_samples


@dataclass(slots=True)
class SampleRemovalInfo:
    """Information about a sample marked for removal during pseudo-label review."""

    sample_index: int  # Index in the source DataFrame
    is_from_known_flares: bool  # True if from known_flares, False if from unlabeled_samples
    sample: LabeledSample  # The labeled sample being removed
    current_prob: float = 0.0  # Current P(flare) that triggered removal


@dataclass
class Checkpoint:
    """Model checkpoint with associated metadata."""

    iteration: int
    model_path: Path
    labeled_train: list[LabeledSample]
    metrics: dict
    config_snapshot: dict
    ban_list: dict = field(default_factory=dict)  # (index, label) tuple string -> expires_at_iteration


@dataclass(slots=True)
class HeldOutSet:
    """Truly held-out set - NEVER used for any training decisions.

    This set is only evaluated once at the very end (in _finalize) to provide
    honest, unbiased metrics that have not been used for model selection,
    rollback decisions, or threshold adjustments.
    """

    flare_indices: np.ndarray  # Indices in known_flares
    negative_indices: np.ndarray  # Indices in unlabeled_samples


@dataclass(slots=True)
class ValidationSet:
    """Validation set used for training decisions.

    Unlike held_out, this set IS used for:
    - Rollback decisions (detecting performance degradation)
    - Model selection (tracking best model)
    - Threshold adjustments
    - ICE tracking for stopping criteria

    Metrics from this set have model selection bias and should not be
    considered "honest" evaluation metrics.

    Extra positives/negatives from unlabeled_samples can be added for
    tracked samples that need permanent validation monitoring.
    """

    flare_indices: np.ndarray  # Indices in known_flares (positives)
    negative_indices: np.ndarray  # Indices in unlabeled_samples (negatives)
    extra_positive_indices: np.ndarray | None = None  # Indices in unlabeled_samples with label=1
    extra_negative_indices: np.ndarray | None = None  # Additional negatives from unlabeled_samples


@dataclass(slots=True)
class IterationMetrics:
    """Metrics collected at each iteration.

    Note: validation_* metrics are from the validation set used for training
    decisions. These have model selection bias. Honest held-out metrics are
    only computed once at the end in _finalize().
    """

    iteration: int

    # Sizes
    train_total: int
    train_seed: int
    train_pseudo_pos: int
    train_pseudo_neg: int
    train_forced_pos: int = 0
    train_forced_neg: int = 0

    # Effective sizes (with weights)
    effective_pos: float
    effective_neg: float

    # Quality metrics (from validation set - used for decisions)
    validation_recall: float
    validation_precision: float
    validation_auc: float
    validation_f1: float

    # Calibration metrics (from validation set)
    validation_logloss: float = 0.0
    validation_brier: float = 0.0
    validation_ice: float = 0.0  # ICE metric (primary stopping criterion)
    validation_pr_auc: float = 0.0  # PR-AUC (important for imbalanced data)

    # Enrichment
    enrichment_factor: float = 0.0
    estimated_flares_top10k: float = 0.0

    # Thresholds (current)
    pseudo_pos_threshold: float = 0.99
    pseudo_neg_threshold: float = 0.05

    # Changes this iteration
    pseudo_pos_added: int = 0
    pseudo_neg_added: int = 0
    pseudo_removed: int = 0

    # Stability
    n_successful_iters: int = 0
    n_rollbacks_recent: int = 0

    # Time tracking
    elapsed_hours: float = 0.0

    # Special row tracking
    tracked_positive_rowid_prob: float | None = None
    tracked_negative_rowid_prob: float | None = None

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# Model Training (Stub - Replace with your implementation)
# =============================================================================


def train_model(
    features: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray | None = None,
    class_weight: dict[int, float] | None = None,
    config: PipelineConfig | None = None,
    random_state: int = 42,
    plot_file: str | Path | None = None,
    iterations_override: int | None = None,
    early_stopping_rounds_override: int | None = None,
) -> CatBoostClassifier:
    """
    Train a CatBoost classifier with GPU support and auto early stopping.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    labels : np.ndarray
        Binary labels of shape (n_samples,).
    sample_weights : np.ndarray, optional
        Per-sample weights of shape (n_samples,).
    class_weight : dict, optional
        Class weights as {0: weight_0, 1: weight_1}.
    config : PipelineConfig, optional
        Pipeline configuration.
    random_state : int
        Random seed for reproducibility.
    plot_file : str or Path, optional
        Path to save training plot.
    iterations_override : int, optional
        Override config.catboost.iterations (for bootstrap models).
    early_stopping_rounds_override : int, optional
        Override config.catboost.early_stopping_rounds (for bootstrap models).

    Returns
    -------
    CatBoostClassifier
        Trained classifier.
    """
    if config is None:
        config = PipelineConfig()

    # Combine sample_weights with class_weight
    if class_weight is not None and sample_weights is not None:
        # Apply class weight to sample weights
        adjusted_weights = sample_weights.copy()
        for cls, weight in class_weight.items():
            adjusted_weights[labels == cls] *= weight
        sample_weights = adjusted_weights
    elif class_weight is not None:
        # Create sample weights from class weights
        sample_weights = np.ones(len(labels))
        for cls, weight in class_weight.items():
            sample_weights[labels == cls] = weight

    # Determine task type for GPU
    task_type = "GPU" if config.catboost.use_gpu else "CPU"

    # Use overrides if provided, otherwise use config values
    iterations = iterations_override if iterations_override is not None else config.catboost.iterations
    early_stopping_rounds = early_stopping_rounds_override if early_stopping_rounds_override is not None else config.catboost.early_stopping_rounds

    # Build model with calibration-focused loss and native eval_fraction
    model = CatBoostClassifier(
        iterations=iterations,
        depth=config.catboost.depth,
        learning_rate=config.catboost.learning_rate,
        random_seed=random_state,
        verbose=config.catboost.verbose,
        auto_class_weights=None,  # We handle weights manually
        loss_function=config.catboost.loss_function,  # Logloss for calibration
        eval_metric=config.catboost.eval_metric,  # Logloss instead of AUC
        task_type=task_type,
        early_stopping_rounds=early_stopping_rounds,
        eval_fraction=config.catboost.eval_fraction if config.catboost.eval_fraction > 0 else None,
        # Regularization parameters
        l2_leaf_reg=config.catboost.l2_leaf_reg,
        min_data_in_leaf=config.catboost.min_data_in_leaf,
        random_strength=config.catboost.random_strength,
    )

    model.fit(
        features,
        labels,
        sample_weight=sample_weights,
        plot=config.catboost.plot and plot_file is not None,
        plot_file=str(plot_file) if plot_file else None,
    )

    return model


def predict_proba_batched(
    model: CatBoostClassifier,
    features: np.ndarray,
    batch_size: int = 100_000_000,
    desc: str = "Predicting",
) -> np.ndarray:
    """
    Predict probabilities in batches for memory efficiency.

    Parameters
    ----------
    model : CatBoostClassifier
        Trained classifier.
    features : np.ndarray
        Feature matrix.
    batch_size : int
        Number of samples per batch. Default 100M (skips batching for typical datasets).
    desc : str
        Description for progress bar.

    Returns
    -------
    np.ndarray
        Predicted probabilities for positive class.
    """
    n_samples = len(features)

    # Smart bypass: if data fits in one batch, predict directly
    if n_samples <= batch_size:
        return model.predict_proba(features)[:, 1].astype(np.float32)

    proba = np.zeros(n_samples, dtype=np.float32)
    n_batches = (n_samples + batch_size - 1) // batch_size
    for i in tqdmu(range(n_batches), desc=desc, disable=n_batches <= 1):
        start = i * batch_size
        end = min(start + batch_size, n_samples)
        proba[start:end] = model.predict_proba(features[start:end])[:, 1]

    return proba


# =============================================================================
# Feature Extraction Helpers
# =============================================================================


def get_feature_columns(
    df: pl.DataFrame,
    exclude: set[str] | None = None,
) -> list[str]:
    """Get feature column names (excluding id, class, metadata)."""
    if exclude is None:
        exclude = {"id", "class", "ts", "index"}
    return [c for c in df.columns if c not in exclude]


def extract_features_array(
    df: pl.DataFrame,
    indices: np.ndarray | list[int],
    feature_cols: list[str],
    out: np.ndarray | None = None,
) -> np.ndarray:
    """
    Extract feature matrix for given indices.

    Parameters
    ----------
    df : pl.DataFrame
        Source DataFrame.
    indices : np.ndarray | list[int]
        Row indices to extract.
    feature_cols : list[str]
        Feature columns to extract.
    out : np.ndarray, optional
        Preallocated output buffer of shape (len(indices), len(feature_cols)).
        If provided, writes directly to this buffer to avoid allocation.

    Returns
    -------
    np.ndarray
        Feature matrix of shape (len(indices), len(feature_cols)).
    """
    subset = df[indices].select(feature_cols)
    if out is not None:
        # Write directly to preallocated buffer
        out[:] = subset.to_numpy()
        return out
    return subset.to_numpy().astype(np.float32)


def plot_sample(
    sample_index: int,
    source_df: pl.DataFrame,
    output_dir: Path,
    prefix: str,
    config: PipelineConfig,
    action: str = "added",
    dataset=None,
    probability: float | None = None,
    iteration: int | None = None,
) -> None:
    """
    Plot a sample in both raw and cleaned modes.

    Parameters
    ----------
    sample_index : int
        Index in the source DataFrame / dataset.
    source_df : pl.DataFrame
        Source DataFrame (unlabeled_samples or known_flares). Used for ID lookup.
    output_dir : Path
        Base output directory.
    prefix : str
        Subdirectory name (e.g., "pseudo_pos", "pseudo_neg", "top_removal_candidates").
    config : PipelineConfig
        Pipeline configuration.
    action : str
        Action description ("added" or "removed").
    dataset : HuggingFace Dataset, optional
        Original light curve dataset for plotting. If provided, uses
        dataset[sample_index] for plotting instead of source_df.
    probability : float, optional
        P(flare) probability to include in filename (as percentage).
    iteration : int, optional
        Iteration number. If provided, creates directory structure: iter{N}/{prefix}/
    """
    if not VIEW_SERIES_AVAILABLE or not config.plot_samples:
        return

    if dataset is None:
        logger.info(f"No dataset provided for plotting sample {sample_index}, skipping")
        return

    try:
        # Get sample ID if available
        sample_id = source_df[sample_index, "id"] if "id" in source_df.columns else sample_index
        row_num = sample_index

        # Create output subdirectory with new structure: iter{N}/{prefix}/ or just {prefix}/
        if iteration is not None:
            plots_dir = output_dir / "sample_plots" / f"iter{iteration:03d}" / prefix
        else:
            plots_dir = output_dir / "sample_plots" / prefix
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Determine backend
        backend = "plotly" if is_jupyter_notebook() else "matplotlib"

        # Get the light curve data from HuggingFace dataset
        lc_data = dataset[sample_index]

        # Build filename with probability if provided
        prob_str = f"_P{probability*100:.2f}pct" if probability is not None else ""
        cleaned_file = plots_dir / f"{action}_{sample_id}_row{row_num}{prob_str}_cleaned.png"

        # Plot cleaned (title will include P(flare) via view_series if supported)
        title = f"P(flare)={probability*100:.2f}%" if probability is not None else None
        fig = view_series(
            lc_data,
            index=sample_index,
            backend=backend,
            singlepoint_min_outlying_factor=config.plot_singlepoint_min_outlying_factor,
            plot_file=str(cleaned_file),
            title=title,
            show=config.display_sample_plots,
        )

    except (IndexError, KeyError, ValueError, OSError) as e:
        logger.warning(f"Failed to plot sample {sample_index}: {e}")


def report_held_out_metrics(
    targets: np.ndarray,
    probs: np.ndarray,
    iteration: int,
    output_dir: Path,
    config: PipelineConfig,
    preds: np.ndarray | None = None,
    save_charts: bool = True,
    display_charts: bool = False,
) -> dict[str, float]:
    """
    Report held-out metrics using report_model_perf.

    Parameters
    ----------
    targets : np.ndarray
        True binary labels.
    probs : np.ndarray
        Predicted probabilities for positive class.
    iteration : int
        Current iteration number. Use -1 for final held-out evaluation.
    output_dir : Path
        Output directory for charts.
    config : PipelineConfig
        Pipeline configuration.
    preds : np.ndarray, optional
        Predicted labels. If None, derived from probs using threshold 0.5.
    save_charts : bool, default True
        Whether to save performance charts to files. Set False for bootstrap models.
    display_charts : bool, default False
        Whether to display charts in matplotlib window.

    Returns
    -------
    dict[str, float]
        Dictionary of metrics from mlframe:
        roc_auc, pr_auc, calibration_mae, calibration_std, brier_loss, log_loss, ice, class_integral_error
    """
    # Handle final held-out evaluation (iteration=-1)
    if iteration < 0:
        iter_name = "final"
        report_title = "Final Held-Out"
    else:
        iter_name = f"iter_{iteration:03d}"
        report_title = f"Held-Out Iter {iteration}"

    report_params = {
        "report_ndigits": 2,
        "calib_report_ndigits": 2,
        "print_report": False,
        "report_title": report_title,
        "use_weights": True,
        "show_perf_chart": display_charts,  # Controls matplotlib display
    }

    # Set up plot_file if saving charts (file saving is independent of display)
    plot_file = None
    if save_charts:
        charts_dir = output_dir / "perf_charts"
        charts_dir.mkdir(parents=True, exist_ok=True)
        plot_file = join(str(charts_dir), iter_name)

    # Convert 1D probs (positive class) to 2D (n_samples, 2) for report_model_perf
    probs_2d = np.column_stack([1 - probs, probs])

    # Derive predictions if not provided
    if preds is None:
        preds = (probs >= DEFAULT_CLASSIFICATION_THRESHOLD).astype(np.int8)

    # metrics dict is mutated by report_model_perf
    metrics: dict = {}
    _, _ = report_model_perf(
        targets=targets.astype(np.int8),
        columns=None,
        df=None,
        model_name=iter_name,
        model=None,
        target_label_encoder=None,
        preds=preds,
        probs=probs_2d,
        plot_file=plot_file,
        metrics=metrics,
        group_ids=None,
        **report_params,
    )

    # metrics dict is keyed by class label (1 for positive class)
    return metrics.get(1, {})


# =============================================================================
# Pipeline Implementation
# =============================================================================


class ActiveLearningPipeline:
    """
    Zero-Expert Self-Training Pipeline for Stellar Flare Detection.

    This pipeline implements iterative pseudo-labeling with bootstrap consensus
    for rare event detection in astronomical data. Starting from a small set of
    known flares (positive examples) and a large unlabeled dataset, it iteratively
    expands the labeled training set through confident pseudo-labeling.

    Problem Context
    ---------------
    Stellar flare detection is a rare event classification problem with:
    - Extreme class imbalance (~0.1% positive rate in unlabeled data)
    - Small positive seed set (tens of known flares from literature)
    - Large unlabeled pool (tens of millions of observations)
    - High cost of false negatives (missing real flares for scientific study)

    Data Split Strategy
    -------------------
    Known flares are split into three disjoint sets:

    - **Training (50 flares + 1000 negatives)**: Initial seed for model training.
      Grows through pseudo-labeling during training.

    - **Validation (25 flares + 5000 negatives)**: Used for ALL training decisions:
      rollback detection, best model selection, threshold adjustments, and stopping
      criteria. Metrics from this set have MODEL SELECTION BIAS and should not be
      considered unbiased estimates of true performance.

    - **Held-out (24 flares + 5000 negatives)**: NEVER touched during training.
      Only evaluated once at the very end (in _finalize) to provide HONEST,
      UNBIASED evaluation metrics. This is the only trustworthy performance estimate.

    Algorithm Overview
    ------------------
    Each iteration performs these phases:

    1. **Validation & Early Stopping**: Evaluate on validation set, detect degradation,
       trigger rollback if ICE metric increases significantly.

    2. **Bootstrap Ensemble Training**: Train N bootstrap models (default 5) on
       resampled training data. This provides consensus estimation and enables
       out-of-bag (OOB) evaluation.

    3. **Full Dataset Prediction**: Predict probabilities on entire unlabeled
       dataset using main model and all bootstrap models.

    4. **Pseudo-label Negatives (Aggressive)**: Select high-confidence negatives
       (P < 0.05) where majority of bootstrap models agree. Add many per iteration
       (default 100) since false negatives are low-risk.

    5. **Pseudo-label Positives (Conservative)**: Select very high-confidence
       positives (P > 0.99) where ALL bootstrap models agree with low variance
       (std < 0.05). Add few per iteration (default 10) to avoid confirmation bias.

    6. **Review Pseudo-labels**: Re-evaluate existing pseudo-labels with current
       model. Remove any that the model no longer confidently predicts, indicating
       they may have been mistakes.

    7. **Retrain Main Model**: Train fresh model on updated labeled set with
       curriculum-based sample weights.

    8. **Adaptive Thresholds**: After consecutive successful iterations, relax
       thresholds to explore more. After rollback, tighten thresholds.

    Key Design Principles
    ---------------------
    - **Asymmetric Pseudo-labeling**: Aggressive for negatives (abundant, low-risk),
      conservative for positives (rare, high-risk of confirmation bias). This
      asymmetry reflects the base rate: random samples are 99.9% likely negative.

    - **Bootstrap Consensus**: Multiple models must agree before accepting a
      pseudo-label. For positives, ALL models must agree; for negatives, majority
      suffices. This reduces variance and detects unstable predictions.

    - **Three-Way Data Split**: Validation set for training decisions (biased),
      held-out set for honest final evaluation (unbiased). This ensures reported
      final metrics are trustworthy.

    - **Adaptive Thresholds**: The pipeline self-regulates: thresholds tighten after
      degradation (fewer pseudo-labels, more conservative) and relax after stable
      iterations (more exploration). This prevents both overfitting and stagnation.

    - **Curriculum Learning**: Sample weights depend on both confidence and model
      maturity (current recall). Early iterations use only very confident pseudo-labels;
      mature models can learn from harder examples.

    - **Rollback on Degradation**: If validation ICE (calibration error) increases
      beyond threshold, rollback to best checkpoint and tighten thresholds. This
      prevents runaway confirmation bias.

    Stopping Criteria
    -----------------
    The pipeline stops when any of these conditions is met:
    - Time limit reached (default 5 hours)
    - Target ICE achieved (if configured)
    - Metrics plateau (no improvement for 10 iterations)
    - Instability detected (>3 rollbacks in 10 iterations)
    - Success criteria met (recall, precision, enrichment targets achieved)

    Limitations & Caveats
    ---------------------
    - Validation metrics have model selection bias (used for decisions during training)
    - Honest held-out metrics are only available at the very end
    - Pseudo-positives risk confirmation bias if thresholds are too loose
    - Assumes unlabeled negatives are true negatives (base rate assumption)
    - Bootstrap consensus adds computational cost (N models per iteration)

    References
    ----------
    - Settles, B. (2012). Active Learning. Morgan & Claypool Publishers.
    - Xie et al. (2020). Self-Training with Noisy Student. CVPR.
    - Arazo et al. (2020). Pseudo-Labeling and Confirmation Bias. IJCNN.
    - Lee (2013). Pseudo-Label: The Simple and Efficient Semi-Supervised Learning.
    - Bengio et al. (2009). Curriculum Learning. ICML.

    See Also
    --------
    run_active_learning_pipeline : Convenience function for running the pipeline.
    PipelineConfig : Configuration dataclass with all tunable parameters.
    """

    def __init__(
        self,
        unlabeled_samples: pl.DataFrame,
        known_flares: pl.DataFrame,
        config: PipelineConfig | None = None,
        output_dir: Path | str = "active_learning_output",
        random_state: int = 42,
        unlabeled_dataset=None,
        known_flares_dataset=None,
        freaky_held_out_indices: list[int] | None = None,
    ):
        """
        Initialize the pipeline.

        Parameters
        ----------
        unlabeled_samples : pl.DataFrame
            Large unlabeled dataset (features for the unlabeled pool).
            Expected columns: feature columns, optionally 'id'.
        known_flares : pl.DataFrame
            Known flares dataset (samples with class=1).
            Expected columns: feature columns, 'class', optionally 'id'.
        config : PipelineConfig, optional
            Pipeline configuration. Uses defaults if not provided.
        output_dir : Path or str
            Directory for saving checkpoints and results.
        random_state : int
            Random seed for reproducibility.
        unlabeled_dataset : HuggingFace Dataset, optional
            Original light curve data for unlabeled_samples, used for plotting.
            Access pattern: dataset[row_index]
        known_flares_dataset : HuggingFace Dataset, optional
            Original light curve data for known_flares, used for plotting.
            Access pattern: dataset[row_index]
        freaky_held_out_indices : list[int], optional
            Row indices in known_flares to exclude from split and report separately.
        """
        self.unlabeled_samples = unlabeled_samples
        self.known_flares = known_flares
        self.config = config or PipelineConfig()
        self.unlabeled_dataset = unlabeled_dataset
        self.known_flares_dataset = known_flares_dataset

        # Add timestamp to output_dir to prevent overwriting previous runs
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"{output_dir}_{run_timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")

        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        # Validate input
        self._validate_inputs()

        # Get feature columns (same for both datasets)
        self._all_feature_cols = get_feature_columns(self.known_flares, self.config.exclude_columns)
        logger.info(f"Total available: {len(self._all_feature_cols)} feature columns")

        # Feature warmup: start with subset, gradually expand
        if self.config.feature_warmup_enabled:
            prefix = self.config.feature_warmup_initial_prefix
            self._warmup_initial_features = [f for f in self._all_feature_cols if f.startswith(prefix)]
            self._warmup_additional_features = [f for f in self._all_feature_cols if not f.startswith(prefix)]
            # Sort additional features for deterministic expansion order
            self._warmup_additional_features.sort()
            self.feature_cols = self._warmup_initial_features.copy()
            logger.info(
                f"Feature warmup enabled: starting with {len(self._warmup_initial_features)} '{prefix}' features, "
                f"{len(self._warmup_additional_features)} to add gradually"
            )
        else:
            self._warmup_initial_features = []
            self._warmup_additional_features = []
            self.feature_cols = self._all_feature_cols

        # State variables
        self.labeled_train: list[LabeledSample] = []
        self.validation: ValidationSet | None = None  # For rollback/model selection
        self.held_out: HeldOutSet | None = None  # Honest eval (never used for decisions)
        self.model: CatBoostClassifier | None = None
        self.bootstrap_models: list[CatBoostClassifier] = []
        self.bootstrap_indices_list: list[np.ndarray] = []  # For OOB computation

        # Metrics tracking (using validation set for decisions)
        self.metrics_history: list[IterationMetrics] = []
        self.best_checkpoint: Checkpoint | None = None
        self.best_validation_recall: float = 0.0
        self.best_validation_ice: float = 1.0  # Lower is better
        self.prev_validation_recall: float = 0.0
        self.prev_validation_precision: float = 0.0
        self.prev_validation_ice: float = 1.0  # ICE starts high (lower is better)
        # Tracked sample probabilities for decision-making (pos should be high, neg should be low)
        self.prev_tracked_pos_prob: float | None = None
        self.prev_tracked_neg_prob: float | None = None
        self.n_successful_iters: int = 0
        self.rollback_history: list[int] = []  # Iteration numbers of rollbacks

        # Current thresholds (mutable)
        self.pseudo_pos_threshold = self.config.thresholds.pseudo_pos_threshold
        self.pseudo_neg_threshold = self.config.thresholds.pseudo_neg_threshold
        self.max_pseudo_pos_per_iter = self.config.thresholds.max_pseudo_pos_per_iter

        # Time tracking
        self.start_time: float | None = None

        # Ban list: prevents re-adding samples with same label for N iterations after rollback
        # Key: (index, label) tuple, Value: expires_at_iteration
        self._ban_list: dict[tuple[int, int], int] = {}
        # O(1) lookup sets for banned indices by label (derived from _ban_list)
        self._banned_pos_set: set[int] = set()
        self._banned_neg_set: set[int] = set()

        # Cached feature views for efficiency (zero-copy pandas view of polars DataFrame)
        self._unlabeled_view: np.ndarray | None = None

        # Track row indices (direct row indices in unlabeled_samples, not ID column values)
        self._tracked_positive_rowid_index: int | None = None
        self._tracked_negative_rowid_index: int | None = None
        n_unlabeled = len(self.unlabeled_samples)

        if self.config.tracked_positive_rowid is not None:
            if 0 <= self.config.tracked_positive_rowid < n_unlabeled:
                self._tracked_positive_rowid_index = self.config.tracked_positive_rowid
                logger.info(f"Tracking positive row index {self._tracked_positive_rowid_index}")
            else:
                logger.warning(f"tracked_positive_rowid {self.config.tracked_positive_rowid} out of bounds (0-{n_unlabeled - 1})")

        if self.config.tracked_negative_rowid is not None:
            if 0 <= self.config.tracked_negative_rowid < n_unlabeled:
                self._tracked_negative_rowid_index = self.config.tracked_negative_rowid
                logger.info(f"Tracking negative row index {self._tracked_negative_rowid_index}")
            else:
                logger.warning(f"tracked_negative_rowid {self.config.tracked_negative_rowid} out of bounds (0-{n_unlabeled - 1})")

        # Running counts for efficiency (avoid O(n) scans)
        self._source_counts: dict[SampleSource, int] = {
            SampleSource.SEED: 0,
            SampleSource.PSEUDO_POS: 0,
            SampleSource.PSEUDO_NEG: 0,
            SampleSource.FORCED_POS: 0,
            SampleSource.FORCED_NEG: 0,
        }
        self._effective_pos: float = 0.0
        self._effective_neg: float = 0.0
        self._labeled_unlabeled_indices: set[int] = set()
        self._labeled_known_flare_indices: set[int] = set()

        # Training data cache (avoids rebuilding features/labels/weights each time)
        self._training_data_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._training_data_dirty: bool = True

        # Validation/held-out feature caches (populated once in initialize(), never change)
        self._validation_features_cache: tuple[np.ndarray, np.ndarray] | None = None  # (flare_features, neg_features)
        self._held_out_features_cache: tuple[np.ndarray, np.ndarray] | None = None  # (flare_features, neg_features)

        # Freaky held-out samples (excluded from split, reported separately)
        self.freaky_held_out_indices = freaky_held_out_indices or []
        self.freaky_set: ValidationSet | None = None  # Initialized during initialize() if freaky indices provided
        self._freaky_features_cache: tuple[np.ndarray, np.ndarray] | None = None
        if self.freaky_held_out_indices:
            logger.info(f"Freaky held-out indices: {len(self.freaky_held_out_indices)} samples will be excluded from split and reported separately")

        # Cached exclusion set for pseudo-labeling (held-out indices never change)
        self._held_out_neg_set: set[int] = set()

        # Cached exclusion set (union of labeled + held-out, invalidated on changes)
        self._exclusion_set_cache: set[int] | None = None

        # Cached exclusion array (numpy version of exclusion set, invalidated with set)
        self._exclusion_array_cache: np.ndarray | None = None

        # Cached tracked row features (indices never change, cache features once)
        self._tracked_features_cache: np.ndarray | None = None

    def _validate_inputs(self) -> None:
        """Validate input DataFrames."""
        n_flares = len(self.known_flares)
        data = self.config.data
        required_flares = data.n_train_flares + data.n_validation_flares + data.n_held_out_flares
        if n_flares < required_flares:
            raise ValueError(f"Need at least {required_flares} known flares, got {n_flares}")

        required_negs = data.n_train_neg_init + data.n_validation_neg + data.n_held_out_neg
        if len(self.unlabeled_samples) < required_negs:
            raise ValueError(f"unlabeled_samples too small for required negative samples ({required_negs})")

        logger.info(f"unlabeled_samples: {len(self.unlabeled_samples):,} samples")
        logger.info(f"known_flares: {len(self.known_flares)} known flares")

    def _add_sample(self, sample: LabeledSample) -> None:
        """Add a sample to labeled_train with running count updates."""
        self.labeled_train.append(sample)
        self._source_counts[sample.source] += 1
        if sample.label == 1:
            self._effective_pos += sample.weight
        else:
            self._effective_neg += sample.weight
        if sample.from_known_flares:
            self._labeled_known_flare_indices.add(sample.index)
        else:
            self._labeled_unlabeled_indices.add(sample.index)
            self._exclusion_set_cache = None  # Invalidate cache
            self._exclusion_array_cache = None  # Invalidate array cache
        self._training_data_dirty = True

    def _add_samples_batch(self, samples: list[LabeledSample]) -> None:
        """Add multiple samples with batched set operations."""
        if not samples:
            return

        # Extend list once
        self.labeled_train.extend(samples)

        # Vectorized weight sums using numpy
        labels = np.array([s.label for s in samples], dtype=np.int8)
        weights = np.array([s.weight for s in samples], dtype=np.float32)
        self._effective_pos += float(weights[labels == 1].sum())
        self._effective_neg += float(weights[labels == 0].sum())

        # Count sources with Counter (O(n) but single pass)
        source_counts = Counter(s.source for s in samples)
        for source, count in source_counts.items():
            self._source_counts[source] += count

        # Batch set updates (more efficient than individual add() calls)
        known_flare_idx = [s.index for s in samples if s.from_known_flares]
        unlabeled_idx = [s.index for s in samples if not s.from_known_flares]
        if known_flare_idx:
            self._labeled_known_flare_indices.update(known_flare_idx)
        if unlabeled_idx:
            self._labeled_unlabeled_indices.update(unlabeled_idx)
            self._exclusion_set_cache = None  # Invalidate cache
            self._exclusion_array_cache = None  # Invalidate array cache

        self._training_data_dirty = True

    def _remove_sample(self, idx: int) -> LabeledSample:
        """Remove a sample from labeled_train by index with running count updates."""
        sample = self.labeled_train.pop(idx)
        self._source_counts[sample.source] -= 1
        if sample.label == 1:
            self._effective_pos -= sample.weight
        else:
            self._effective_neg -= sample.weight
        if sample.from_known_flares:
            self._labeled_known_flare_indices.discard(sample.index)
        else:
            self._labeled_unlabeled_indices.discard(sample.index)
            self._exclusion_set_cache = None  # Invalidate cache
            self._exclusion_array_cache = None  # Invalidate array cache
        self._training_data_dirty = True
        return sample

    def _get_labeled_indices(self) -> tuple[set[int], set[int]]:
        """Get sets of already labeled indices (unlabeled_samples, known_flares)."""
        return self._labeled_unlabeled_indices, self._labeled_known_flare_indices

    def _build_pseudo_label_exclusion_set(self) -> set[int]:
        """
        Build set of unlabeled_samples indices to exclude from pseudo-labeling.

        Excludes:
        - Already labeled samples (from _labeled_unlabeled_indices)
        - Held-out negative samples (cached in _held_out_neg_set)
        - Tracked samples (used for rollback decisions - must stay unlabeled)

        Returns
        -------
        set[int]
            Indices in unlabeled_samples that should not receive pseudo-labels.

        Notes
        -----
        Result is cached and invalidated when _labeled_unlabeled_indices changes.
        """
        if self._exclusion_set_cache is None:
            exclusion = self._labeled_unlabeled_indices | self._held_out_neg_set
            # Exclude tracked samples from pseudo-labeling to preserve rollback sanity check
            if self.config.tracked_positive_rowid is not None:
                exclusion.add(self.config.tracked_positive_rowid)
            if self.config.tracked_negative_rowid is not None:
                exclusion.add(self.config.tracked_negative_rowid)
            self._exclusion_set_cache = exclusion
        return self._exclusion_set_cache

    def _build_pseudo_label_exclusion_array(self) -> np.ndarray:
        """
        Build numpy array of unlabeled_samples indices to exclude from pseudo-labeling.

        This is the array version of _build_pseudo_label_exclusion_set, cached
        to avoid repeated set→list→array conversions. More efficient for np.isin().

        Returns
        -------
        np.ndarray
            Indices in unlabeled_samples that should not receive pseudo-labels.

        Notes
        -----
        Result is cached and invalidated when _labeled_unlabeled_indices changes.
        """
        if self._exclusion_array_cache is None:
            exclusion_set = self._build_pseudo_label_exclusion_set()
            # np.fromiter is faster than np.array(list(...)) for large sets
            self._exclusion_array_cache = (
                np.fromiter(exclusion_set, dtype=np.int64, count=len(exclusion_set)) if exclusion_set else np.array([], dtype=np.int64)
            )
        return self._exclusion_array_cache

    def _build_training_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build feature matrix, labels, and weights from labeled_train (batched by source)."""
        # Return cached data if available and not dirty
        if not self._training_data_dirty and self._training_data_cache is not None:
            return self._training_data_cache

        n_samples = len(self.labeled_train)
        features = np.zeros((n_samples, len(self.feature_cols)), dtype=np.float32)

        # Vectorized labels and weights extraction
        labels = np.array([s.label for s in self.labeled_train], dtype=np.int32)
        weights = np.array([s.weight for s in self.labeled_train], dtype=np.float32)

        # Build masks for batched feature extraction
        from_known_flares_mask = np.array([s.from_known_flares for s in self.labeled_train], dtype=bool)
        oos_positions = np.where(from_known_flares_mask)[0]
        big_positions = np.where(~from_known_flares_mask)[0]

        # Batch extract from known_flares
        if len(oos_positions) > 0:
            oos_source_indices = [self.labeled_train[i].index for i in oos_positions]
            oos_features = extract_features_array(self.known_flares, oos_source_indices, self.feature_cols)
            features[oos_positions] = oos_features

        # Batch extract from unlabeled_samples
        if len(big_positions) > 0:
            big_source_indices = [self.labeled_train[i].index for i in big_positions]
            big_features = extract_features_array(self.unlabeled_samples, big_source_indices, self.feature_cols)
            features[big_positions] = big_features

        # Cache the result
        self._training_data_cache = (features, labels, weights)
        self._training_data_dirty = False

        return features, labels, weights

    def _compute_metrics_for_set(
        self,
        flare_indices: np.ndarray,
        negative_indices: np.ndarray,
        iteration: int = 0,
        cached_features: tuple[np.ndarray, np.ndarray] | None = None,
        save_charts: bool = True,
        extra_positive_indices: np.ndarray | None = None,
        extra_negative_indices: np.ndarray | None = None,
    ) -> dict[str, float]:
        """
        Compute classification metrics on a labeled set using mlframe.

        Shared implementation used by both validation and held-out evaluation.

        Parameters
        ----------
        flare_indices : np.ndarray
            Indices of positive samples in known_flares.
        negative_indices : np.ndarray
            Indices of negative samples in unlabeled_samples.
        iteration : int
            Current iteration number for chart naming.
        cached_features : tuple[np.ndarray, np.ndarray], optional
            Pre-cached (flare_features, neg_features) to avoid re-extraction.
        save_charts : bool, default True
            Whether to save performance charts to files.
        extra_positive_indices : np.ndarray, optional
            Additional positive samples from unlabeled_samples (e.g., tracked positives).
        extra_negative_indices : np.ndarray, optional
            Additional negative samples from unlabeled_samples (e.g., tracked negatives).

        Returns
        -------
        dict[str, float]
            Metrics from mlframe (mapped to expected keys):
            recall, precision, f1, auc, logloss, brier, pr_auc, ice, etc.
        """
        if self.model is None:
            return {"recall": 0.0, "precision": 0.0, "auc": 0.5, "f1": 0.0, "logloss": 1.0, "brier": 0.25, "ice": 1.0}

        # Use cached features if provided, otherwise extract
        if cached_features is not None:
            flare_features, neg_features = cached_features
        else:
            flare_features = extract_features_array(self.known_flares, flare_indices, self.feature_cols)
            neg_features = extract_features_array(self.unlabeled_samples, negative_indices, self.feature_cols)

        flare_proba = self.model.predict_proba(flare_features)[:, 1]
        flare_labels = np.ones(len(flare_indices), dtype=np.int32)

        neg_proba = self.model.predict_proba(neg_features)[:, 1]
        neg_labels = np.zeros(len(negative_indices), dtype=np.int32)

        # Collect all probas and labels
        all_proba_parts = [flare_proba, neg_proba]
        all_labels_parts = [flare_labels, neg_labels]

        # Add extra positives from unlabeled_samples (e.g., tracked positive)
        if extra_positive_indices is not None and len(extra_positive_indices) > 0:
            extra_pos_features = extract_features_array(self.unlabeled_samples, extra_positive_indices, self.feature_cols)
            extra_pos_proba = self.model.predict_proba(extra_pos_features)[:, 1]
            extra_pos_labels = np.ones(len(extra_positive_indices), dtype=np.int32)
            all_proba_parts.append(extra_pos_proba)
            all_labels_parts.append(extra_pos_labels)

        # Add extra negatives from unlabeled_samples (e.g., tracked negative)
        if extra_negative_indices is not None and len(extra_negative_indices) > 0:
            extra_neg_features = extract_features_array(self.unlabeled_samples, extra_negative_indices, self.feature_cols)
            extra_neg_proba = self.model.predict_proba(extra_neg_features)[:, 1]
            extra_neg_labels = np.zeros(len(extra_negative_indices), dtype=np.int32)
            all_proba_parts.append(extra_neg_proba)
            all_labels_parts.append(extra_neg_labels)

        # Combined
        all_proba = np.concatenate(all_proba_parts)
        all_labels = np.concatenate(all_labels_parts)
        all_preds = (all_proba >= DEFAULT_CLASSIFICATION_THRESHOLD).astype(np.int8)

        # Get all metrics from mlframe (display controlled by config.display_perf_charts)
        mlframe_metrics = report_held_out_metrics(
            all_labels,
            all_proba,
            iteration,
            self.output_dir,
            self.config,
            preds=all_preds,
            save_charts=save_charts,
            display_charts=self.config.display_perf_charts,
        )

        # Map mlframe keys to our expected keys
        return {
            "recall": mlframe_metrics.get("recall", 0.0),
            "precision": mlframe_metrics.get("precision", 0.0),
            "f1": mlframe_metrics.get("f1", 0.0),
            "auc": mlframe_metrics.get("roc_auc", 0.5),
            "logloss": mlframe_metrics.get("log_loss", 1.0),
            "brier": mlframe_metrics.get("brier_loss", 0.25),
            "pr_auc": mlframe_metrics.get("pr_auc", 0.0),
            "ice": mlframe_metrics.get("ice", 1.0),
            "calibration_mae": mlframe_metrics.get("calibration_mae", 0.0),
            "calibration_std": mlframe_metrics.get("calibration_std", 0.0),
        }

    def _compute_validation_metrics(self, iteration: int = 0) -> dict[str, float]:
        """
        Compute metrics on validation set (used for training decisions).

        This set IS used for rollback decisions, model selection, and threshold
        adjustments. Metrics from this set have model selection bias.

        Includes tracked positive/negative samples if configured.
        """
        if self.validation is None or self.model is None:
            return {"recall": 0.0, "precision": 0.0, "auc": 0.5, "f1": 0.0, "logloss": 1.0, "brier": 0.25, "ice": 1.0}

        return self._compute_metrics_for_set(
            self.validation.flare_indices,
            self.validation.negative_indices,
            iteration=iteration,
            cached_features=self._validation_features_cache,
            save_charts=True,
            extra_positive_indices=self.validation.extra_positive_indices,
            extra_negative_indices=self.validation.extra_negative_indices,
        )

    def _compute_held_out_metrics_final(self) -> dict[str, float]:
        """
        Compute metrics on the truly held-out set (ONLY called in _finalize).

        This method is called exactly once at the end of training to provide
        honest, unbiased evaluation metrics. The held-out set was NEVER used
        for any training decisions (rollback, model selection, thresholds).
        """
        if self.held_out is None or self.model is None or len(self.held_out.flare_indices) == 0:
            return {"recall": 0.0, "precision": 0.0, "auc": 0.5, "f1": 0.0, "logloss": 1.0, "brier": 0.25, "ice": 1.0}

        return self._compute_metrics_for_set(
            self.held_out.flare_indices,
            self.held_out.negative_indices,
            iteration=-1,  # -1 indicates final held-out evaluation
            cached_features=self._held_out_features_cache,
            save_charts=True,
        )

    def _compute_freaky_metrics(self, iteration: int) -> dict[str, float] | None:
        """
        Compute metrics for freaky held-out samples.

        These are known flares excluded from the train/val/held-out split
        for diagnostic purposes. Uses the same metric computation as validation
        for consistent comparison.

        Parameters
        ----------
        iteration : int
            Current iteration number (for logging).

        Returns
        -------
        dict[str, float] | None
            Metrics dict with full metrics (recall, precision, AUC, PR-AUC, ICE, etc.)
            Returns None if no freaky samples configured.
        """
        if self.freaky_set is None or self._freaky_features_cache is None or self.model is None:
            return None

        # Use same computation as validation (save_charts=False to avoid clutter)
        metrics = self._compute_metrics_for_set(
            self.freaky_set.flare_indices,
            self.freaky_set.negative_indices,
            iteration=iteration,
            cached_features=self._freaky_features_cache,
            save_charts=False,
        )

        # Log in same format as validation
        logger.info(f"  {self._format_metrics_log(metrics, 'Freaky: ')}")

        return metrics

    def _save_checkpoint(self, iteration: int, metrics: dict) -> Checkpoint:
        """Save model checkpoint."""
        model_path = self.output_dir / f"model_iter_{iteration:03d}.joblib"
        joblib.dump(self.model, model_path)

        # Convert ban list keys to strings for JSON compatibility
        ban_list_serializable = {str(k): v for k, v in self._ban_list.items()}

        checkpoint = Checkpoint(
            iteration=iteration,
            model_path=model_path,
            labeled_train=self.labeled_train.copy(),
            metrics=metrics,
            config_snapshot={
                "pseudo_pos_threshold": self.pseudo_pos_threshold,
                "pseudo_neg_threshold": self.pseudo_neg_threshold,
                "max_pseudo_pos_per_iter": self.max_pseudo_pos_per_iter,
            },
            ban_list=ban_list_serializable,
        )

        checkpoint_meta_path = self.output_dir / f"checkpoint_iter_{iteration:03d}.json"
        with open(checkpoint_meta_path, "w") as f:
            json.dump(
                {
                    "iteration": checkpoint.iteration,
                    "model_path": str(checkpoint.model_path),
                    "metrics": checkpoint.metrics,
                    "config_snapshot": checkpoint.config_snapshot,
                    "n_labeled_samples": len(checkpoint.labeled_train),
                    "n_banned_samples": len(checkpoint.ban_list),
                },
                f,
                indent=2,
            )

        return checkpoint

    def _reset_running_counts(self) -> None:
        """Reset all running counts to zero. Called before rebuilding from checkpoint."""
        self._source_counts = {
            SampleSource.SEED: 0,
            SampleSource.PSEUDO_POS: 0,
            SampleSource.PSEUDO_NEG: 0,
            SampleSource.FORCED_POS: 0,
            SampleSource.FORCED_NEG: 0,
        }
        self._effective_pos = 0.0
        self._effective_neg = 0.0
        self._labeled_unlabeled_indices = set()
        self._labeled_known_flare_indices = set()

    def _load_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Load model from checkpoint and rebuild running counts."""
        self.model = joblib.load(checkpoint.model_path)
        self.pseudo_pos_threshold = checkpoint.config_snapshot["pseudo_pos_threshold"]
        self.pseudo_neg_threshold = checkpoint.config_snapshot["pseudo_neg_threshold"]
        self.max_pseudo_pos_per_iter = checkpoint.config_snapshot["max_pseudo_pos_per_iter"]

        # Rebuild labeled_train and counts using _add_sample for consistency
        self._reset_running_counts()
        self.labeled_train = []
        for sample in checkpoint.labeled_train:
            self._add_sample(sample)

    def _count_by_source(self) -> dict[SampleSource, int]:
        """Count samples by source. Returns internal dict - do not modify."""
        return self._source_counts

    def _compute_effective_sizes(self) -> tuple[float, float]:
        """Compute effective class sizes (returns cached sums of weights)."""
        return self._effective_pos, self._effective_neg

    def _get_elapsed_hours(self) -> float:
        """Get elapsed time since pipeline start in hours."""
        if self.start_time is None:
            return 0.0
        return (time.time() - self.start_time) / 3600.0

    def _tighten_thresholds(self) -> None:
        """Tighten pseudo-labeling thresholds after degradation or rollback."""
        thresholds = self.config.thresholds
        self.pseudo_pos_threshold = min(
            MAX_PSEUDO_POS_THRESHOLD_BOUND,
            self.pseudo_pos_threshold + thresholds.tighten_pos_delta,
        )
        self.pseudo_neg_threshold = max(
            MIN_PSEUDO_NEG_THRESHOLD_BOUND,
            self.pseudo_neg_threshold - thresholds.tighten_neg_delta,
        )
        self.max_pseudo_pos_per_iter = max(
            thresholds.min_pseudo_pos_per_iter,
            self.max_pseudo_pos_per_iter - thresholds.tighten_max_pos_delta,
        )
        logger.info(f"Thresholds tightened: pos>{self.pseudo_pos_threshold:.3f}, neg<{self.pseudo_neg_threshold:.3f}")

    def _add_to_ban_list(self, samples: list[LabeledSample], current_iter: int) -> None:
        """
        Add samples to the ban list after rollback.

        Banned samples cannot be re-added with the same label for N iterations.

        Parameters
        ----------
        samples : list[LabeledSample]
            Samples to ban.
        current_iter : int
            Current iteration number.
        """
        ban_duration = self.config.thresholds.ban_iterations
        expires_at = current_iter + ban_duration

        for sample in samples:
            key = (sample.index, sample.label)
            self._ban_list[key] = expires_at
            # Update O(1) lookup sets
            if sample.label == 1:
                self._banned_pos_set.add(sample.index)
            else:
                self._banned_neg_set.add(sample.index)

        if samples:
            logger.info(f"Banned {len(samples)} samples until iteration {expires_at}")

    def _cleanup_expired_bans(self, current_iter: int) -> int:
        """
        Remove expired bans from the ban list.

        Parameters
        ----------
        current_iter : int
            Current iteration number.

        Returns
        -------
        int
            Number of bans still active after cleanup.
        """
        expired_keys = [key for key, expires_at in self._ban_list.items() if expires_at <= current_iter]
        for key in expired_keys:
            idx, lbl = key
            del self._ban_list[key]
            # Remove from O(1) lookup sets
            if lbl == 1:
                self._banned_pos_set.discard(idx)
            else:
                self._banned_neg_set.discard(idx)

        if expired_keys:
            logger.info(f"Expired {len(expired_keys)} bans, {len(self._ban_list)} still active")

        return len(self._ban_list)

    def _update_feature_warmup(self, iteration: int) -> bool:
        """
        Update active features based on warmup schedule.

        Gradually expands feature set from initial prefix-matched features
        to full feature set over multiple iterations.

        Parameters
        ----------
        iteration : int
            Current iteration number.

        Returns
        -------
        bool
            True if features were expanded this iteration, False otherwise.
        """
        if not self.config.feature_warmup_enabled:
            return False

        # Force full features after specified iteration
        if self.config.feature_warmup_full_features_iter is not None:
            if iteration >= self.config.feature_warmup_full_features_iter:
                if len(self.feature_cols) < len(self._all_feature_cols):
                    self.feature_cols = self._all_feature_cols
                    logger.info(f"Feature warmup: forced full features at iter {iteration} ({len(self.feature_cols)} features)")
                    return True
                return False

        # Check if already using all features
        if len(self.feature_cols) >= len(self._all_feature_cols):
            return False

        # Only expand at specified intervals
        if iteration % self.config.feature_warmup_expansion_interval != 0:
            return False

        # Calculate how many additional features to add
        n_additional_total = len(self._warmup_additional_features)
        n_additional_current = len(self.feature_cols) - len(self._warmup_initial_features)
        n_to_add = max(1, int(n_additional_total * self.config.feature_warmup_expansion_rate))

        # Get next batch of features
        start_idx = n_additional_current
        end_idx = min(start_idx + n_to_add, n_additional_total)

        if start_idx >= n_additional_total:
            return False  # All additional features already added

        new_features = self._warmup_additional_features[start_idx:end_idx]
        self.feature_cols = self.feature_cols + new_features

        pct_complete = len(self.feature_cols) / len(self._all_feature_cols) * 100
        logger.info(
            f"Feature warmup: added {len(new_features)} features at iter {iteration} "
            f"({len(self.feature_cols)}/{len(self._all_feature_cols)} = {pct_complete:.1f}%)"
        )
        return True

    def _is_banned(self, index: int, label: int, current_iter: int) -> bool:
        """
        Check if a sample is banned from being added with the given label.

        Parameters
        ----------
        index : int
            Sample index in the source DataFrame.
        label : int
            Label to check (0 or 1).
        current_iter : int
            Current iteration number.

        Returns
        -------
        bool
            True if the sample is banned, False otherwise.
        """
        key = (index, label)
        if key not in self._ban_list:
            return False
        return self._ban_list[key] > current_iter

    def _get_banned_indices(self, label: int, _current_iter: int) -> set[int]:
        """
        Get set of banned indices for a given label.

        O(1) lookup using pre-computed sets that are maintained by
        _add_to_ban_list() and _cleanup_expired_bans().

        Parameters
        ----------
        label : int
            Label to check (0 for negatives, 1 for positives).
        _current_iter : int
            Unused - kept for API compatibility. Sets are cleaned each iteration.

        Returns
        -------
        set[int]
            Set of sample indices that are banned for this label.
        """
        return self._banned_pos_set if label == 1 else self._banned_neg_set

    def _get_unlabeled_features_for_prediction(self) -> np.ndarray | None:
        """
        Get feature view for unlabeled_samples using zero-copy pandas view.

        Uses get_pandas_view_of_polars_df from mlframe to create a zero-copy
        pandas view of the polars DataFrame. This avoids duplicating the ~20GB
        feature matrix in memory.

        The view is cached after first creation. Do NOT call .values on the
        result - that triggers pandas block consolidation and copies memory!

        Returns
        -------
        np.ndarray or None
            Feature array (zero-copy view), or None if unavailable.
        """
        if self._unlabeled_view is not None:
            return self._unlabeled_view

        # Report RAM before
        ram_before = psutil.Process().memory_info().rss / 1024**3
        logger.info(f"Creating zero-copy view of unlabeled_samples for prediction... (RAM: {ram_before:.2f} GB)")

        if MLFRAME_AVAILABLE and get_pandas_view_of_polars_df is not None:
            # Use zero-copy pandas view - do NOT call .values (causes copy!)
            self._unlabeled_view = get_pandas_view_of_polars_df(self.unlabeled_samples.select(self.feature_cols))
            # Report RAM after
            ram_after = psutil.Process().memory_info().rss / 1024**3
            logger.info(
                f"Zero-copy view created: shape={self._unlabeled_view.shape}, "
                f"RAM: {ram_before:.2f} -> {ram_after:.2f} GB (delta: {ram_after - ram_before:+.2f} GB)"
            )
        else:
            logger.warning("get_pandas_view_of_polars_df not available, falling back to batched prediction")
            return None

        return self._unlabeled_view

    def clear_caches(self) -> None:
        """Clear cached data to free memory."""
        self._unlabeled_view = None
        self.bootstrap_models = []
        self.bootstrap_indices_list = []
        clean_ram()
        logger.info("Caches cleared")

    def _predict_unlabeled_features_batched(self, model: CatBoostClassifier | None = None) -> np.ndarray:
        """
        Predict on unlabeled_samples using cached view or batched extraction.

        Priority order:
        1. Zero-copy pandas view (single call if fits in batch_size, else batched)
        2. Batch-by-batch extraction (fallback)

        Parameters
        ----------
        model : CatBoostClassifier, optional
            Model to use. Defaults to self.model.

        Returns
        -------
        np.ndarray
            Predictions (probabilities) for all samples in unlabeled_samples.
        """
        if model is None:
            model = self.model
        if model is None:
            raise ValueError("No model available for prediction")

        n_total = len(self.unlabeled_samples)
        batch_size = self.config.prediction_batch_size

        # Option 1: Try cached zero-copy view
        features_view = self._get_unlabeled_features_for_prediction()
        if features_view is not None:
            # Smart bypass: if data fits in one batch, predict directly without batching
            if n_total <= batch_size:
                # logger.info(f"Predicting on {n_total:,} samples (single pass)...")
                return model.predict_proba(features_view)[:, 1].astype(np.float32)
            else:
                # logger.info(f"Predicting on {n_total:,} samples using cached view (batched)...")
                return predict_proba_batched(
                    model,
                    features_view,
                    batch_size=batch_size,
                    desc="Prediction",
                )

        # Fallback: batch-by-batch extraction (for when zero-copy view unavailable)
        # Single-pass if data fits in one batch
        if n_total <= batch_size:
            # logger.info(f"Predicting on {n_total:,} samples using batched extraction (single pass)...")
            all_features = extract_features_array(self.unlabeled_samples, np.arange(n_total), self.feature_cols)
            return model.predict_proba(all_features)[:, 1].astype(np.float32)

        # Multi-batch extraction
        logger.info(f"Predicting on {n_total:,} samples using batched extraction...")
        all_preds = np.zeros(n_total, dtype=np.float32)

        for start in tqdmu(range(0, n_total, batch_size), desc="Batched prediction"):
            end = min(start + batch_size, n_total)
            batch_features = extract_features_array(self.unlabeled_samples, np.arange(start, end), self.feature_cols)
            all_preds[start:end] = model.predict_proba(batch_features)[:, 1]
            del batch_features

        clean_ram()
        return all_preds

    def _get_tracked_probabilities(self, log: bool = False) -> tuple[float | None, float | None]:
        """Get predicted probabilities for tracked row indices.

        Uses cached features to avoid repeated feature extraction across iterations.
        Features are extracted once on first call and reused thereafter.

        Parameters
        ----------
        log : bool, default False
            If True, log the tracked probabilities.

        Returns:
            (positive_prob, negative_prob) tuple
        """
        if self.model is None:
            return None, None

        pos_prob = None
        neg_prob = None

        # Build indices list for tracking (needed for cache key and result mapping)
        indices_to_predict = []
        pos_batch_idx = None
        neg_batch_idx = None

        if self._tracked_positive_rowid_index is not None:
            pos_batch_idx = len(indices_to_predict)
            indices_to_predict.append(self._tracked_positive_rowid_index)

        if self._tracked_negative_rowid_index is not None:
            neg_batch_idx = len(indices_to_predict)
            indices_to_predict.append(self._tracked_negative_rowid_index)

        if not indices_to_predict:
            return None, None

        try:
            # Cache features on first call (indices never change)
            if self._tracked_features_cache is None:
                self._tracked_features_cache = extract_features_array(self.unlabeled_samples, indices_to_predict, self.feature_cols)

            probs = self.model.predict_proba(self._tracked_features_cache)[:, 1]

            if pos_batch_idx is not None:
                pos_prob = float(probs[pos_batch_idx])
            if neg_batch_idx is not None:
                neg_prob = float(probs[neg_batch_idx])

            # Log in single line with requested format (only if log=True)
            if log:
                parts = []
                if neg_prob is not None:
                    parts.append(f"negative [row {self._tracked_negative_rowid_index}] P(flare)={neg_prob*100:.2f}%")
                if pos_prob is not None:
                    parts.append(f"positive [row {self._tracked_positive_rowid_index}] P(flare)={pos_prob*100:.2f}%")
                if parts:
                    logger.info(f"Tracked: {', '.join(parts)}")

        except Exception as e:
            logger.warning(f"Failed to get probabilities for tracked rows: {e}")

        return pos_prob, neg_prob

    # =========================================================================
    # Phase 0: Initialization
    # =========================================================================

    def initialize(self) -> None:
        """
        Phase 0: Initialize the pipeline.

        1. Stratified split of known flares into train/val/held-out
        2. Sample negatives from unlabeled_samples
        3. Train initial model
        4. Compute baseline metrics

        The stratified split ensures each subset contains representative
        examples from different flare types (amplitude, duration, etc.).
        """
        # Start time tracking
        self.start_time = time.time()

        logger.info("=" * 60)
        logger.info("PHASE 0: INITIALIZATION")
        logger.info("=" * 60)

        # Stratified split of known flares into train/validation/held-out
        n_flares = len(self.known_flares)
        data = self.config.data
        train_ratio = data.n_train_flares / n_flares
        val_ratio = data.n_validation_flares / n_flares
        held_out_ratio = data.n_held_out_flares / n_flares

        train_flare_indices, val_flare_indices, held_out_flare_indices = stratified_flare_split(
            self.known_flares,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            held_out_ratio=held_out_ratio,
            random_state=self.random_state,
            exclude_indices=self.freaky_held_out_indices,
        )

        val_flare_indices = np.array(val_flare_indices)
        held_out_flare_indices = np.array(held_out_flare_indices)

        logger.info(
            f"Flares split (stratified): train={len(train_flare_indices)}, " f"validation={len(val_flare_indices)}, held_out={len(held_out_flare_indices)}"
        )

        # Sample negatives from unlabeled_samples for train, validation, and held-out
        n_unlabeled = len(self.unlabeled_samples)
        all_neg_indices = self.rng.permutation(n_unlabeled)

        # Split negatives: train | validation | held-out
        train_neg_end = data.n_train_neg_init
        val_neg_end = train_neg_end + data.n_validation_neg
        held_out_neg_end = val_neg_end + data.n_held_out_neg

        train_neg_indices = all_neg_indices[:train_neg_end].tolist()
        val_neg_indices = all_neg_indices[train_neg_end:val_neg_end]
        held_out_neg_indices = all_neg_indices[val_neg_end:held_out_neg_end]

        # Verify no overlap between negative index sets (defensive check)
        train_set = set(train_neg_indices)
        val_set = set(val_neg_indices)
        held_out_set = set(held_out_neg_indices)
        assert train_set.isdisjoint(val_set), "Train and validation negative indices overlap!"
        assert train_set.isdisjoint(held_out_set), "Train and held-out negative indices overlap!"
        assert val_set.isdisjoint(held_out_set), "Validation and held-out negative indices overlap!"

        logger.info(f"Negatives sampled: train={len(train_neg_indices)}, " f"validation={len(val_neg_indices)}, held_out={len(held_out_neg_indices)}")

        # Build extra validation indices from tracked samples and forced samples
        extra_pos_list = []
        extra_neg_list = []

        # Add tracked samples to validation
        if self._tracked_positive_rowid_index is not None:
            extra_pos_list.append(self._tracked_positive_rowid_index)
            logger.info(f"Adding tracked positive (row {self._tracked_positive_rowid_index}) to validation set")
        if self._tracked_negative_rowid_index is not None:
            extra_neg_list.append(self._tracked_negative_rowid_index)
            logger.info(f"Adding tracked negative (row {self._tracked_negative_rowid_index}) to validation set")

        # Add forced samples to validation (they influence decision-making)
        n_unlabeled = len(self.unlabeled_samples)
        if self.config.forced_positive_indices:
            valid_forced_pos = [idx for idx in self.config.forced_positive_indices if 0 <= idx < n_unlabeled]
            extra_pos_list.extend(valid_forced_pos)
            if valid_forced_pos:
                logger.info(f"Adding {len(valid_forced_pos)} forced positives to validation set")
        if self.config.forced_negative_indices:
            valid_forced_neg = [idx for idx in self.config.forced_negative_indices if 0 <= idx < n_unlabeled]
            extra_neg_list.extend(valid_forced_neg)
            if valid_forced_neg:
                logger.info(f"Adding {len(valid_forced_neg)} forced negatives to validation set")

        extra_pos_indices = np.array(extra_pos_list, dtype=np.int64) if extra_pos_list else None
        extra_neg_indices = np.array(extra_neg_list, dtype=np.int64) if extra_neg_list else None

        # Create validation set (used for rollback/model selection decisions)
        self.validation = ValidationSet(
            flare_indices=val_flare_indices,
            negative_indices=val_neg_indices,
            extra_positive_indices=extra_pos_indices,
            extra_negative_indices=extra_neg_indices,
        )

        # Create held-out set (truly honest - NEVER used for any decisions)
        self.held_out = HeldOutSet(
            flare_indices=held_out_flare_indices,
            negative_indices=held_out_neg_indices,
        )

        # Cache validation/held-out features once (they never change)
        logger.info("Caching validation and held-out features...")
        self._validation_features_cache = (
            extract_features_array(self.known_flares, self.validation.flare_indices, self.feature_cols),
            extract_features_array(self.unlabeled_samples, self.validation.negative_indices, self.feature_cols),
        )
        self._held_out_features_cache = (
            extract_features_array(self.known_flares, self.held_out.flare_indices, self.feature_cols),
            extract_features_array(self.unlabeled_samples, self.held_out.negative_indices, self.feature_cols),
        )
        # Cache held-out negative indices for exclusion set
        self._held_out_neg_set = set(held_out_neg_indices.tolist())

        # Cache tracked row features (they never change)
        if self._tracked_positive_rowid_index is not None or self._tracked_negative_rowid_index is not None:
            indices_to_cache = []
            if self._tracked_positive_rowid_index is not None:
                indices_to_cache.append(self._tracked_positive_rowid_index)
            if self._tracked_negative_rowid_index is not None:
                indices_to_cache.append(self._tracked_negative_rowid_index)
            self._tracked_features_cache = extract_features_array(self.unlabeled_samples, indices_to_cache, self.feature_cols)
            logger.info(f"Cached tracked row features: {len(indices_to_cache)} samples")

        # Cache freaky held-out set if configured (similar to validation/held-out)
        if self.freaky_held_out_indices:
            freaky_flare_indices = np.array(self.freaky_held_out_indices)
            # Sample negatives for freaky set (same ratio as validation set)
            neg_per_pos_ratio = self.config.data.n_validation_neg // max(1, self.config.data.n_validation_flares)
            n_freaky_negs = max(FREAKY_NEG_MIN_COUNT, len(freaky_flare_indices) * neg_per_pos_ratio)
            # Exclude existing train/val/held-out negatives to avoid overlap
            existing_neg_set = set(train_neg_indices) | set(val_neg_indices) | self._held_out_neg_set
            available_for_freaky = [i for i in range(len(self.unlabeled_samples)) if i not in existing_neg_set]
            freaky_neg_indices = self.rng.choice(available_for_freaky, size=min(n_freaky_negs, len(available_for_freaky)), replace=False)

            self.freaky_set = ValidationSet(
                flare_indices=freaky_flare_indices,
                negative_indices=freaky_neg_indices,
            )
            self._freaky_features_cache = (
                extract_features_array(self.known_flares, freaky_flare_indices, self.feature_cols),
                extract_features_array(self.unlabeled_samples, freaky_neg_indices, self.feature_cols),
            )
            logger.info(f"Cached freaky held-out set: {len(freaky_flare_indices)} flares, {len(freaky_neg_indices)} negatives")
        else:
            self.freaky_set = None
            self._freaky_features_cache = None

        # Add flares and negatives to labeled_train using batch method
        flare_samples = [
            LabeledSample(
                index=int(idx),
                label=1,
                weight=1.0,
                source=SampleSource.SEED,
                added_iter=0,
                confidence=1.0,
                consensus_score=1.0,
                from_known_flares=True,
            )
            for idx in train_flare_indices
        ]
        neg_samples = [
            LabeledSample(
                index=int(idx),
                label=0,
                weight=1.0,
                source=SampleSource.SEED,
                added_iter=0,
                confidence=1.0,
                consensus_score=1.0,
                from_known_flares=False,
            )
            for idx in train_neg_indices
        ]
        self._add_samples_batch(flare_samples + neg_samples)

        # Add forced samples from unlabeled pool (user-verified samples)
        forced_samples = []
        if self.config.forced_positive_indices:
            n_unlabeled = len(self.unlabeled_samples)
            valid_forced_pos = [idx for idx in self.config.forced_positive_indices if 0 <= idx < n_unlabeled]
            if len(valid_forced_pos) < len(self.config.forced_positive_indices):
                logger.warning(f"Skipped {len(self.config.forced_positive_indices) - len(valid_forced_pos)} out-of-bounds forced positive indices")
            for idx in valid_forced_pos:
                forced_samples.append(LabeledSample(
                    index=int(idx),
                    label=1,
                    weight=1.0,
                    source=SampleSource.FORCED_POS,
                    added_iter=0,
                    confidence=1.0,
                    consensus_score=1.0,
                    from_known_flares=False,
                ))
            if valid_forced_pos:
                logger.info(f"Added {len(valid_forced_pos)} forced positives from unlabeled pool")

        if self.config.forced_negative_indices:
            n_unlabeled = len(self.unlabeled_samples)
            valid_forced_neg = [idx for idx in self.config.forced_negative_indices if 0 <= idx < n_unlabeled]
            if len(valid_forced_neg) < len(self.config.forced_negative_indices):
                logger.warning(f"Skipped {len(self.config.forced_negative_indices) - len(valid_forced_neg)} out-of-bounds forced negative indices")
            for idx in valid_forced_neg:
                forced_samples.append(LabeledSample(
                    index=int(idx),
                    label=0,
                    weight=1.0,
                    source=SampleSource.FORCED_NEG,
                    added_iter=0,
                    confidence=1.0,
                    consensus_score=1.0,
                    from_known_flares=False,
                ))
            if valid_forced_neg:
                logger.info(f"Added {len(valid_forced_neg)} forced negatives from unlabeled pool")

        if forced_samples:
            self._add_samples_batch(forced_samples)

        logger.info(f"Initial labeled_train size: {len(self.labeled_train)}")

        # Train initial model
        logger.info("Training initial model...")
        features, labels, weights = self._build_training_data()
        training_curves_dir = self.output_dir / "training_curves"
        training_curves_dir.mkdir(parents=True, exist_ok=True)
        plot_file = training_curves_dir / "training_iter000.html"
        self.model = train_model(features, labels, weights, config=self.config, random_state=self.random_state, plot_file=plot_file)

        # Compute baseline validation metrics (used for decisions)
        validation_metrics = self._compute_validation_metrics(iteration=0)

        self.prev_validation_recall = validation_metrics["recall"]
        self.prev_validation_precision = validation_metrics["precision"]
        self.prev_validation_ice = validation_metrics.get("ice", 1.0)
        self.best_validation_recall = validation_metrics["recall"]
        self.best_validation_ice = validation_metrics.get("ice", 1.0)

        # Initialize tracked sample probabilities for decision-making
        self.prev_tracked_pos_prob, self.prev_tracked_neg_prob = self._get_tracked_probabilities()

        # Save initial checkpoint as best
        self.best_checkpoint = self._save_checkpoint(0, validation_metrics)

        # Log initial metrics
        counts = self._count_by_source()
        eff_pos, eff_neg = self._compute_effective_sizes()
        tracked_pos_prob, tracked_neg_prob = self.prev_tracked_pos_prob, self.prev_tracked_neg_prob

        initial_metrics = IterationMetrics(
            iteration=0,
            train_total=len(self.labeled_train),
            train_seed=counts[SampleSource.SEED],
            train_pseudo_pos=counts[SampleSource.PSEUDO_POS],
            train_pseudo_neg=counts[SampleSource.PSEUDO_NEG],
            train_forced_pos=counts[SampleSource.FORCED_POS],
            train_forced_neg=counts[SampleSource.FORCED_NEG],
            effective_pos=eff_pos,
            effective_neg=eff_neg,
            validation_recall=validation_metrics["recall"],
            validation_precision=validation_metrics["precision"],
            validation_auc=validation_metrics["auc"],
            validation_f1=validation_metrics["f1"],
            validation_logloss=validation_metrics.get("logloss", 0.0),
            validation_brier=validation_metrics.get("brier", 0.0),
            validation_ice=validation_metrics.get("ice", 0.0),
            validation_pr_auc=validation_metrics.get("pr_auc", 0.0),
            enrichment_factor=0.0,
            estimated_flares_top10k=0.0,
            pseudo_pos_threshold=self.pseudo_pos_threshold,
            pseudo_neg_threshold=self.pseudo_neg_threshold,
            pseudo_pos_added=0,
            pseudo_neg_added=0,
            pseudo_removed=0,
            n_successful_iters=0,
            n_rollbacks_recent=0,
            elapsed_hours=self._get_elapsed_hours(),
            tracked_positive_rowid_prob=tracked_pos_prob,
            tracked_negative_rowid_prob=tracked_neg_prob,
        )
        self.metrics_history.append(initial_metrics)

        logger.info("=" * 60)
        logger.info("ITERATION 0 COMPLETE")
        logger.info(f"  Train: {len(self.labeled_train)} samples")
        logger.info(f"  {self._format_metrics_log(validation_metrics, 'Validation: ')}")
        logger.info("=" * 60)

    # =========================================================================
    # Main Loop Phases
    # =========================================================================

    def _validate_and_check_stopping(self, iteration: int) -> tuple[dict, bool, str | None]:
        """
        Validate model and check for early stopping or rollback.

        Decision criteria combines:
        1. Validation ICE (Integrated Calibration Error) - primary metric
        2. Tracked sample predictions - secondary signal for model quality

        Tracked samples contribute to the decision in two ways:
        - Degradation detection: if tracked positive's P(flare) drops significantly,
          or tracked negative's P(flare) rises significantly, triggers rollback
        - Best model selection: combined score weighs ICE and tracked sample accuracy

        Uses the validation set (not held-out) for all training decisions.
        The held-out set is only evaluated once at the end in _finalize().

        Returns:
            (validation_metrics, rollback_occurred, stop_reason)
        """
        validation_metrics = self._compute_validation_metrics(iteration=iteration)

        current_recall = validation_metrics["recall"]
        current_precision = validation_metrics["precision"]
        current_ice = validation_metrics.get("ice", 1.0)

        # Get tracked sample probabilities
        tracked_pos_prob, tracked_neg_prob = self._get_tracked_probabilities()

        # =====================================================================
        # Check for degradation using ICE AND tracked samples
        # =====================================================================
        ice_increased = current_ice > self.prev_validation_ice + self.config.ice_increase_threshold

        # Check tracked sample degradation (only if we have previous values to compare)
        tracked_pos_degraded = False
        tracked_neg_degraded = False

        if tracked_pos_prob is not None and self.prev_tracked_pos_prob is not None:
            # Tracked positive: P(flare) should stay high - degradation if it drops
            pos_drop = self.prev_tracked_pos_prob - tracked_pos_prob
            tracked_pos_degraded = pos_drop > self.config.tracked_pos_degradation_threshold

        if tracked_neg_prob is not None and self.prev_tracked_neg_prob is not None:
            # Tracked negative: P(flare) should stay low - degradation if it rises
            neg_rise = tracked_neg_prob - self.prev_tracked_neg_prob
            tracked_neg_degraded = neg_rise > self.config.tracked_neg_degradation_threshold

        # Trigger rollback if ICE degraded OR tracked samples degraded significantly
        degradation_detected = ice_increased or tracked_pos_degraded or tracked_neg_degraded

        if degradation_detected:
            reasons = []
            if ice_increased:
                reasons.append(f"ICE: {self.prev_validation_ice:.4f} -> {current_ice:.4f}")
            if tracked_pos_degraded:
                reasons.append(f"tracked_pos P(flare): {self.prev_tracked_pos_prob:.4f} -> {tracked_pos_prob:.4f}")
            if tracked_neg_degraded:
                reasons.append(f"tracked_neg P(flare): {self.prev_tracked_neg_prob:.4f} -> {tracked_neg_prob:.4f}")

            logger.warning(f"DEGRADATION DETECTED: {', '.join(reasons)}")
            logger.warning(f"  (Recall: {self.prev_validation_recall:.3f} -> {current_recall:.3f})")
            logger.warning(f"  (Precision: {self.prev_validation_precision:.3f} -> {current_precision:.3f})")

            # Rollback
            if self.best_checkpoint is not None:
                # Ban samples added since best checkpoint (before loading checkpoint)
                # Protect SEED, FORCED_POS, FORCED_NEG from banning
                protected_sources = {SampleSource.SEED, SampleSource.FORCED_POS, SampleSource.FORCED_NEG}
                samples_to_ban = [s for s in self.labeled_train if s.added_iter > self.best_checkpoint.iteration and s.source not in protected_sources]
                self._add_to_ban_list(samples_to_ban, iteration)

                logger.info(f"Rolling back to iteration {self.best_checkpoint.iteration}")
                self._load_checkpoint(self.best_checkpoint)

                # Restore ban list from checkpoint if available
                if self.best_checkpoint.ban_list:
                    for key_str, expires_at in self.best_checkpoint.ban_list.items():
                        # Convert string key back to tuple
                        idx, lbl = map(int, key_str.strip("()").split(","))
                        self._ban_list[(idx, lbl)] = expires_at

            # Tighten thresholds after rollback
            self._tighten_thresholds()

            self.n_successful_iters = 0
            self.rollback_history.append(iteration)

            # Recompute metrics after rollback and update baseline
            # This is CRITICAL to prevent infinite rollback loops
            validation_metrics = self._compute_validation_metrics()
            self.prev_validation_recall = validation_metrics["recall"]
            self.prev_validation_precision = validation_metrics["precision"]
            self.prev_validation_ice = validation_metrics.get("ice", 1.0)
            # Also update tracked probabilities after rollback
            self.prev_tracked_pos_prob, self.prev_tracked_neg_prob = self._get_tracked_probabilities()

            return validation_metrics, True, None  # rollback_occurred=True, skip rest of iteration

        # =====================================================================
        # No degradation - check if this is the best model
        # =====================================================================
        self.n_successful_iters += 1

        # Compute combined decision score (lower is better)
        # Score = ICE * (1 - w) + tracked_error * w
        # where tracked_error = (1 - pos_prob) + neg_prob (both should be low for good model)
        ice_available = "ice" in validation_metrics
        tracked_weight = self.config.tracked_sample_decision_weight

        if ice_available:
            current_score = current_ice
            best_score = self.best_validation_ice

            # Add tracked sample contribution if available
            if tracked_weight > 0 and (tracked_pos_prob is not None or tracked_neg_prob is not None):
                tracked_error = 0.0
                n_tracked = 0
                if tracked_pos_prob is not None:
                    tracked_error += 1.0 - tracked_pos_prob  # Lower is better (high prob = good)
                    n_tracked += 1
                if tracked_neg_prob is not None:
                    tracked_error += tracked_neg_prob  # Lower is better (low prob = good)
                    n_tracked += 1
                if n_tracked > 0:
                    tracked_error /= n_tracked  # Normalize

                # Combine: weighted average of ICE and tracked_error
                current_score = current_ice * (1 - tracked_weight) + tracked_error * tracked_weight

                # Compute best_score with same formula (use stored probs or assume perfect)
                best_tracked_error = 0.0
                if self.prev_tracked_pos_prob is not None:
                    best_tracked_error += 1.0 - self.prev_tracked_pos_prob
                if self.prev_tracked_neg_prob is not None:
                    best_tracked_error += self.prev_tracked_neg_prob
                if n_tracked > 0:
                    best_tracked_error /= n_tracked
                best_score = self.best_validation_ice * (1 - tracked_weight) + best_tracked_error * tracked_weight

            is_better = current_score < best_score
        else:
            # ICE unavailable: prefer higher recall
            is_better = current_recall > self.best_validation_recall

        if is_better:
            self.best_validation_ice = current_ice
            self.best_validation_recall = current_recall
            self.best_checkpoint = self._save_checkpoint(iteration, validation_metrics)
            if ice_available:
                msg = f"New best model saved (ICE={current_ice:.4f}, RE={current_recall:.3f}"
                if tracked_weight > 0 and (tracked_pos_prob is not None or tracked_neg_prob is not None):
                    msg += f", combined_score={current_score:.4f}"
                    if tracked_pos_prob is not None:
                        msg += f", tracked_pos={tracked_pos_prob:.3f}"
                    if tracked_neg_prob is not None:
                        msg += f", tracked_neg={tracked_neg_prob:.3f}"
                msg += ")"
                logger.info(msg)
            else:
                logger.info(f"New best model saved (validation recall={current_recall:.3f}, ICE not available)")

        # Check for success
        enrichment, _ = self._compute_enrichment_factor() if iteration > 0 else (0.0, 0.0)
        if (
            current_recall >= self.config.target_recall
            and current_precision >= self.config.target_precision
            and enrichment >= self.config.target_enrichment
            and self.n_successful_iters >= self.config.min_successful_iters_for_success
        ):
            return validation_metrics, False, "SUCCESS"

        return validation_metrics, False, None  # rollback_occurred=False, continue normally

    def _train_bootstrap_ensemble(self, iteration: int) -> tuple[list[CatBoostClassifier], list[np.ndarray]]:
        """
        Train bootstrap models for consensus estimation and OOB evaluation.

        Bootstrap aggregation (bagging) provides two benefits:
        1. Consensus estimation: Multiple models must agree on pseudo-labels
        2. OOB evaluation: Each sample has ~37% chance of being OOB for each model

        Returns
        -------
        tuple[list[CatBoostClassifier], list[np.ndarray]]
            (bootstrap_models, bootstrap_indices_list)
        """
        bootstrap_models = []
        bootstrap_indices_list = []
        features, labels, weights = self._build_training_data()
        n_samples = len(labels)

        for seed in tqdmu(range(self.config.thresholds.n_bootstrap_models), desc="Bootstrap models", leave=False):
            # Bootstrap sample with replacement
            bootstrap_rng = np.random.default_rng(seed + iteration * 100 + self.random_state)
            bootstrap_indices = bootstrap_rng.choice(n_samples, size=n_samples, replace=True)

            bootstrap_features = features[bootstrap_indices]
            bootstrap_labels = labels[bootstrap_indices]
            bootstrap_weights = weights[bootstrap_indices]

            model = train_model(
                bootstrap_features,
                bootstrap_labels,
                bootstrap_weights,
                config=self.config,
                random_state=seed + iteration * 100,
                iterations_override=self.config.thresholds.bootstrap_iterations,
                early_stopping_rounds_override=self.config.thresholds.bootstrap_early_stopping_rounds,
            )
            bootstrap_models.append(model)
            bootstrap_indices_list.append(bootstrap_indices)

        self.bootstrap_models = bootstrap_models
        self.bootstrap_indices_list = bootstrap_indices_list
        return bootstrap_models, bootstrap_indices_list

    def _predict_unlabeled_pool(self) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        """
        Predict on entire unlabeled pool using main and bootstrap models.

        Uses batched prediction to avoid memory issues. Predictions are made on ALL
        samples - held_out exclusion happens at the pseudo-label selection stage.

        Returns
        -------
        tuple[np.ndarray, list[np.ndarray], np.ndarray]
            (main_predictions, bootstrap_predictions, all_indices)
            all_indices is just np.arange(n_unlabeled) for full dataset

        Performance Notes (GPU-specific)
        --------------------------------
        The following optimizations were considered but NOT implemented:

        Option A: Skip bootstrap predictions for negatives
          - Rationale: Bootstrap consensus is less critical for negatives (base rate favors them)
          - Trade-off: Slightly higher false negative risk in pseudo-neg selection
          - Decision: NOT IMPLEMENTED - maintain consistency in consensus checking

        Option B: Sample-based prediction for negative candidate search
          - Instead of predicting ALL unlabeled, sample ~10% for negative selection
          - Trade-off: May miss some good negative candidates
          - Decision: NOT IMPLEMENTED - prefer exhaustive search for quality
        """
        n_unlabeled = len(self.unlabeled_samples)
        all_indices = np.arange(n_unlabeled)

        # Main model predictions
        main_preds = self._predict_unlabeled_features_batched(self.model)

        # Bootstrap model predictions (sequential - GPU already parallelizes internally)
        bootstrap_preds = []
        for bm in tqdmu(self.bootstrap_models, desc="Predicting models", leave=False):
            bp = self._predict_unlabeled_features_batched(bm)
            bootstrap_preds.append(bp)

        clean_ram()
        return main_preds, bootstrap_preds, all_indices

    def _filter_candidates_by_threshold(
        self,
        main_preds: np.ndarray,
        prediction_indices: np.ndarray,
        threshold: float,
        above: bool,
        subsample_limit: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter candidates by threshold and exclusion set.

        Shared logic for both positive and negative pseudo-labeling.

        Parameters
        ----------
        main_preds : np.ndarray
            Main model predictions for all samples.
        prediction_indices : np.ndarray
            Maps position in main_preds to index in unlabeled_samples.
        threshold : float
            Probability threshold for filtering.
        above : bool
            If True, keep candidates with prob > threshold (positives).
            If False, keep candidates with prob < threshold (negatives).
        subsample_limit : int, optional
            If provided, randomly subsample to this many candidates.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            (positions, indices, probs) where:
            - positions: indices into main_preds array
            - indices: indices into unlabeled_samples
            - probs: main model probabilities for candidates
        """
        # Find candidates by threshold
        if above:
            candidate_positions = np.where(main_preds > threshold)[0]
        else:
            candidate_positions = np.where(main_preds < threshold)[0]

        if len(candidate_positions) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        # Map to unlabeled_samples indices and filter exclusions
        candidate_indices = prediction_indices[candidate_positions]
        exclusion_array = self._build_pseudo_label_exclusion_array()
        valid_mask = ~np.isin(candidate_indices, exclusion_array)

        positions = candidate_positions[valid_mask]
        indices = candidate_indices[valid_mask]

        if len(positions) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        # Random subsample if requested
        if subsample_limit is not None and len(positions) > subsample_limit:
            subsample_idx = self.rng.choice(len(positions), size=subsample_limit, replace=False)
            positions = positions[subsample_idx]
            indices = indices[subsample_idx]

        probs = main_preds[positions]
        return positions, indices, probs

    def _compute_consensus_and_stats(
        self,
        bootstrap_preds: list[np.ndarray],
        positions: np.ndarray,
        is_positive: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute bootstrap consensus filter and statistics for candidates.

        Parameters
        ----------
        bootstrap_preds : list[np.ndarray]
            Bootstrap model predictions for all samples.
        positions : np.ndarray
            Positions in prediction arrays for candidates.
        is_positive : bool
            If True, check for high probability consensus (positives).
            If False, check for low probability consensus (negatives).

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (passes_consensus, means, stds, consensus_scores)
        """
        thresholds = self.config.thresholds

        # Stack bootstrap predictions for candidates: (n_models, n_candidates)
        candidate_bootstrap = np.stack([bp[positions] for bp in bootstrap_preds], axis=0)

        if is_positive:
            # Positive: ALL models must agree (high threshold)
            if NUMBA_AVAILABLE:
                passes_consensus = _check_all_high_consensus_numba(candidate_bootstrap, thresholds.consensus_threshold)
            else:
                passes_consensus = (candidate_bootstrap > thresholds.consensus_threshold).all(axis=0)
        else:
            # Negative: majority must agree (low threshold)
            if NUMBA_AVAILABLE:
                n_low = _count_low_prob_consensus_numba(candidate_bootstrap, thresholds.neg_consensus_min_low_prob)
            else:
                n_low = (candidate_bootstrap < thresholds.neg_consensus_min_low_prob).sum(axis=0)
            n_required = max(MIN_BOOTSTRAP_CONSENSUS_MODELS, int(thresholds.n_bootstrap_models * BOOTSTRAP_CONSENSUS_FRACTION))
            passes_consensus = n_low >= n_required

        # Compute stats
        means, stds, consensus_scores = compute_bootstrap_stats(candidate_bootstrap)

        return passes_consensus, means, stds, consensus_scores

    def _pseudo_label_negatives(
        self,
        main_preds: np.ndarray,
        bootstrap_preds: list[np.ndarray],
        prediction_indices: np.ndarray,
        iteration: int,
    ) -> int:
        """
        Phase 5: Aggressive pseudo-labeling of negatives.

        In rare event detection, the negative class is abundant but trivial to
        learn. We use aggressive pseudo-labeling for negatives because:

        1. Base rate is ~99.9% negative, so random samples are likely true negatives
        2. False negatives (missing a flare) are costly, false positives less so
        3. Expanding the negative set improves precision without hurting recall

        The strategy:
        - Low threshold (P < 0.05): Only very confident negative predictions
        - Majority consensus: At least half of bootstrap models must agree
        - High volume: Add up to 100 negatives per iteration

        Parameters
        ----------
        main_preds : np.ndarray
            Predictions array (length = len(prediction_indices))
        bootstrap_preds : list[np.ndarray]
            Bootstrap predictions list
        prediction_indices : np.ndarray
            Maps position in predictions to index in unlabeled_samples
        iteration : int
            Current iteration number

        Returns
        -------
        int
            Number of negatives added.

        References
        ----------
        - Xie et al. (2020). Self-Training with Noisy Student. CVPR.
        - Arazo et al. (2020). Pseudo-Labeling and Confirmation Bias. IJCNN.
        """
        thresholds = self.config.thresholds

        # Filter candidates using shared helper
        subsample_limit = thresholds.max_pseudo_neg_per_iter * PSEUDO_NEG_SUBSAMPLE_MULTIPLIER
        positions, big_indices, main_probs = self._filter_candidates_by_threshold(
            main_preds, prediction_indices, self.pseudo_neg_threshold, above=False, subsample_limit=subsample_limit
        )
        if len(positions) == 0:
            return 0

        # Compute consensus and stats using shared helper
        passes_consensus, bootstrap_mean, bootstrap_std, consensus_scores = self._compute_consensus_and_stats(bootstrap_preds, positions, is_positive=False)

        # Compute confidences
        avg_probs = (main_probs + bootstrap_mean) / 2
        confidences = 1 - avg_probs

        # Track bootstrap-discarded samples
        n_candidates = len(positions)
        n_passed_consensus = int(passes_consensus.sum())
        n_discarded_by_bootstrap = n_candidates - n_passed_consensus

        # Log and plot one example of discarded negative (if any)
        if n_discarded_by_bootstrap > 0:
            discarded_mask = ~passes_consensus
            discarded_idx = int(big_indices[discarded_mask][0])  # First discarded
            discarded_prob = float(main_probs[discarded_mask][0])
            discarded_consensus = float(consensus_scores[discarded_mask][0])
            logger.info(f"Bootstrap discarded {n_discarded_by_bootstrap} neg candidates (of {n_candidates})")
            logger.info(f"  Example discarded neg: idx={discarded_idx}, P(flare)={discarded_prob:.4f}, consensus={discarded_consensus:.4f}")
            plot_sample(
                discarded_idx,
                self.unlabeled_samples,
                self.output_dir,
                "bootstrap_discarded",
                self.config,
                action="discarded_neg",
                dataset=self.unlabeled_dataset,
                probability=discarded_prob,
                iteration=iteration,
            )

        # Filter to passing candidates
        passing_mask = passes_consensus
        confirmed_indices = big_indices[passing_mask]
        confirmed_confidences = confidences[passing_mask]
        confirmed_consensus = consensus_scores[passing_mask]

        # Sort by confidence (descending) and take top
        sort_order = np.argsort(-confirmed_confidences)[: thresholds.max_pseudo_neg_per_iter]

        # Pre-filter banned samples (vectorized)
        banned_indices = self._get_banned_indices(0, iteration)
        if banned_indices:
            banned_array = np.array(list(banned_indices), dtype=np.int64)
            valid_mask = ~np.isin(confirmed_indices[sort_order], banned_array)
            valid_sort_order = sort_order[valid_mask]
            n_banned_skipped = len(sort_order) - len(valid_sort_order)
        else:
            valid_sort_order = sort_order
            n_banned_skipped = 0

        # Compute weight based on successful iterations
        weight = self._compute_pseudo_label_weight(SampleSource.PSEUDO_NEG)

        # Batch add to training (more efficient than individual _add_sample calls)
        new_samples = [
            LabeledSample(
                index=int(confirmed_indices[i]),
                label=0,
                weight=weight,
                source=SampleSource.PSEUDO_NEG,
                added_iter=iteration,
                confidence=float(confirmed_confidences[i]),
                consensus_score=float(confirmed_consensus[i]),
                from_known_flares=False,
            )
            for i in valid_sort_order
        ]
        self._add_samples_batch(new_samples)

        n_actually_added = len(new_samples)
        if n_banned_skipped > 0:
            logger.info(f"Pseudo-negatives: skipped {n_banned_skipped} banned samples")
        logger.info(f"Pseudo-negatives added: {n_actually_added} (of {n_candidates} candidates, {n_discarded_by_bootstrap} discarded by bootstrap), weight={weight:.2f}")
        return n_actually_added

    def _pseudo_label_positives(
        self,
        main_preds: np.ndarray,
        bootstrap_preds: list[np.ndarray],
        prediction_indices: np.ndarray,
        iteration: int,
    ) -> int:
        """
        Phase 6: Conservative pseudo-labeling of positives.

        Unlike negatives, pseudo-labeled positives require extreme caution:

        1. True positives are rare (~0.1%), so even high-confidence predictions
           may contain substantial false positive contamination
        2. Confirmation bias: Model can reinforce its own mistakes
        3. Class imbalance amplification: Bad pseudo-positives hurt more

        The strategy:
        - High threshold (P > 0.99): Only extremely confident predictions
        - Full consensus: ALL bootstrap models must agree (P > 0.95)
        - Low variance: Bootstrap std < 0.05 (stable prediction)
        - Low volume: Maximum 10 positives per iteration

        This asymmetric approach (aggressive neg, conservative pos) is optimal
        for rare event detection where precision matters but positive examples
        are scarce and valuable.

        Parameters
        ----------
        main_preds : np.ndarray
            Predictions array (length = len(prediction_indices))
        bootstrap_preds : list[np.ndarray]
            Bootstrap predictions list
        prediction_indices : np.ndarray
            Maps position in predictions to index in unlabeled_samples
        iteration : int
            Current iteration number

        Returns
        -------
        int
            Number of positives added.

        References
        ----------
        - Lee (2013). Pseudo-Label: The Simple and Efficient Semi-Supervised
          Learning Method. ICML Workshop.
        - Zou et al. (2019). Confidence Regularized Self-Training. ICCV.
        """
        thresholds = self.config.thresholds

        # Filter candidates using shared helper (no subsampling for positives)
        positions, big_indices, main_probs = self._filter_candidates_by_threshold(main_preds, prediction_indices, self.pseudo_pos_threshold, above=True)
        if len(positions) == 0:
            return 0

        # Take top-K by probability (use argpartition for efficiency)
        top_k_limit = self.max_pseudo_pos_per_iter * PSEUDO_POS_TOP_K_MULTIPLIER
        if len(positions) > top_k_limit:
            top_indices = np.argpartition(main_probs, -top_k_limit)[-top_k_limit:]
            big_indices = big_indices[top_indices]
            positions = positions[top_indices]
            main_probs = main_probs[top_indices]

        # Compute consensus and stats using shared helper
        passes_consensus, bootstrap_mean, bootstrap_std, consensus_scores = self._compute_consensus_and_stats(bootstrap_preds, positions, is_positive=True)

        # Additional filter for positives: low variance required
        low_variance = bootstrap_std < thresholds.bootstrap_variance_threshold
        passes_all = passes_consensus & low_variance

        # Track bootstrap-discarded samples (breakdown by mutually exclusive reasons)
        n_candidates = len(positions)
        failed_consensus_only = (~passes_consensus) & low_variance
        failed_variance_only = passes_consensus & (~low_variance)
        failed_both = (~passes_consensus) & (~low_variance)
        n_failed_consensus_only = int(failed_consensus_only.sum())
        n_failed_variance_only = int(failed_variance_only.sum())
        n_failed_both = int(failed_both.sum())
        n_discarded_by_bootstrap = n_failed_consensus_only + n_failed_variance_only + n_failed_both

        # Log and plot up to 5 examples each for failed consensus and failed variance
        if n_discarded_by_bootstrap > 0:
            logger.info(
                f"Bootstrap discarded {n_discarded_by_bootstrap} pos candidates (of {n_candidates}): "
                f"{n_failed_consensus_only} consensus-only, {n_failed_variance_only} variance-only, {n_failed_both} both"
            )

            # Plot up to 5 examples that failed consensus (but passed variance)
            n_consensus_examples = min(5, n_failed_consensus_only)
            if n_consensus_examples > 0:
                consensus_indices = np.where(failed_consensus_only)[0][:n_consensus_examples]
                logger.info(f"  Failed consensus examples ({n_consensus_examples}):")
                for j in consensus_indices:
                    idx = int(big_indices[j])
                    prob = float(main_probs[j])
                    cons = float(consensus_scores[j])
                    std = float(bootstrap_std[j])
                    logger.info(f"    idx={idx}, P(flare)={prob:.4f}, consensus={cons:.4f}, std={std:.4f}")
                    plot_sample(
                        idx,
                        self.unlabeled_samples,
                        self.output_dir,
                        "bootstrap_discarded_consensus",
                        self.config,
                        action="discarded_pos_consensus",
                        dataset=self.unlabeled_dataset,
                        probability=prob,
                        iteration=iteration,
                    )

            # Plot up to 5 examples that failed variance (but passed consensus)
            n_variance_examples = min(5, n_failed_variance_only)
            if n_variance_examples > 0:
                variance_indices = np.where(failed_variance_only)[0][:n_variance_examples]
                logger.info(f"  Failed variance examples ({n_variance_examples}):")
                for j in variance_indices:
                    idx = int(big_indices[j])
                    prob = float(main_probs[j])
                    cons = float(consensus_scores[j])
                    std = float(bootstrap_std[j])
                    logger.info(f"    idx={idx}, P(flare)={prob:.4f}, consensus={cons:.4f}, std={std:.4f}")
                    plot_sample(
                        idx,
                        self.unlabeled_samples,
                        self.output_dir,
                        "bootstrap_discarded_variance",
                        self.config,
                        action="discarded_pos_variance",
                        dataset=self.unlabeled_dataset,
                        probability=prob,
                        iteration=iteration,
                    )

        # Apply filter
        confirmed_indices = big_indices[passes_all]
        confirmed_main_probs = main_probs[passes_all]
        confirmed_bootstrap_mean = bootstrap_mean[passes_all]
        confirmed_consensus = consensus_scores[passes_all]
        confirmed_confidences = (confirmed_main_probs + confirmed_bootstrap_mean) / 2

        # Sort by (consensus, confidence) descending - use lexsort with negated values (sorts by last key first)
        if len(confirmed_indices) == 0:
            logger.info("Pseudo-positives added: 0")
            return 0

        sort_order = np.lexsort((-confirmed_confidences, -confirmed_consensus))[: self.max_pseudo_pos_per_iter]

        # Pre-filter banned samples (vectorized)
        banned_indices = self._get_banned_indices(1, iteration)
        if banned_indices:
            banned_array = np.array(list(banned_indices), dtype=np.int64)
            valid_mask = ~np.isin(confirmed_indices[sort_order], banned_array)
            valid_sort_order = sort_order[valid_mask]
            n_banned_skipped = len(sort_order) - len(valid_sort_order)
        else:
            valid_sort_order = sort_order
            n_banned_skipped = 0

        # Compute weight (lower than negatives, scaled increment)
        weight = self._compute_pseudo_label_weight(SampleSource.PSEUDO_POS)

        # Build samples list for batch addition
        new_samples = [
            LabeledSample(
                index=int(confirmed_indices[i]),
                label=1,
                weight=weight,
                source=SampleSource.PSEUDO_POS,
                added_iter=iteration,
                confidence=float(confirmed_confidences[i]),
                consensus_score=float(confirmed_consensus[i]),
                from_known_flares=False,
            )
            for i in valid_sort_order
        ]

        # Batch add to training (more efficient than individual _add_sample calls)
        self._add_samples_batch(new_samples)

        # Log and plot each sample (I/O bound, kept separate)
        has_id_col = "id" in self.unlabeled_samples.columns
        for sample in new_samples:
            sample_id = self.unlabeled_samples[sample.index, "id"] if has_id_col else sample.index
            logger.info(f"  Added pseudo_pos: id={sample_id}, row={sample.index}, conf={sample.confidence:.4f}")

            plot_sample(
                sample.index,
                self.unlabeled_samples,
                self.output_dir,
                "pseudo_pos",
                self.config,
                action="added",
                dataset=self.unlabeled_dataset,
                probability=sample.confidence,
                iteration=iteration,
            )

        n_actually_added = len(new_samples)
        if n_banned_skipped > 0:
            logger.info(f"Pseudo-positives: skipped {n_banned_skipped} banned samples")
        logger.info(f"Pseudo-positives added: {n_actually_added}, weight={weight:.2f}")
        return n_actually_added

    def _collect_reviewable_samples(self) -> tuple[list[tuple[int, LabeledSample]], list[tuple[int, LabeledSample]]]:
        """
        Collect samples eligible for review, grouped by origin.

        Returns
        -------
        tuple[list, list]
            (known_flare_samples, unlabeled_samples) where each is a list of
            (list_index, sample) tuples. known_flare_samples are from known_flares,
            unlabeled_samples are from the unlabeled pool.
        """
        known_flare_samples = []
        unlabeled_samples = []

        for i, sample in enumerate(self.labeled_train):
            # Skip seed positives (known flares - ground truth)
            if sample.source == SampleSource.SEED and sample.label == 1:
                continue

            # Handle seed negatives based on config
            if sample.source == SampleSource.SEED and sample.label == 0:
                if not self.config.review_seed_negatives:
                    continue

            if sample.from_known_flares:
                known_flare_samples.append((i, sample))
            else:
                unlabeled_samples.append((i, sample))

        return known_flare_samples, unlabeled_samples

    def _review_sample_batch(
        self,
        samples: list[tuple[int, LabeledSample]],
        source_df: pl.DataFrame,
        is_from_known_flares: bool,
        source_name: str,
    ) -> tuple[list[int], list[SampleRemovalInfo]]:
        """
        Review a batch of samples from a single source for removal.

        Uses vectorized checks for efficiency - builds boolean masks for each
        removal condition rather than looping through samples individually.

        Parameters
        ----------
        samples : list[tuple[int, LabeledSample]]
            List of (list_index, sample) tuples to review.
        source_df : pl.DataFrame
            Source DataFrame (known_flares or unlabeled_samples).
        is_from_known_flares : bool
            True if samples are from known_flares, False if from unlabeled_samples.
        source_name : str
            Name for logging ("known_flares" or "unlabeled").

        Returns
        -------
        tuple[list[int], list[SampleRemovalInfo]]
            (indices_to_remove, removal_info_list)
        """
        if not samples:
            return [], []

        indices = [s.index for _, s in samples]
        features = extract_features_array(source_df, indices, self.feature_cols)

        main_probs = self.model.predict_proba(features)[:, 1]
        bootstrap_probs_all = np.array([bm.predict_proba(features)[:, 1] for bm in self.bootstrap_models])

        # Compute bootstrap stats once for all samples using wrapper
        means, stds, consensus_scores = compute_bootstrap_stats(bootstrap_probs_all)

        # Build vectorized metadata arrays
        labels = np.array([s.label for _, s in samples], dtype=np.int8)
        sources = np.array([s.source for _, s in samples])

        thresholds = self.config.thresholds

        # Vectorized removal condition masks
        is_pseudo_pos = (labels == 1) & (sources == SampleSource.PSEUDO_POS)
        is_pseudo_neg = (labels == 0) & (sources == SampleSource.PSEUDO_NEG)
        is_seed_neg = (labels == 0) & (sources == SampleSource.SEED)

        # Pseudo-pos removal: prob dropped OR unstable
        pos_prob_dropped = main_probs < thresholds.pseudo_pos_removal_prob
        pos_unstable = (stds > thresholds.bootstrap_instability_std) & (means < thresholds.bootstrap_instability_mean)
        remove_pseudo_pos = is_pseudo_pos & (pos_prob_dropped | pos_unstable)

        # Pseudo-neg removal: prob rose
        neg_prob_rose = main_probs > thresholds.pseudo_neg_promotion_prob
        remove_pseudo_neg = is_pseudo_neg & neg_prob_rose

        # Seed neg removal: model confident it's a flare
        seed_neg_flare = (main_probs > thresholds.seed_neg_removal_prob) & (means > thresholds.seed_neg_removal_bootstrap_mean)
        remove_seed_neg = is_seed_neg & seed_neg_flare

        # Combined removal mask
        should_remove_mask = remove_pseudo_pos | remove_pseudo_neg | remove_seed_neg

        # Build removal lists (still need per-sample info for logging)
        to_remove: list[int] = []
        to_plot: list[SampleRemovalInfo] = []

        for j in np.where(should_remove_mask)[0]:
            i, sample = samples[j]
            current_prob = float(main_probs[j])
            bootstrap_mean = float(means[j])
            bootstrap_std = float(stds[j])

            # Determine reason for logging
            if is_pseudo_pos[j]:
                if pos_prob_dropped[j]:
                    reason = f"prob dropped: was {sample.confidence:.3f}, now {current_prob:.3f}"
                else:
                    reason = f"unstable: std={bootstrap_std:.3f}, mean={bootstrap_mean:.3f}"
            elif is_pseudo_neg[j]:
                reason = f"prob rose: was {1-sample.confidence:.3f}, now {current_prob:.3f}"
            else:  # seed_neg
                reason = f"seed neg looks like flare: PR={current_prob:.3f}, bootstrap_mean={bootstrap_mean:.3f}"

            to_remove.append(i)
            to_plot.append(
                SampleRemovalInfo(
                    sample_index=sample.index,
                    is_from_known_flares=is_from_known_flares,
                    sample=sample,
                    current_prob=current_prob,
                )
            )
            logger.warning(f"Removing {sample.source} from {source_name}: {reason}")

        # Update weights for samples NOT being removed
        not_removed_mask = ~should_remove_mask
        for j in np.where(not_removed_mask)[0]:
            _, sample = samples[j]
            current_prob = float(main_probs[j])
            stats = (float(means[j]), float(stds[j]), float(consensus_scores[j]))
            self._update_sample_weight(sample, current_prob, stats)

        return to_remove, to_plot

    def _apply_removals(self, to_remove: list[int], to_plot: list[SampleRemovalInfo], iteration: int) -> None:
        """
        Plot removed samples and remove them from labeled_train.

        Parameters
        ----------
        to_remove : list[int]
            Indices in labeled_train to remove.
        to_plot : list[SampleRemovalInfo]
            Removal info for plotting.
        iteration : int
            Current iteration number.
        """
        # Plot removed samples
        for info in to_plot:
            # Resolve source DataFrame and dataset from boolean flag
            if info.is_from_known_flares:
                source_df = self.known_flares
                dataset = self.known_flares_dataset
            else:
                source_df = self.unlabeled_samples
                dataset = self.unlabeled_dataset

            sample_id = source_df[info.sample_index, "id"] if "id" in source_df.columns else info.sample_index
            logger.info(f"  Removed {info.sample.source}: id={sample_id}, row={info.sample_index}, P(flare)={info.current_prob:.4f}")
            plot_sample(
                info.sample_index,
                source_df,
                self.output_dir,
                f"removed_{info.sample.source.name}",
                self.config,
                action="removed",
                dataset=dataset,
                probability=info.current_prob,
                iteration=iteration,
            )

        # Remove in reverse order to preserve indices
        for i in sorted(to_remove, reverse=True):
            self._remove_sample(i)

    def _review_pseudo_labels(self, iteration: int) -> int:
        """
        Review and remove pseudo-labels that have become inconsistent.

        Uses batching by source for efficiency. Also reviews seed negatives
        if config.review_seed_negatives is True.

        Parameters
        ----------
        iteration : int
            Current iteration number.

        Returns
        -------
        int
            Number of samples removed.
        """
        # Collect samples to review
        known_flare_samples, unlabeled_samples = self._collect_reviewable_samples()

        # Review each batch (pass is_from_known_flares flag instead of dataset)
        flare_remove, flare_plot = self._review_sample_batch(known_flare_samples, self.known_flares, is_from_known_flares=True, source_name="known_flares")
        unlabeled_remove, unlabeled_plot = self._review_sample_batch(
            unlabeled_samples, self.unlabeled_samples, is_from_known_flares=False, source_name="unlabeled"
        )

        # Combine and apply removals
        to_remove = flare_remove + unlabeled_remove
        to_plot = flare_plot + unlabeled_plot
        self._apply_removals(to_remove, to_plot, iteration)

        logger.info(f"Review: removed {len(to_remove)} pseudo-labels (known_flares: {len(known_flare_samples)}, unlabeled: {len(unlabeled_samples)} reviewed)")
        return len(to_remove)

    def _log_top_removal_candidates(self, iteration: int) -> None:
        """
        Log and plot the top removal candidates from pseudo-labeled samples.

        This helps identify samples the model is most uncertain about:
        - Pseudo-neg with highest P(flare): Most likely to be a missed flare
        - Pseudo-pos with lowest P(flare): Most likely to be a false positive

        Parameters
        ----------
        iteration : int
            Current iteration number.
        """
        # Collect pseudo-pos and pseudo-neg samples
        pseudo_pos_samples: list[tuple[int, LabeledSample]] = []
        pseudo_neg_samples: list[tuple[int, LabeledSample]] = []

        for i, sample in enumerate(self.labeled_train):
            if sample.source == SampleSource.PSEUDO_POS:
                pseudo_pos_samples.append((i, sample))
            elif sample.source == SampleSource.PSEUDO_NEG:
                pseudo_neg_samples.append((i, sample))

        if not pseudo_pos_samples and not pseudo_neg_samples:
            logger.info("No pseudo-labeled samples to check for removal candidates")
            return

        # Extract features for all pseudo-labeled samples in single batch
        pos_indices = [s.index for _, s in pseudo_pos_samples]
        neg_indices = [s.index for _, s in pseudo_neg_samples]
        all_indices = pos_indices + neg_indices

        all_features = extract_features_array(self.unlabeled_samples, all_indices, self.feature_cols)
        all_probs = self.model.predict_proba(all_features)[:, 1]

        # Split predictions back
        n_pos = len(pos_indices)
        pos_probs = all_probs[:n_pos] if n_pos > 0 else np.array([])
        neg_probs = all_probs[n_pos:] if len(neg_indices) > 0 else np.array([])

        # Find worst pseudo-pos (lowest P(flare) - most likely false positive)
        worst_pseudo_pos: tuple[LabeledSample, float] | None = None
        if len(pos_probs) > 0:
            min_idx = int(np.argmin(pos_probs))
            worst_pseudo_pos = (pseudo_pos_samples[min_idx][1], float(pos_probs[min_idx]))

        # Find worst pseudo-neg (highest P(flare) - most likely missed flare)
        worst_pseudo_neg: tuple[LabeledSample, float] | None = None
        if len(neg_probs) > 0:
            max_idx = int(np.argmax(neg_probs))
            worst_pseudo_neg = (pseudo_neg_samples[max_idx][1], float(neg_probs[max_idx]))

        # Log results
        if worst_pseudo_pos:
            sample, prob = worst_pseudo_pos
            logger.info(f"Top removal candidate (pseudo_pos): idx={sample.index}, P(flare)={prob:.4f}, original_conf={sample.confidence:.4f}")

        if worst_pseudo_neg:
            sample, prob = worst_pseudo_neg
            logger.info(f"Top removal candidate (pseudo_neg): idx={sample.index}, P(flare)={prob:.4f}, original_conf={sample.confidence:.4f}")

        # Plot both candidates
        if worst_pseudo_pos and self.config.plot_samples:
            sample, prob = worst_pseudo_pos
            plot_sample(
                sample.index,
                self.unlabeled_samples,
                self.output_dir,
                "top_removal_candidates",
                self.config,
                action="candidate_pseudo_pos",
                dataset=self.unlabeled_dataset,
                probability=prob,
                iteration=iteration,
            )

        if worst_pseudo_neg and self.config.plot_samples:
            sample, prob = worst_pseudo_neg
            plot_sample(
                sample.index,
                self.unlabeled_samples,
                self.output_dir,
                "top_removal_candidates",
                self.config,
                action="candidate_pseudo_neg",
                dataset=self.unlabeled_dataset,
                probability=prob,
                iteration=iteration,
            )

    def _check_sample_for_removal(
        self,
        sample: LabeledSample,
        current_prob: float,
        bootstrap_stats: tuple[float, float, float],
    ) -> tuple[bool, str]:
        """
        Check if a sample should be removed based on current predictions.

        Parameters
        ----------
        sample : LabeledSample
            The sample to check.
        current_prob : float
            Main model's predicted probability.
        bootstrap_stats : tuple[float, float, float]
            Pre-computed (mean, std, consensus) from bootstrap models.

        Returns
        -------
        tuple[bool, str]
            (should_remove, reason)
        """
        thresholds = self.config.thresholds
        bootstrap_mean, bootstrap_std, _ = bootstrap_stats

        # Check pseudo-positives
        if sample.label == 1 and sample.source == SampleSource.PSEUDO_POS:

            # Model changed its mind?
            if current_prob < thresholds.pseudo_pos_removal_prob:
                return True, f"prob dropped: was {sample.confidence:.3f}, now {current_prob:.3f}"

            # Bootstrap diverged?
            if bootstrap_std > thresholds.bootstrap_instability_std and bootstrap_mean < thresholds.bootstrap_instability_mean:
                return True, f"unstable: std={bootstrap_std:.3f}, mean={bootstrap_mean:.3f}"

            return False, ""

        # Check pseudo-negatives
        if sample.label == 0 and sample.source == SampleSource.PSEUDO_NEG:

            # Model changed its mind? (less strict)
            if current_prob > thresholds.pseudo_neg_promotion_prob:
                return True, f"prob rose: was {1-sample.confidence:.3f}, now {current_prob:.3f}"
            return False, ""

        # Check seed negatives (if review_seed_negatives is enabled)
        if sample.label == 0 and sample.source == SampleSource.SEED:

            # Very high threshold for removing seed samples - model must be very confident
            if current_prob > thresholds.seed_neg_removal_prob and bootstrap_mean > thresholds.seed_neg_removal_bootstrap_mean:
                return True, f"seed neg looks like flare: PR={current_prob:.3f}, bootstrap_mean={bootstrap_mean:.3f}"
            return False, ""

        return False, ""

    def _update_sample_weight(
        self,
        sample: LabeledSample,
        current_prob: float,
        bootstrap_stats: tuple[float, float, float],
    ) -> None:
        """
        Update sample weight based on current prediction confidence.

        Parameters
        ----------
        sample : LabeledSample
            The sample to update.
        current_prob : float
            Main model's predicted probability.
        bootstrap_stats : tuple[float, float, float]
            Pre-computed (mean, std, consensus) from bootstrap models.
        """
        if sample.label == 1 and sample.source == SampleSource.PSEUDO_POS:
            bootstrap_mean = bootstrap_stats[0]
            new_confidence = (current_prob + bootstrap_mean) / 2
            if sample.confidence > 0:
                sample.weight = min(1.0, sample.weight * (new_confidence / sample.confidence))

    def _compute_pseudo_label_weight(self, source: SampleSource) -> float:
        """
        Compute weight for pseudo-labeled samples based on training stability.

        Weight increases with successful iterations (model becoming more stable).
        Positives get lower initial weight and slower increment (more conservative).

        Parameters
        ----------
        source : SampleSource
            Either PSEUDO_POS or PSEUDO_NEG.

        Returns
        -------
        float
            Sample weight in [initial_weight, max_weight].
        """
        thresholds = self.config.thresholds
        if source == SampleSource.PSEUDO_POS:
            return min(
                thresholds.max_pseudo_pos_weight,
                thresholds.initial_pseudo_pos_weight + self.n_successful_iters * thresholds.weight_increment * PSEUDO_POS_WEIGHT_INCREMENT_SCALE,
            )
        else:  # PSEUDO_NEG
            return min(
                1.0,
                thresholds.initial_pseudo_neg_weight + self.n_successful_iters * thresholds.weight_increment,
            )

    def _format_metrics_log(self, metrics: dict, prefix: str = "") -> str:
        """
        Format metrics dict for consistent logging.

        Parameters
        ----------
        metrics : dict
            Metrics dictionary with keys like 'recall', 'precision', 'auc', etc.
        prefix : str
            Optional prefix for the log line (e.g., "Validation: ", "Freaky: ").

        Returns
        -------
        str
            Formatted metrics string.
        """
        return (
            f"{prefix}RE={metrics.get('recall', 0):.3f}, PR={metrics.get('precision', 0):.3f}, "
            f"ROC-AUC={metrics.get('auc', 0):.3f}, PR-AUC={metrics.get('pr_auc', 0):.3f}, "
            f"ICE={metrics.get('ice', 0):.4f}, LL={metrics.get('logloss', 0):.4f}, "
            f"Brier={metrics.get('brier', 0):.4f}"
        )

    def _balance_class_weights(self) -> dict[int, float] | None:
        """
        Compute class weights if the class imbalance exceeds threshold.

        Returns class_weight dict or None.
        """
        eff_pos, eff_neg = self._compute_effective_sizes()
        logger.info(f"Effective train size: pos={eff_pos:.1f}, neg={eff_neg:.1f}")

        if eff_pos == 0:
            return None

        if eff_neg / eff_pos > self.config.class_imbalance_threshold:
            class_weight = {0: 1.0, 1: eff_neg / eff_pos / self.config.class_weight_divisor}
            logger.info(f"Class weight adjusted: {class_weight}")
            return class_weight

        return None

    def _adjust_thresholds(self) -> None:
        """Adjust pseudo-labeling thresholds based on training stability."""
        thresholds = self.config.thresholds

        # Skip if adaptive thresholds disabled
        if not thresholds.enable_adaptive_thresholds:
            return

        if self.n_successful_iters >= thresholds.relax_successful_iters:
            # Model stable - relax thresholds
            self.pseudo_pos_threshold = max(
                thresholds.min_pseudo_pos_threshold,
                self.pseudo_pos_threshold - thresholds.relax_pos_delta,
            )
            self.pseudo_neg_threshold = min(
                thresholds.max_pseudo_neg_threshold,
                self.pseudo_neg_threshold + thresholds.relax_neg_delta,
            )
            self.max_pseudo_pos_per_iter = min(
                thresholds.max_pseudo_pos_cap,
                self.max_pseudo_pos_per_iter + thresholds.relax_max_pos_delta,
            )
            logger.info(f"Thresholds relaxed: pos>{self.pseudo_pos_threshold:.3f}, " f"neg<{self.pseudo_neg_threshold:.3f}")

        elif self.n_successful_iters == 0:
            # Just rolled back - tighten
            self._tighten_thresholds()

    def _compute_enrichment_factor(self, main_preds: np.ndarray | None = None) -> tuple[float, float]:
        """
        Phase 11: Compute enrichment factor.

        Parameters
        ----------
        main_preds : np.ndarray, optional
            Precomputed predictions. If None, will compute them using batched prediction.

        Returns
        -------
        tuple[float, float]
            (enrichment_factor, estimated_flares_in_top10k)
        """
        if self.model is None:
            return 0.0, 0.0

        # Use precomputed predictions or compute them using batched prediction
        if main_preds is None:
            # logger.info("Computing enrichment (batched prediction)...")
            proba = self._predict_unlabeled_features_batched(self.model)
        else:
            proba = main_preds

        top_k = self.config.stopping.top_k_candidates
        # Use argpartition for O(n) instead of argsort O(n log n)
        top_k_indices = np.argpartition(proba, -top_k)[-top_k:]
        top_k_probs = proba[top_k_indices]

        # Estimated flares in top-K
        estimated_flares = float(np.sum(top_k_probs))

        # Random baseline at assumed prevalence
        random_baseline = top_k * self.config.assumed_prevalence

        enrichment = estimated_flares / random_baseline if random_baseline > 0 else 0.0

        return enrichment, estimated_flares

    def _save_top_candidates(self, main_preds: np.ndarray, iteration: int) -> None:
        """
        Save top-K flare candidates to parquet file.

        Parameters
        ----------
        main_preds : np.ndarray
            Predicted probabilities for all unlabeled samples.
        iteration : int
            Current iteration number.
        """
        top_k = self.config.stopping.top_k_candidates_save
        n_samples = len(main_preds)
        actual_k = min(top_k, n_samples)

        # Get top-K indices using argpartition for efficiency
        top_k_indices = np.argpartition(main_preds, -actual_k)[-actual_k:]
        # Sort by probability descending
        sorted_order = np.argsort(-main_preds[top_k_indices])
        top_k_indices = top_k_indices[sorted_order]
        top_k_probs = main_preds[top_k_indices]

        # Build DataFrame with row index, ID (if available), and probability
        data = {
            "row_index": top_k_indices.tolist(),
            "probability": top_k_probs.tolist(),
        }

        # Add ID column if available
        if "id" in self.unlabeled_samples.columns:
            ids = [self.unlabeled_samples[int(idx), "id"] for idx in top_k_indices]
            data["id"] = ids

        top_df = pl.DataFrame(data)

        # Save to parquet
        candidates_dir = self.output_dir / "top_candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)
        path = candidates_dir / f"top_{top_k}_iter_{iteration:03d}.parquet"
        top_df.write_parquet(path, compression="zstd")
        logger.info(f"Top-{actual_k} candidates saved to {path}")

    def _check_stopping_criteria(
        self,
        iteration: int,
        validation_metrics: dict,
        pseudo_pos_added: int,
    ) -> str | None:
        """
        Check stopping criteria. Returns reason string or None to continue.

        Priority order:
        1. Time-based (default: 5 hours)
        2. ICE-based (if target_ice configured)
        3. Plateau detection
        4. Max iterations (optional)
        5. Instability
        """
        stopping = self.config.stopping

        # 1. Time-based stopping (if max_time_hours is set)
        if self.config.max_time_hours is not None:
            elapsed_hours = self._get_elapsed_hours()
            if elapsed_hours >= self.config.max_time_hours:
                logger.info(f"Time limit reached: {elapsed_hours:.2f} hours >= {self.config.max_time_hours} hours")
                return "TIME_LIMIT"

        # 2. ICE-based stopping (if target_ice is set)
        if self.config.target_ice is not None:
            current_ice = validation_metrics.get("ice", 1.0)
            if current_ice <= self.config.target_ice:
                logger.info(f"ICE target reached: {current_ice:.4f} <= {self.config.target_ice}")
                return "ICE_TARGET_REACHED"

        # 3. Plateau check
        if len(self.metrics_history) >= stopping.plateau_window:
            # Check both recall and ICE for plateau
            recent_recalls = [m.validation_recall for m in self.metrics_history[-stopping.plateau_window :]]
            recent_ices = [m.validation_ice for m in self.metrics_history[-stopping.plateau_window :] if m.validation_ice > 0]

            recall_range = max(recent_recalls) - min(recent_recalls)
            ice_stable = len(recent_ices) >= 5 and (max(recent_ices) - min(recent_ices)) < stopping.plateau_ice_range

            if recall_range < stopping.plateau_recall_range and ice_stable:
                recent_pos_added = [m.pseudo_pos_added for m in self.metrics_history[-stopping.plateau_min_zero_pos_iters :]]
                if sum(recent_pos_added) == 0:
                    return "PLATEAU"

        # 4. Max iterations (optional - only if explicitly set)
        if self.config.max_iters is not None and iteration >= self.config.max_iters:
            return "MAX_ITERATIONS"

        # 5. Instability
        recent_rollbacks = sum(1 for r in self.rollback_history if r > iteration - stopping.rollback_window)
        if recent_rollbacks > stopping.max_rollbacks_for_instability:
            return "UNSTABLE"

        return None

    # =========================================================================
    # Main Run Method
    # =========================================================================

    def run(self) -> dict:
        """
        Run the full active learning pipeline.

        Returns
        -------
        dict
            Results dictionary with:
            - final_model: trained model
            - labeled_train: final labeled training samples
            - candidates: dict of candidate DataFrames by purity level
            - metrics_history: list of iteration metrics
            - best_iteration: int
            - stop_reason: str
        """
        # Phase 0: Initialize
        self.initialize()

        stop_reason = None
        iteration = 0

        # Use a very large number if max_iters is None (time-based stopping will handle it)
        max_iterations = self.config.max_iters if self.config.max_iters is not None else FALLBACK_MAX_ITERATIONS

        for iteration in range(1, max_iterations + 1):
            elapsed = self._get_elapsed_hours()
            logger.info("=" * 60)
            logger.info(f"ITERATION {iteration} (elapsed: {elapsed:.2f} hours)")
            logger.info("=" * 60)

            # Clean up expired bans at start of each iteration
            n_active_bans = self._cleanup_expired_bans(iteration)
            if n_active_bans > 0:
                logger.info(f"Ban list: {n_active_bans} samples still banned")

            # Update feature set if warmup is enabled (gradual feature introduction)
            features_expanded = self._update_feature_warmup(iteration)
            if features_expanded:
                # Invalidate all caches that depend on feature_cols
                self._training_data_dirty = True
                self._training_data_cache = None
                self._unlabeled_view = None
                self._validation_features_cache = None
                self._held_out_features_cache = None
                self._tracked_features_cache = None
                self._freaky_features_cache = None

            # Validate and check for early stopping
            validation_metrics, rollback_occurred, stop = self._validate_and_check_stopping(iteration)

            # Report freaky held-out metrics (if configured)
            self._compute_freaky_metrics(iteration)

            if stop:
                stop_reason = stop
                break
            if rollback_occurred:
                continue  # Skip rest of iteration after rollback

            # Train bootstrap ensemble for consensus estimation
            logger.info("Training bootstrap models...")
            self._train_bootstrap_ensemble(iteration)

            # Predict on unlabeled pool
            main_preds, bootstrap_preds, prediction_indices = self._predict_unlabeled_pool()

            # Pseudo-label negatives (aggressive)
            pseudo_neg_added = self._pseudo_label_negatives(main_preds, bootstrap_preds, prediction_indices, iteration)

            # Pseudo-label positives (conservative)
            pseudo_pos_added = self._pseudo_label_positives(main_preds, bootstrap_preds, prediction_indices, iteration)

            # Save top-K flare candidates for this iteration (configured in stopping.top_k_candidates_save)
            self._save_top_candidates(main_preds, iteration)

            # Compute enrichment BEFORE deleting main_preds (avoids redundant full prediction)
            should_compute_enrichment = iteration == 1 or (self.config.enrichment_every_n_iters > 0 and iteration % self.config.enrichment_every_n_iters == 0)
            if should_compute_enrichment:
                enrichment, estimated_flares_top10k = self._compute_enrichment_factor(main_preds)
            else:
                # Use last known enrichment or 0
                enrichment = self.metrics_history[-1].enrichment_factor if self.metrics_history else 0.0
                estimated_flares_top10k = self.metrics_history[-1].estimated_flares_top10k if self.metrics_history else 0.0

            # Log prediction distribution stats
            n_above_90 = int((main_preds > 0.90).sum())
            n_above_99 = int((main_preds > 0.99).sum())
            max_prob = float(main_preds.max())
            logger.info(f"Predictions: {n_above_90:,} with P>0.90, {n_above_99:,} with P>0.99, max(P)={max_prob:.6f}")

            # Clear predictions (no longer needed)
            del main_preds, bootstrap_preds, prediction_indices
            clean_ram()

            # Review and remove inconsistent pseudo-labels
            pseudo_removed = self._review_pseudo_labels(iteration)

            # Log and plot top removal candidates
            self._log_top_removal_candidates(iteration)

            # Balance class weights if needed
            class_weight = self._balance_class_weights()

            # Build training data once for both retraining and OOB
            logger.info("Building training data...")
            features, labels, weights = self._build_training_data()

            # Retrain model with updated data (with training curve plot for main model only)
            logger.info("Retraining main model...")
            training_curves_dir = self.output_dir / "training_curves"
            training_curves_dir.mkdir(parents=True, exist_ok=True)
            plot_file = training_curves_dir / f"training_iter{iteration:03d}.html"
            self.model = train_model(
                features,
                labels,
                weights,
                class_weight=class_weight,
                config=self.config,
                random_state=self.random_state + len(self.metrics_history),
                plot_file=plot_file,
            )

            # Save feature importances plot (top 20 features)
            fi_dir = self.output_dir / "feature_importances"
            fi_dir.mkdir(parents=True, exist_ok=True)
            fi_plot_file = fi_dir / f"feature_importance_iter{iteration:03d}.png"
            plot_model_feature_importances(
                model=self.model,
                columns=self.feature_cols,
                model_name=f"Iteration {iteration}",
                num_factors=20,
                show_plots=False,
                plot_file=str(fi_plot_file),
            )

            # Adjust thresholds based on stability
            self._adjust_thresholds()

            # Compute OOB metrics for stability monitoring (reuse training data)
            # Note: enrichment was already computed earlier (before del main_preds)
            if self.bootstrap_indices_list:
                oob_metrics = compute_oob_metrics(self.bootstrap_models, features, labels, self.bootstrap_indices_list)
                logger.info(
                    f"OOB metrics: recall={oob_metrics['oob_recall']:.3f}, "
                    f"precision={oob_metrics['oob_precision']:.3f}, "
                    f"coverage={oob_metrics['oob_coverage']:.2%}"
                )

                # Check OOB/validation divergence
                divergence = abs(oob_metrics["oob_recall"] - validation_metrics["recall"])
                if divergence > self.config.oob_divergence_warning:
                    logger.warning(f"OOB/validation recall divergence: {divergence:.3f} — possible instability")

            # Clean up bootstrap models and training data (no longer needed this iteration)
            self.bootstrap_models = []
            self.bootstrap_indices_list = []
            del features, labels, weights
            clean_ram()

            # Get tracked rowid probabilities (log once per iteration here)
            tracked_pos_prob, tracked_neg_prob = self._get_tracked_probabilities(log=True)

            # Phase 12: Logging and checkpointing
            counts = self._count_by_source()
            eff_pos, eff_neg = self._compute_effective_sizes()
            recent_rollbacks = sum(1 for r in self.rollback_history if r > iteration - 10)

            metrics = IterationMetrics(
                iteration=iteration,
                train_total=len(self.labeled_train),
                train_seed=counts[SampleSource.SEED],
                train_pseudo_pos=counts[SampleSource.PSEUDO_POS],
                train_pseudo_neg=counts[SampleSource.PSEUDO_NEG],
                train_forced_pos=counts[SampleSource.FORCED_POS],
                train_forced_neg=counts[SampleSource.FORCED_NEG],
                effective_pos=eff_pos,
                effective_neg=eff_neg,
                validation_recall=validation_metrics["recall"],
                validation_precision=validation_metrics["precision"],
                validation_auc=validation_metrics["auc"],
                validation_f1=validation_metrics["f1"],
                validation_logloss=validation_metrics.get("logloss", 0.0),
                validation_brier=validation_metrics.get("brier", 0.0),
                validation_ice=validation_metrics.get("ice", 0.0),
                validation_pr_auc=validation_metrics.get("pr_auc", 0.0),
                enrichment_factor=enrichment,
                estimated_flares_top10k=estimated_flares_top10k,
                pseudo_pos_threshold=self.pseudo_pos_threshold,
                pseudo_neg_threshold=self.pseudo_neg_threshold,
                pseudo_pos_added=pseudo_pos_added,
                pseudo_neg_added=pseudo_neg_added,
                pseudo_removed=pseudo_removed,
                n_successful_iters=self.n_successful_iters,
                n_rollbacks_recent=recent_rollbacks,
                elapsed_hours=self._get_elapsed_hours(),
                tracked_positive_rowid_prob=tracked_pos_prob,
                tracked_negative_rowid_prob=tracked_neg_prob,
            )
            self.metrics_history.append(metrics)
            self._save_checkpoint(iteration, asdict(metrics))

            # Log summary
            logger.info("=" * 60)
            logger.info(f"ITERATION {iteration} COMPLETE")
            train_parts = f"seed:{counts[SampleSource.SEED]}, pseudo_pos:{counts[SampleSource.PSEUDO_POS]}, pseudo_neg:{counts[SampleSource.PSEUDO_NEG]}"
            if counts[SampleSource.FORCED_POS] > 0 or counts[SampleSource.FORCED_NEG] > 0:
                train_parts += f", forced_pos:{counts[SampleSource.FORCED_POS]}, forced_neg:{counts[SampleSource.FORCED_NEG]}"
            logger.info(f"  Train: {len(self.labeled_train)} ({train_parts})")
            logger.info(f"  {self._format_metrics_log(validation_metrics, 'Validation: ')}")
            logger.info(f"  Enrichment: {enrichment:.1f}x")
            logger.info(f"  Elapsed: {self._get_elapsed_hours():.2f} hours")
            logger.info(f"  Successful iters: {self.n_successful_iters}")
            logger.info("=" * 60)

            # Update previous metrics (for next iteration's degradation check)
            self.prev_validation_recall = validation_metrics["recall"]
            self.prev_validation_precision = validation_metrics["precision"]
            self.prev_validation_ice = validation_metrics.get("ice", 1.0)
            # Update tracked sample probabilities for decision-making
            self.prev_tracked_pos_prob, self.prev_tracked_neg_prob = self._get_tracked_probabilities()

            # Check stopping criteria
            stop_reason = self._check_stopping_criteria(iteration, validation_metrics, pseudo_pos_added)
            if stop_reason:
                break

        # Finalization
        return self._finalize(stop_reason or "TIME_LIMIT")

    def _load_best_model(self) -> None:
        """Load best model from checkpoint if available."""
        if self.best_checkpoint is not None:
            logger.info(f"Loading best model from iteration {self.best_checkpoint.iteration}")
            self.model = joblib.load(self.best_checkpoint.model_path)

    def _save_model_and_training_data(self) -> pl.DataFrame:
        """Save final model and labeled training set. Returns labeled_train as DataFrame."""
        final_model_path = self.output_dir / "final_model.joblib"
        joblib.dump(self.model, final_model_path)
        logger.info(f"Final model saved to {final_model_path}")

        labeled_train_data = [asdict(s) for s in self.labeled_train]
        labeled_train_df = pl.DataFrame(labeled_train_data)
        labeled_train_path = self.output_dir / "labeled_train.parquet"
        labeled_train_df.write_parquet(labeled_train_path, compression="zstd")
        logger.info(f"Labeled train saved to {labeled_train_path}")

        return labeled_train_df

    def _generate_and_save_candidates(self) -> dict[str, pl.DataFrame]:
        """Generate candidate lists at different purity levels."""
        logger.info("Generating candidate lists (batched prediction)...")
        final_proba = self._predict_unlabeled_features_batched(self.model)

        # Clear caches AFTER prediction to avoid 54GB memory spike from recreating view
        self.clear_caches()

        pool_with_proba = self.unlabeled_samples.with_columns(pl.Series("proba", final_proba))

        candidates = {
            "high_purity": pool_with_proba.filter(pl.col("proba") > 0.95),
            "balanced": pool_with_proba.filter(pl.col("proba") > 0.80),
            "high_recall": pool_with_proba.filter(pl.col("proba") > 0.50),
        }

        del pool_with_proba, final_proba
        clean_ram()

        for name, df in candidates.items():
            path = self.output_dir / f"candidates_{name}.parquet"
            df.write_parquet(path, compression="zstd")
            logger.info(f"  {name}: {len(df):,} candidates saved to {path}")

        return candidates

    def _save_metrics(self, honest_metrics: dict) -> None:
        """Save metrics history and honest held-out metrics."""
        metrics_path = self.output_dir / "metrics_history.json"
        with open(metrics_path, "w") as f:
            json.dump([asdict(m) for m in self.metrics_history], f, indent=2, default=str)
        logger.info(f"Metrics history saved to {metrics_path}")

        honest_metrics_path = self.output_dir / "honest_held_out_metrics.json"
        with open(honest_metrics_path, "w") as f:
            json.dump(honest_metrics, f, indent=2)
        logger.info(f"Honest held-out metrics saved to {honest_metrics_path}")

    def _log_final_summary(self, honest_metrics: dict, candidates: dict[str, pl.DataFrame]) -> None:
        """Log final pipeline summary."""
        best_metrics = self.best_checkpoint.metrics if self.best_checkpoint else {}
        best_iter = self.best_checkpoint.iteration if self.best_checkpoint else 0

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Best iteration (by validation ICE): {best_iter}")
        logger.info(f"Best validation recall: {best_metrics.get('recall', 0):.3f}")
        logger.info(f"Best validation precision: {best_metrics.get('precision', 0):.3f}")
        logger.info("-" * 40)
        logger.info(f"HONEST held-out recall: {honest_metrics.get('recall', 0.0):.3f}")
        logger.info(f"HONEST held-out precision: {honest_metrics.get('precision', 0.0):.3f}")
        logger.info(f"HONEST held-out ICE: {honest_metrics.get('ice', 0.0):.4f}")
        logger.info("-" * 40)
        logger.info(f"Final train size: {len(self.labeled_train)}")
        logger.info(f"Candidates (high purity): {len(candidates['high_purity']):,}")
        logger.info(f"Candidates (balanced): {len(candidates['balanced']):,}")
        logger.info(f"Candidates (high recall): {len(candidates['high_recall']):,}")
        logger.info("=" * 60)

    def _finalize(self, stop_reason: str) -> dict:
        """Finalize pipeline and save results."""
        logger.info("=" * 60)
        logger.info("FINALIZATION")
        logger.info(f"Stop reason: {stop_reason}")
        logger.info("=" * 60)

        # Load best model
        self._load_best_model()

        # Compute HONEST held-out metrics (first and only time!)
        logger.info("Computing honest held-out metrics (never used for training decisions)...")
        honest_metrics = self._compute_held_out_metrics_final()
        logger.info(self._format_metrics_log(honest_metrics, "HONEST HELD-OUT: "))

        # Save artifacts
        labeled_train_df = self._save_model_and_training_data()
        candidates = self._generate_and_save_candidates()
        self._save_metrics(honest_metrics)
        self._log_final_summary(honest_metrics, candidates)

        return {
            "final_model": self.model,
            "labeled_train": labeled_train_df,
            "candidates": candidates,
            "metrics_history": self.metrics_history,
            "best_iteration": self.best_checkpoint.iteration if self.best_checkpoint else 0,
            "stop_reason": stop_reason,
            "honest_held_out_metrics": honest_metrics,
        }


# =============================================================================
# Convenience Function
# =============================================================================


def run_active_learning_pipeline(
    unlabeled_samples: pl.DataFrame,
    known_flares: pl.DataFrame,
    config: PipelineConfig | None = None,
    output_dir: Path | str = "active_learning_output",
    random_state: int = 42,
    unlabeled_dataset=None,
    known_flares_dataset=None,
    freaky_held_out_indices: list[int] | None = None,
    forced_positive_indices: list[int] | None = None,
    forced_negative_indices: list[int] | None = None,
) -> dict:
    """
    Run zero-expert self-training pipeline.

    Parameters
    ----------
    unlabeled_samples : pl.DataFrame
        Large unlabeled dataset (features for the unlabeled pool).
        Expected columns: feature columns, optionally 'id'.
    known_flares : pl.DataFrame
        Known flares dataset (samples with class=1).
        Expected columns: feature columns, 'class', optionally 'id'.
    config : PipelineConfig, optional
        Pipeline configuration. Uses defaults if not provided.
    output_dir : Path or str
        Directory for saving checkpoints and results.
    random_state : int
        Random seed for reproducibility.
    unlabeled_dataset : HuggingFace Dataset, optional
        Original light curve data for unlabeled_samples, used for plotting.
        Access pattern: dataset[row_index]
    known_flares_dataset : HuggingFace Dataset, optional
        Original light curve data for known_flares, used for plotting.
        Access pattern: dataset[row_index]
    freaky_held_out_indices : list[int], optional
        Row indices in known_flares to exclude from the train/val/held-out split.
        These "freaky" samples are reported separately each iteration for diagnostic purposes.
    forced_positive_indices : list[int], optional
        Row indices in unlabeled_samples to force as positives in training.
        These get weight=1.0 (same as known_flares) and are protected from relabeling/banning.
    forced_negative_indices : list[int], optional
        Row indices in unlabeled_samples to force as negatives in training.
        These get weight=1.0 (same as known_flares) and are protected from relabeling/banning.

    Returns
    -------
    dict
        Results dictionary with:
        - final_model: trained model
        - labeled_train: pl.DataFrame of final labeled training samples
        - candidates: dict of candidate DataFrames by purity level
        - metrics_history: list of IterationMetrics
        - best_iteration: int
        - stop_reason: str
    """
    # Apply forced indices to config
    if config is None:
        config = PipelineConfig()
    if forced_positive_indices is not None:
        config.forced_positive_indices = forced_positive_indices
    if forced_negative_indices is not None:
        config.forced_negative_indices = forced_negative_indices

    pipeline = ActiveLearningPipeline(
        unlabeled_samples=unlabeled_samples,
        known_flares=known_flares,
        config=config,
        output_dir=output_dir,
        random_state=random_state,
        unlabeled_dataset=unlabeled_dataset,
        known_flares_dataset=known_flares_dataset,
        freaky_held_out_indices=freaky_held_out_indices,
    )
    return pipeline.run()


# =============================================================================
# Logging Setup Helper
# =============================================================================


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | str | None = None,
) -> None:
    """
    Setup logging for the pipeline.

    Parameters
    ----------
    level : int
        Logging level (e.g., logging.INFO, logging.DEBUG).
    log_file : Path or str, optional
        Path to log file. If None, logs only to console.
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example usage (requires actual data)
    setup_logging(level=logging.INFO)

    # Load your data
    # unlabeled_samples = pl.read_parquet("path/to/unlabeled_samples.parquet")
    # known_flares = pl.read_parquet("path/to/known_flares.parquet")

    # Run pipeline
    # results = run_active_learning_pipeline(
    #     unlabeled_samples=unlabeled_samples,
    #     known_flares=known_flares,
    #     output_dir="active_learning_output",
    # )

    print("Pipeline module loaded. Use run_active_learning_pipeline() to start.")

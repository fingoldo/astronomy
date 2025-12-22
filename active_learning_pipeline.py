"""
Zero-Expert Self-Training Pipeline v2 for Stellar Flare Detection.

This module implements an active learning pipeline that iteratively trains
a classifier to detect stellar flares in ZTF light curves using only
a small set of known flares and a large unlabeled dataset.

Key features:
- Asymmetric pseudo-labeling (aggressive for negatives, conservative for positives)
- Bootstrap consensus for pseudo-label validation
- Held-out set as the ultimate arbiter
- Automatic rollback on degradation
- Adaptive thresholds based on model stability
"""

from __future__ import annotations

import json
import logging
import psutil
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from os.path import join
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import polars as pl
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import recall_score, precision_score, roc_auc_score, f1_score, brier_score_loss, log_loss, average_precision_score
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

# Optional report_model_perf integration
try:
    from mlframe.training_old import report_model_perf

    REPORT_PERF_AVAILABLE = True
except ImportError:
    REPORT_PERF_AVAILABLE = False
    report_model_perf = None

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


logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions for Improved Pipeline
# =============================================================================


def _random_split(
    n_samples: int,
    train_ratio: float,
    val_ratio: float,
    random_state: int,
) -> tuple[list[int], list[int], list[int]]:
    """Random fallback split for stratified_flare_split."""
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(n_samples)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    return (
        shuffled[:n_train].tolist(),
        shuffled[n_train : n_train + n_val].tolist(),
        shuffled[n_train + n_val :].tolist(),
    )


def stratified_flare_split(
    oos_features: pl.DataFrame,
    train_ratio: float = 0.10,
    val_ratio: float = 0.40,
    held_out_ratio: float = 0.50,
    stratify_cols: list[str] | None = None,
    random_state: int = 42,
) -> tuple[list[int], list[int], list[int]]:
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
    oos_features : pl.DataFrame
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

    Returns
    -------
    tuple[list[int], list[int], list[int]]
        (train_indices, val_indices, held_out_indices) as lists of row indices.

    Notes
    -----
    If stratification fails (e.g., too few samples per stratum), falls back
    to random splitting with a warning.

    References
    ----------
    - Settles, B. (2012). Active Learning. Morgan & Claypool.
      Chapter on query strategies and sample selection.
    """
    n_samples = len(oos_features)
    all_idx = np.arange(n_samples)

    # Determine stratification columns
    if stratify_cols is None:
        stratify_cols = []
        if "norm_amplitude_sigma" in oos_features.columns:
            stratify_cols.append("norm_amplitude_sigma")
        if "npoints" in oos_features.columns:
            stratify_cols.append("npoints")

    # If no stratification possible, use random split
    if not stratify_cols:
        logger.warning("No stratification columns found, using random split")
        return _random_split(n_samples, train_ratio, val_ratio, random_state)

    # Build stratification labels
    try:
        # Extract each column separately from polars to avoid shape issues
        strat_arrays = []
        for col in stratify_cols:
            col_data = oos_features[col].to_numpy()
            # Ensure 1D and float
            col_data = np.asarray(col_data, dtype=np.float64).flatten()
            strat_arrays.append(col_data)

        # Stack into 2D array (n_samples, n_cols)
        if len(strat_arrays) == 1:
            strat_data = strat_arrays[0].reshape(-1, 1)
        else:
            strat_data = np.column_stack(strat_arrays)

        # Use quantile binning to create strata
        n_bins = min(5, n_samples // 10)  # At least 10 samples per bin
        if n_bins < 2:
            raise ValueError("Too few samples for stratification")

        binner = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
        strata = np.asarray(binner.fit_transform(strat_data))

        # Combine multiple columns into single stratum label
        if strata.shape[1] > 1:
            strata_labels = (strata[:, 0] * n_bins + strata[:, 1]).astype(int)
        else:
            strata_labels = strata.ravel().astype(int)

        # First split: train vs rest
        train_idx, rest_idx = train_test_split(
            all_idx,
            train_size=train_ratio,
            stratify=strata_labels,
            random_state=random_state,
        )

        # Second split: val vs held-out from rest
        rest_strata = strata_labels[rest_idx]
        val_frac = val_ratio / (val_ratio + held_out_ratio)

        val_idx, held_out_idx = train_test_split(
            rest_idx,
            train_size=val_frac,
            stratify=rest_strata,
            random_state=random_state,
        )

        logger.info(f"Stratified split: train={len(train_idx)}, val={len(val_idx)}, " f"held_out={len(held_out_idx)}")

        return train_idx.tolist(), val_idx.tolist(), held_out_idx.tolist()

    except Exception as e:
        logger.warning(f"Stratified split failed ({e}), falling back to random")
        return _random_split(n_samples, train_ratio, val_ratio, random_state)


def get_adaptive_curriculum_weight(
    confidence: float,
    current_recall: float,
) -> float:
    """
    Compute sample weight using adaptive curriculum learning.

    Curriculum learning gradually introduces harder examples as the model
    improves. This function determines sample weights based on both the
    confidence of the pseudo-label AND the current model quality (recall).

    The key insight: Early in training (low recall), the model makes many
    mistakes. Adding low-confidence pseudo-labels amplifies these errors.
    As the model matures (high recall), it can benefit from harder examples.

    Phases
    ------
    Phase 1 (recall < 0.5):
        Only accept high-confidence pseudo-labels (>0.95).
        The model is still learning basic patterns.

    Phase 2 (0.5 <= recall < 0.65):
        Accept medium-confidence labels (>0.85) with reduced weight.
        The model is improving and can handle some uncertainty.

    Phase 3 (recall >= 0.65):
        Accept lower-confidence labels (>0.70) with weight proportional
        to confidence. The model is mature enough to learn from edge cases.

    Parameters
    ----------
    confidence : float
        Pseudo-label confidence in [0, 1]. For positives, this is P(flare).
        For negatives, this is 1 - P(flare).
    current_recall : float
        Current held-out recall in [0, 1]. Measures model maturity.

    Returns
    -------
    float
        Sample weight in [0, 1]. Weight of 0 means "don't include this sample".

    References
    ----------
    - Bengio et al. (2009). Curriculum Learning. ICML.
    - Kumar et al. (2010). Self-Paced Learning for Latent Variable Models.
    """
    if current_recall < 0.5:
        # Phase 1: Strict - only very confident pseudo-labels
        return 1.0 if confidence > 0.95 else 0.0

    elif current_recall < 0.65:
        # Phase 2: Medium - expand with reduced weights
        if confidence > 0.95:
            return 1.0
        elif confidence > 0.85:
            return 0.7
        else:
            return 0.0

    else:
        # Phase 3: Mature - use gradient weights
        if confidence > 0.95:
            return 1.0
        elif confidence > 0.70:
            return confidence  # Linear weight
        else:
            return 0.0


def select_hard_examples_simple(
    val_pool_indices: list[int],
    probas: np.ndarray,
    n_select: int = 3,
) -> list[int]:
    """
    Select hard examples with varying difficulty levels for diverse learning.

    In active learning, selecting only the "hardest" examples (closest to
    decision boundary) can lead to sampling similar examples repeatedly.
    This function selects examples from different difficulty strata to ensure
    the model sees diverse challenges.

    The approach is simpler than clustering-based methods but effective:
    we sort by hardness and sample from different quantiles.

    Parameters
    ----------
    val_pool_indices : list[int]
        Indices of samples in the validation pool.
    probas : np.ndarray
        Predicted probabilities for each sample in val_pool.
        For flares (positive class), P(flare).
    n_select : int, default 3
        Number of examples to select.

    Returns
    -------
    list[int]
        Selected indices from val_pool_indices.

    Notes
    -----
    Hardness is defined as |P - 0.5|: samples where the model is uncertain
    are considered "hard". We select:
    - 1 from the hardest (closest to 0.5)
    - 1 from medium difficulty
    - 1 from easier cases (to maintain baseline accuracy)
    """
    if len(val_pool_indices) <= n_select:
        return val_pool_indices

    # Compute hardness: distance from decision boundary
    # For detected flares (P > 0.5), hardest are those with P closest to 0.5
    # Only consider detected flares
    hardness = [(idx, i, abs(probas[i] - 0.5)) for i, idx in enumerate(val_pool_indices) if probas[i] > 0.5]

    if len(hardness) == 0:
        return []

    # Sort by hardness (ascending = hardest first)
    hardness.sort(key=lambda x: x[2])

    # Select from different strata
    n = len(hardness)
    selected_positions = [0]  # Hardest

    if n >= 3:
        selected_positions.append(n // 3)  # Medium-hard
        selected_positions.append(2 * n // 3)  # Easier

    elif n >= 2:
        selected_positions.append(n - 1)  # Easiest available

    # Get unique selections
    selected = []
    for pos in selected_positions:
        if pos < n and len(selected) < n_select:
            idx = hardness[pos][0]
            if idx not in selected:
                selected.append(idx)

    return selected[:n_select]


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
    big_features : pl.DataFrame
        Large unlabeled dataset (for pseudo-labeled negatives/positives).
    oos_features : pl.DataFrame
        Known flares dataset (for seed positives and val_pool samples).
    feature_cols : list[str]
        Feature column names to use.
    current_recall : float
        Current held-out recall for adaptive curriculum weighting.
    verbose : int
        Verbosity level.

    Notes
    -----
    mlframe will receive a DataFrame containing only the labeled samples,
    which is more memory-efficient than passing the full 94M dataset.
    """

    def __init__(
        self,
        labeled_samples: list,  # list[LabeledSample]
        big_features: pl.DataFrame,
        oos_features: pl.DataFrame,
        feature_cols: list[str],
        current_recall: float = 0.0,
        verbose: int = 1,
    ):
        if not MLFRAME_AVAILABLE:
            raise ImportError("mlframe is required for this feature. Install it first.")

        super().__init__(
            columns_to_drop={"_label", "_weight", "_source", "_confidence"},
            verbose=verbose,
        )
        self.labeled_samples = labeled_samples
        self.big_features = big_features
        self.oos_features = oos_features
        self.feature_cols = feature_cols
        self.current_recall = current_recall
        self._prepared_df: pl.DataFrame | None = None

    def add_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Build DataFrame from labeled samples.

        This method constructs the training DataFrame by extracting features
        from the appropriate source (oos_features or big_features) for each
        labeled sample. The result is cached for efficiency.

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

        rows = []
        for sample in self.labeled_samples:
            # Get features from appropriate source
            if sample.is_flare_source:
                source_df = self.oos_features
            else:
                source_df = self.big_features

            # Extract row as dict
            row_data = {}
            for col in self.feature_cols:
                row_data[col] = source_df[sample.index, col]

            # Add metadata
            row_data["_label"] = sample.label
            row_data["_confidence"] = sample.confidence
            row_data["_source"] = sample.source

            # Compute adaptive curriculum weight
            weight = get_adaptive_curriculum_weight(sample.confidence, self.current_recall)
            # Combine with sample's stored weight
            row_data["_weight"] = sample.weight * weight if weight > 0 else 0.0

            rows.append(row_data)

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
    big_features: pl.DataFrame,
    oos_features: pl.DataFrame,
    feature_cols: list[str],
    current_recall: float,
    iteration: int,
    output_dir: Path,
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
    big_features : pl.DataFrame
        Large dataset for feature lookup.
    oos_features : pl.DataFrame
        Known flares dataset for feature lookup.
    feature_cols : list[str]
        Feature column names.
    current_recall : float
        Current held-out recall for curriculum weighting.
    iteration : int
        Current iteration number.
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
        big_features=big_features,
        oos_features=oos_features,
        feature_cols=feature_cols,
        current_recall=current_recall,
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
class PipelineConfig:
    """Configuration for the active learning pipeline."""

    # Data splits
    n_train_flares_init: int = 10
    n_train_neg_init: int = 1000
    n_val_pool: int = 39
    n_held_out_flares: int = 50
    n_held_out_neg: int = 10000

    # Thresholds (mutable during training)
    pseudo_pos_threshold: float = 0.99
    pseudo_neg_threshold: float = 0.05
    consensus_threshold: float = 0.95

    # Limits per iteration
    max_pseudo_pos_per_iter: int = 10
    max_pseudo_neg_per_iter: int = 100
    n_bootstrap_models: int = 5  # Increased for better OOB coverage

    # Stopping criteria
    max_iters: int | None = None  # Optional iteration-based stopping (not set by default)
    max_time_hours: float = 5.0  # Time-based stopping (default 5 hours)
    target_ice: float | None = None  # Optional ICE-based stopping threshold

    # Sample weights
    initial_pseudo_pos_weight: float = 0.2
    initial_pseudo_neg_weight: float = 0.8
    weight_increment: float = 0.1

    # Rollback criteria (ICE-based: lower is better, so we detect increases)
    ice_increase_threshold: float = 0.05  # Trigger rollback if ICE increases by this much

    # Success targets
    target_recall: float = 0.75
    target_precision: float = 0.60
    target_enrichment: float = 50.0

    # Prevalence assumption (for enrichment calculation)
    assumed_prevalence: float = 0.001  # 0.1%

    # Prediction batch size for large datasets
    prediction_batch_size: int = 5_000_000

    # Quantized pool for faster predictions (CatBoost-specific optimization)
    use_quantized_pool: bool = False  # Set True to use quantized CatBoost pool

    # Enrichment calculation frequency (every N iterations, 0 to disable)
    enrichment_every_n_iters: int = 5

    # CatBoost parameters
    catboost_iterations: int = 1000
    catboost_depth: int = 8  # Increased from 6
    catboost_learning_rate: float = 0.1
    catboost_verbose: bool = False
    catboost_use_gpu: bool = True  # Enable GPU by default
    catboost_eval_fraction: float = 0.1  # Fraction for auto early stopping
    catboost_early_stopping_rounds: int = 50
    catboost_plot: bool = True  # Plot training progress
    catboost_loss_function: str = "Logloss"  # Calibrated metric (Logloss or CrossEntropy)
    catboost_eval_metric: str = "Logloss"  # Calibrated eval metric (was AUC)

    # Seed negative review (review seed negatives that may be mislabeled)
    review_seed_negatives: bool = True

    # Plotting configuration
    plot_samples: bool = True  # Plot pseudo-positives when added/removed
    plot_singlepoint_min_outlying_factor: float = 10.0  # For cleaned plots

    # Special row tracking
    track_rowid: int | None = 55554273  # Log probability for this rowid each iteration

    # Pseudo-label review thresholds
    pseudo_pos_removal_prob: float = 0.3  # Remove pseudo_pos if prob drops below this
    pseudo_neg_promotion_prob: float = 0.7  # Remove pseudo_neg if prob rises above this
    bootstrap_instability_std: float = 0.2  # Max std before considering unstable
    bootstrap_instability_mean: float = 0.7  # Min mean when std is high
    seed_neg_removal_prob: float = 0.9  # Remove seed neg if prob exceeds this
    seed_neg_removal_bootstrap_mean: float = 0.85  # Min bootstrap mean for seed neg removal

    # Curriculum learning phase thresholds
    curriculum_phase1_recall: float = 0.5  # Below this: strict phase
    curriculum_phase2_recall: float = 0.65  # Below this: medium phase
    curriculum_phase1_conf: float = 0.95  # Required confidence in phase 1
    curriculum_phase2_conf: float = 0.85  # Required confidence in phase 2
    curriculum_phase3_conf: float = 0.70  # Required confidence in phase 3
    curriculum_phase2_weight: float = 0.7  # Weight for medium-confidence samples in phase 2

    # Adaptive threshold adjustments
    threshold_relax_successful_iters: int = 3  # Iters before relaxing thresholds
    threshold_relax_pos_delta: float = 0.005  # How much to relax pos threshold
    threshold_relax_neg_delta: float = 0.01  # How much to relax neg threshold
    threshold_relax_max_pos_delta: int = 1  # How much to increase max_pseudo_pos
    threshold_tighten_pos_delta: float = 0.01  # How much to tighten pos threshold
    threshold_tighten_neg_delta: float = 0.02  # How much to tighten neg threshold
    threshold_tighten_max_pos_delta: int = 3  # How much to decrease max_pseudo_pos
    min_pseudo_pos_threshold: float = 0.95  # Minimum pos threshold after relaxation
    max_pseudo_neg_threshold: float = 0.10  # Maximum neg threshold after relaxation
    min_pseudo_pos_per_iter: int = 3  # Minimum pseudo_pos limit
    max_pseudo_pos_cap: int = 20  # Maximum pseudo_pos limit

    # Success criteria
    min_successful_iters_for_success: int = 5  # Required successful iters for SUCCESS stop
    oob_divergence_warning: float = 0.15  # Warn if OOB/held-out recall diverge by this

    # Bootstrap consensus for negatives
    neg_consensus_min_low_prob: float = 0.1  # Threshold for "low" in neg consensus

    # mlframe integration (optional)
    use_mlframe: bool = False  # Set True to use mlframe for training
    mlframe_models: list[str] = field(default_factory=lambda: ["cb"])  # ["cb", "lgb", "xgb"]


@dataclass(slots=True)
class LabeledSample:
    """A single labeled sample in the training set."""

    index: int  # Index in the source DataFrame (big_features or oos_features)
    label: int  # 0 or 1
    weight: float  # Sample weight for training
    source: Literal["seed", "val_pool", "pseudo_pos", "pseudo_neg"]
    added_iter: int  # Iteration when added
    confidence: float  # P(label) when added
    consensus_score: float  # Bootstrap consensus [0, 1]
    is_flare_source: bool = False  # True if from oos_features, False if from big_features


@dataclass(slots=True)
class Checkpoint:
    """Model checkpoint with associated metadata."""

    iteration: int
    model_path: Path
    labeled_train: list[LabeledSample]
    metrics: dict
    config_snapshot: dict


@dataclass(slots=True)
class HeldOutSet:
    """Held-out evaluation set."""

    flare_indices: np.ndarray  # Indices in oos_features
    negative_indices: np.ndarray  # Indices in big_features


@dataclass(slots=True)
class IterationMetrics:
    """Metrics collected at each iteration."""

    iteration: int

    # Sizes
    train_total: int
    train_seed: int
    train_val_pool: int
    train_pseudo_pos: int
    train_pseudo_neg: int
    val_pool_remaining: int

    # Effective sizes (with weights)
    effective_pos: float
    effective_neg: float

    # Quality metrics
    val_recall: float
    held_out_recall: float
    held_out_precision: float
    held_out_auc: float
    held_out_f1: float

    # Calibration metrics
    held_out_logloss: float = 0.0
    held_out_brier: float = 0.0
    held_out_ice: float = 0.0  # ICE metric (primary stopping criterion)
    held_out_pr_auc: float = 0.0  # PR-AUC (important for imbalanced data)

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
    val_pool_moved: int = 0

    # Stability
    n_successful_iters: int = 0
    n_rollbacks_recent: int = 0

    # Time tracking
    elapsed_hours: float = 0.0

    # Special row tracking
    tracked_rowid_prob: float | None = None

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
    task_type = "GPU" if config.catboost_use_gpu else "CPU"

    # Build model with calibration-focused loss and native eval_fraction
    model = CatBoostClassifier(
        iterations=config.catboost_iterations,
        depth=config.catboost_depth,
        learning_rate=config.catboost_learning_rate,
        random_seed=random_state,
        verbose=config.catboost_verbose,
        auto_class_weights=None,  # We handle weights manually
        loss_function=config.catboost_loss_function,  # Logloss for calibration
        eval_metric=config.catboost_eval_metric,  # Logloss instead of AUC
        task_type=task_type,
        early_stopping_rounds=config.catboost_early_stopping_rounds,
        eval_fraction=config.catboost_eval_fraction if config.catboost_eval_fraction > 0 else None,
    )

    model.fit(
        features,
        labels,
        sample_weight=sample_weights,
        # plot=config.catboost_plot,
        # plot_file=str(plot_file) if plot_file else None,
    )

    return model


def evaluate_model(
    model: CatBoostClassifier,
    features: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Evaluate model on a labeled set with calibration metrics.

    Parameters
    ----------
    model : CatBoostClassifier
        Trained classifier.
    features : np.ndarray
        Feature matrix.
    labels : np.ndarray
        True binary labels.
    threshold : float
        Classification threshold.

    Returns
    -------
    dict
        Dictionary with recall, precision, auc, f1, logloss, brier, ice metrics.
    """
    proba = model.predict_proba(features)[:, 1]
    preds = (proba >= threshold).astype(int)

    # Handle edge cases for metrics
    if len(np.unique(labels)) < 2:
        auc = 0.5  # Can't compute AUC with single class
        logloss_val = 1.0
    else:
        auc = roc_auc_score(labels, proba)
        logloss_val = log_loss(labels, proba)

    brier = brier_score_loss(labels, proba)

    # ICE = Integrated Calibration Error (simplified approximation)
    # Use binned calibration error
    ice = _compute_ice(labels, proba)

    return {
        "recall": recall_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "auc": auc,
        "f1": f1_score(labels, preds, zero_division=0),
        "logloss": logloss_val,
        "brier": brier,
        "ice": ice,
    }


def _compute_ice(labels: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Integrated Calibration Error (binned approximation).

    ICE measures how well predicted probabilities match observed frequencies.
    Lower is better (0 = perfectly calibrated).
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ice = 0.0
    total_count = 0

    for i in range(n_bins):
        mask = (proba >= bin_edges[i]) & (proba < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge in last bin
            mask = (proba >= bin_edges[i]) & (proba <= bin_edges[i + 1])

        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_mean_pred = np.mean(proba[mask])
            bin_mean_true = np.mean(labels[mask])
            ice += bin_count * abs(bin_mean_pred - bin_mean_true)
            total_count += bin_count

    return ice / total_count if total_count > 0 else 0.0


def predict_proba_batched(
    model: CatBoostClassifier,
    features: np.ndarray,
    batch_size: int = 5_000_000,
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
        Number of samples per batch.
    desc : str
        Description for progress bar.

    Returns
    -------
    np.ndarray
        Predicted probabilities for positive class.
    """
    n_samples = len(features)
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


def get_feature_columns(df: pl.DataFrame) -> list[str]:
    """Get feature column names (excluding id, class, metadata)."""
    exclude = {"id", "class", "ts", "index"}
    return [c for c in df.columns if c not in exclude]


def extract_features_array(
    df: pl.DataFrame,
    indices: np.ndarray | list[int],
    feature_cols: list[str],
) -> np.ndarray:
    """Extract feature matrix for given indices."""
    subset = df[indices].select(feature_cols)
    return subset.to_numpy().astype(np.float32)


def plot_sample(
    sample_index: int,
    source_df: pl.DataFrame,
    output_dir: Path,
    prefix: str,
    config: "PipelineConfig",
    action: str = "added",
) -> None:
    """
    Plot a sample in both raw and cleaned modes.

    Parameters
    ----------
    sample_index : int
        Index in the source DataFrame.
    source_df : pl.DataFrame
        Source DataFrame (big_features or oos_features).
    output_dir : Path
        Base output directory.
    prefix : str
        Prefix for filenames (e.g., "pseudo_pos", "pseudo_neg").
    config : PipelineConfig
        Pipeline configuration.
    action : str
        Action description ("added" or "removed").
    """
    if not VIEW_SERIES_AVAILABLE or not config.plot_samples:
        return

    try:
        # Get sample ID if available
        sample_id = source_df[sample_index, "id"] if "id" in source_df.columns else sample_index
        row_num = sample_index

        # Create output subdirectory
        plots_dir = output_dir / "sample_plots" / prefix
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Determine backend
        backend = "plotly" if is_jupyter_notebook() else "matplotlib"

        # Plot raw
        raw_file = plots_dir / f"{action}_{sample_id}_row{row_num}_raw.png"
        fig = view_series(
            source_df,
            sample_index,
            backend=backend,
            plot_file=str(raw_file) if backend == "matplotlib" else None,
        )
        # Display in Jupyter if using plotly
        if backend == "plotly" and fig is not None:
            fig.show()

        # Plot cleaned
        cleaned_file = plots_dir / f"{action}_{sample_id}_row{row_num}_cleaned.png"
        fig = view_series(
            source_df,
            sample_index,
            backend=backend,
            singlepoint_min_outlying_factor=config.plot_singlepoint_min_outlying_factor,
            plot_file=str(cleaned_file) if backend == "matplotlib" else None,
        )
        # Display in Jupyter if using plotly
        if backend == "plotly" and fig is not None:
            fig.show()

        logger.info(f"Plotted sample {sample_id} (row {row_num}) - {action}")

    except Exception as e:
        logger.warning(f"Failed to plot sample {sample_index}: {e}")


def report_held_out_metrics(
    targets: np.ndarray,
    probs: np.ndarray,
    iteration: int,
    output_dir: Path,
    config: "PipelineConfig",
) -> dict[str, float]:
    """
    Report held-out metrics using report_model_perf if available.

    Parameters
    ----------
    targets : np.ndarray
        True binary labels.
    probs : np.ndarray
        Predicted probabilities for positive class.
    iteration : int
        Current iteration number.
    output_dir : Path
        Output directory for charts.
    config : PipelineConfig
        Pipeline configuration.

    Returns
    -------
    dict[str, float]
        Dictionary of metrics including ICE.
    """
    metrics = {}

    if REPORT_PERF_AVAILABLE and report_model_perf is not None:
        try:
            charts_dir = output_dir / "perf_charts"
            charts_dir.mkdir(parents=True, exist_ok=True)

            report_params = {
                "report_ndigits": 2,
                "calib_report_ndigits": 2,
                "print_report": True,
                "report_title": f"Held-Out Iter {iteration}",
                "use_weights": True,
                "show_perf_chart": True,
            }

            plot_file = join(str(charts_dir), f"iter_{iteration:03d}")

            # Convert 1D probs (positive class) to 2D (n_samples, 2) for report_model_perf
            probs_2d = np.column_stack([1 - probs, probs])

            _, returned_metrics = report_model_perf(
                targets=targets.astype(np.int8),
                columns=None,
                df=None,
                model_name=f"iter_{iteration:03d}",
                model=None,
                target_label_encoder=None,
                preds=None,
                probs=probs_2d,
                plot_file=plot_file,
                metrics={},
                group_ids=None,
                **report_params,
            )

            if returned_metrics:
                metrics.update(returned_metrics)

        except Exception as e:
            logger.warning(f"report_model_perf failed: {e}")

    # Always compute ICE even if report_model_perf unavailable
    if "ice" not in metrics:
        metrics["ice"] = _compute_ice(targets, probs)

    return metrics


# =============================================================================
# Pipeline Implementation
# =============================================================================


class ActiveLearningPipeline:
    """
    Zero-Expert Self-Training Pipeline for Stellar Flare Detection.

    This pipeline implements iterative pseudo-labeling with bootstrap consensus
    for rare event detection in astronomical data. Starting from a small set of
    known flares (positive examples) and a large unlabeled dataset, it iteratively:

    1. Trains a classifier on current labeled data
    2. Predicts on the full unlabeled dataset
    3. Selects high-confidence predictions as pseudo-labels
    4. Validates consistency using bootstrap ensemble
    5. Retrains with expanded labeled set

    Key Design Principles
    ---------------------
    - **Asymmetric Pseudo-labeling**: Aggressive for negatives (abundant, low-risk),
      conservative for positives (rare, high-risk of confirmation bias)
    - **Bootstrap Consensus**: Multiple models must agree before accepting a
      pseudo-label, reducing variance and detecting instability
    - **Held-out as Arbiter**: A fixed held-out set (never touched during training)
      provides honest evaluation and rollback signal
    - **Adaptive Thresholds**: Thresholds tighten after failures, relax after
      successful iterations, allowing self-regulating exploration

    The approach is "zero-expert" because it does not require human labeling
    during the iterative process. However, it can be enhanced with expert
    labeling at strategic points (e.g., uncertain examples near decision boundary).

    References
    ----------
    - Settles, B. (2012). Active Learning. Morgan & Claypool Publishers.
    - Xie et al. (2020). Self-Training with Noisy Student. CVPR.
    - Arazo et al. (2020). Pseudo-Labeling and Confirmation Bias. IJCNN.

    See Also
    --------
    run_active_learning_pipeline : Convenience function for running the pipeline.
    """

    def __init__(
        self,
        big_features: pl.DataFrame,
        oos_features: pl.DataFrame,
        config: PipelineConfig | None = None,
        output_dir: Path | str = "active_learning_output",
        random_state: int = 42,
    ):
        """
        Initialize the pipeline.

        Parameters
        ----------
        big_features : pl.DataFrame
            Large unlabeled dataset (~94M observations).
            Expected columns: feature columns, optionally 'id'.
        oos_features : pl.DataFrame
            Known flares dataset (99 samples with class=1).
            Expected columns: feature columns, 'class', optionally 'id'.
        config : PipelineConfig, optional
            Pipeline configuration. Uses defaults if not provided.
        output_dir : Path or str
            Directory for saving checkpoints and results.
        random_state : int
            Random seed for reproducibility.
        """
        self.big_features = big_features
        self.oos_features = oos_features
        self.config = config or PipelineConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        # Validate input
        self._validate_inputs()

        # Get feature columns (same for both datasets)
        self.feature_cols = get_feature_columns(self.oos_features)
        logger.info(f"Using {len(self.feature_cols)} feature columns")

        # State variables
        self.labeled_train: list[LabeledSample] = []
        self.validation_pool: list[int] = []  # Indices in oos_features
        self.held_out: HeldOutSet | None = None
        self.model: CatBoostClassifier | None = None
        self.bootstrap_models: list[CatBoostClassifier] = []
        self.bootstrap_indices_list: list[np.ndarray] = []  # For OOB computation

        # Metrics tracking
        self.metrics_history: list[IterationMetrics] = []
        self.best_checkpoint: Checkpoint | None = None
        self.best_held_out_recall: float = 0.0
        self.best_held_out_ice: float = 1.0  # Lower is better
        self.prev_held_out_recall: float = 0.0
        self.prev_held_out_precision: float = 0.0
        self.prev_held_out_ice: float = 1.0  # ICE starts high (lower is better)
        self.prev_val_recall: float = 0.0
        self.n_successful_iters: int = 0
        self.rollback_history: list[int] = []  # Iteration numbers of rollbacks

        # Current thresholds (mutable)
        self.pseudo_pos_threshold = self.config.pseudo_pos_threshold
        self.pseudo_neg_threshold = self.config.pseudo_neg_threshold
        self.max_pseudo_pos_per_iter = self.config.max_pseudo_pos_per_iter

        # Time tracking
        self.start_time: float | None = None

        # Cached feature views for efficiency (zero-copy pandas view of polars DataFrame)
        self._big_features_view: np.ndarray | None = None
        # Cached quantized CatBoost pool for faster predictions
        self._quantized_pool: Pool | None = None

        # Track rowid index mapping
        self._tracked_rowid_index: int | None = None
        if self.config.track_rowid is not None and "id" in self.big_features.columns:
            # Find the index of the tracked rowid
            mask = self.big_features["id"] == self.config.track_rowid
            indices = np.where(mask.to_numpy())[0]
            if len(indices) > 0:
                self._tracked_rowid_index = int(indices[0])
                logger.info(f"Tracking rowid {self.config.track_rowid} at index {self._tracked_rowid_index}")

    def _validate_inputs(self) -> None:
        """Validate input DataFrames."""
        n_flares = len(self.oos_features)
        required_flares = self.config.n_train_flares_init + self.config.n_val_pool + self.config.n_held_out_flares
        if n_flares < required_flares:
            raise ValueError(f"Need at least {required_flares} known flares, got {n_flares}")

        if len(self.big_features) < self.config.n_train_neg_init + self.config.n_held_out_neg:
            raise ValueError(f"big_features too small for required negative samples")

        logger.info(f"big_features: {len(self.big_features):,} samples")
        logger.info(f"oos_features: {len(self.oos_features)} known flares")

    def _get_labeled_indices(self) -> tuple[set[int], set[int]]:
        """Get sets of already labeled indices (big_features, oos_features)."""
        big_indices = set()
        oos_indices = set()
        for sample in self.labeled_train:
            if sample.is_flare_source:
                oos_indices.add(sample.index)
            else:
                big_indices.add(sample.index)
        return big_indices, oos_indices

    def _build_training_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build feature matrix, labels, and weights from labeled_train (batched by source)."""
        # Separate by source for batched extraction
        oos_items = [(i, s) for i, s in enumerate(self.labeled_train) if s.is_flare_source]
        big_items = [(i, s) for i, s in enumerate(self.labeled_train) if not s.is_flare_source]

        n_samples = len(self.labeled_train)
        features = np.zeros((n_samples, len(self.feature_cols)), dtype=np.float32)
        labels = np.zeros(n_samples, dtype=np.int32)
        weights = np.zeros(n_samples, dtype=np.float32)

        # Batch extract from oos_features
        if oos_items:
            oos_indices = [s.index for _, s in oos_items]
            oos_features = extract_features_array(self.oos_features, oos_indices, self.feature_cols)
            for j, (i, sample) in enumerate(oos_items):
                features[i] = oos_features[j]
                labels[i] = sample.label
                weights[i] = sample.weight

        # Batch extract from big_features
        if big_items:
            big_indices = [s.index for _, s in big_items]
            big_features_batch = extract_features_array(self.big_features, big_indices, self.feature_cols)
            for j, (i, sample) in enumerate(big_items):
                features[i] = big_features_batch[j]
                labels[i] = sample.label
                weights[i] = sample.weight

        return features, labels, weights

    def _compute_val_metrics(self) -> float:
        """Compute recall on validation pool (all are flares)."""
        if not self.validation_pool or self.model is None:
            return 0.0

        features = extract_features_array(self.oos_features, self.validation_pool, self.feature_cols)
        proba = self.model.predict_proba(features)[:, 1]
        preds = (proba >= 0.5).astype(int)
        # All validation pool samples are flares (label=1)
        labels = np.ones(len(self.validation_pool), dtype=np.int32)
        return recall_score(labels, preds, zero_division=0)

    def _compute_held_out_metrics(self, iteration: int = 0) -> dict[str, float]:
        """
        Compute metrics on held-out set including calibration metrics.

        Uses report_model_perf if available for comprehensive metrics including ICE.
        """
        if self.held_out is None or self.model is None:
            return {"recall": 0.0, "precision": 0.0, "auc": 0.5, "f1": 0.0, "logloss": 1.0, "brier": 0.25, "ice": 0.5}

        # Flares from oos_features
        flare_features = extract_features_array(self.oos_features, self.held_out.flare_indices, self.feature_cols)
        flare_proba = self.model.predict_proba(flare_features)[:, 1]
        flare_preds = (flare_proba >= 0.5).astype(int)
        flare_labels = np.ones(len(self.held_out.flare_indices), dtype=np.int32)

        # Negatives from big_features
        neg_features = extract_features_array(self.big_features, self.held_out.negative_indices, self.feature_cols)
        neg_proba = self.model.predict_proba(neg_features)[:, 1]
        neg_preds = (neg_proba >= 0.5).astype(int)
        neg_labels = np.zeros(len(self.held_out.negative_indices), dtype=np.int32)

        # Combined
        all_proba = np.concatenate([flare_proba, neg_proba])
        all_preds = np.concatenate([flare_preds, neg_preds])
        all_labels = np.concatenate([flare_labels, neg_labels])

        # Basic metrics
        metrics = {
            "recall": recall_score(all_labels, all_preds, zero_division=0),
            "precision": precision_score(all_labels, all_preds, zero_division=0),
            "auc": roc_auc_score(all_labels, all_proba) if len(np.unique(all_labels)) > 1 else 0.5,
            "f1": f1_score(all_labels, all_preds, zero_division=0),
        }

        # Calibration metrics
        if len(np.unique(all_labels)) > 1:
            metrics["logloss"] = log_loss(all_labels, all_proba)
        else:
            metrics["logloss"] = 1.0

        metrics["brier"] = brier_score_loss(all_labels, all_proba)
        metrics["ice"] = _compute_ice(all_labels, all_proba)

        # PR-AUC (Average Precision) - important for imbalanced data
        if len(np.unique(all_labels)) > 1:
            metrics["pr_auc"] = average_precision_score(all_labels, all_proba)
        else:
            metrics["pr_auc"] = 0.0

        # Use report_model_perf for additional metrics if available
        additional = report_held_out_metrics(all_labels, all_proba, iteration, self.output_dir, self.config)
        for key, value in additional.items():
            if key not in metrics:
                metrics[key] = value

        return metrics

    def _save_checkpoint(self, iteration: int, metrics: dict) -> Checkpoint:
        """Save model checkpoint."""
        model_path = self.output_dir / f"model_iter_{iteration:03d}.joblib"
        joblib.dump(self.model, model_path)

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
                },
                f,
                indent=2,
            )

        return checkpoint

    def _load_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Load model from checkpoint."""
        self.model = joblib.load(checkpoint.model_path)
        self.labeled_train = checkpoint.labeled_train.copy()
        self.pseudo_pos_threshold = checkpoint.config_snapshot["pseudo_pos_threshold"]
        self.pseudo_neg_threshold = checkpoint.config_snapshot["pseudo_neg_threshold"]
        self.max_pseudo_pos_per_iter = checkpoint.config_snapshot["max_pseudo_pos_per_iter"]

    def _count_by_source(self) -> dict[str, int]:
        """Count samples by source."""
        counts = {"seed": 0, "val_pool": 0, "pseudo_pos": 0, "pseudo_neg": 0}
        for sample in self.labeled_train:
            counts[sample.source] += 1
        return counts

    def _compute_effective_sizes(self) -> tuple[float, float]:
        """Compute effective class sizes (sum of weights)."""
        pos_weight = sum(s.weight for s in self.labeled_train if s.label == 1)
        neg_weight = sum(s.weight for s in self.labeled_train if s.label == 0)
        return pos_weight, neg_weight

    def _get_elapsed_hours(self) -> float:
        """Get elapsed time since pipeline start in hours."""
        if self.start_time is None:
            return 0.0
        return (time.time() - self.start_time) / 3600.0

    def _get_big_features_for_prediction(self):
        """
        Get feature view for big_features using zero-copy pandas view.

        Uses get_pandas_view_of_polars_df from mlframe to create a zero-copy
        pandas view of the polars DataFrame. This avoids duplicating the ~20GB
        feature matrix in memory.

        The view is cached after first creation. Do NOT call .values on the
        result - that triggers pandas block consolidation and copies memory!

        Returns
        -------
        pd.DataFrame or None
            Feature DataFrame (zero-copy view), or None if unavailable.
        """
        if self._big_features_view is not None:
            return self._big_features_view

        # Report RAM before
        ram_before = psutil.Process().memory_info().rss / 1024**3
        logger.info(f"Creating zero-copy view of big_features for prediction... (RAM: {ram_before:.2f} GB)")

        if MLFRAME_AVAILABLE and get_pandas_view_of_polars_df is not None:
            # Use zero-copy pandas view - do NOT call .values (causes copy!)
            self._big_features_view = get_pandas_view_of_polars_df(self.big_features.select(self.feature_cols))
            # Report RAM after
            ram_after = psutil.Process().memory_info().rss / 1024**3
            logger.info(
                f"Zero-copy view created: shape={self._big_features_view.shape}, "
                f"RAM: {ram_before:.2f} -> {ram_after:.2f} GB (delta: {ram_after - ram_before:+.2f} GB)"
            )
        else:
            logger.warning("get_pandas_view_of_polars_df not available, falling back to batched prediction")
            return None

        return self._big_features_view

    def _get_quantized_pool(self) -> Pool | None:
        """
        Get or create a quantized CatBoost Pool for big_features.

        Quantized pools pre-compute feature binning once, making subsequent
        predictions significantly faster. The pool is cached after first creation.

        Returns
        -------
        Pool or None
            Quantized CatBoost Pool, or None if creation fails.
        """
        if self._quantized_pool is not None:
            return self._quantized_pool

        if not self.config.use_quantized_pool:
            return None

        # Report RAM before
        ram_before = psutil.Process().memory_info().rss / 1024**3
        logger.info(f"Creating quantized CatBoost pool... (RAM: {ram_before:.2f} GB)")

        try:
            # Get the pandas view first (zero-copy)
            features_view = self._get_big_features_for_prediction()
            if features_view is None:
                logger.warning("Cannot create quantized pool: no features view available")
                return None

            # Create the pool
            logger.info("Creating Pool object...")
            self._quantized_pool = Pool(data=features_view)

            # Actually quantize it - this caches feature binning
            logger.info("Quantizing pool (this caches feature binning)...")
            self._quantized_pool.quantize()

            # Verify it's quantized
            if not self._quantized_pool.is_quantized():
                logger.warning("Pool.quantize() called but pool reports is_quantized=False!")

            # Report RAM after
            ram_after = psutil.Process().memory_info().rss / 1024**3
            logger.info(
                f"Quantized pool created: {len(features_view):,} samples, "
                f"is_quantized={self._quantized_pool.is_quantized()}, "
                f"RAM: {ram_before:.2f} -> {ram_after:.2f} GB (delta: {ram_after - ram_before:+.2f} GB)"
            )

            return self._quantized_pool

        except Exception as e:
            logger.warning(f"Failed to create quantized pool: {e}")
            return None

    def _predict_big_features_batched(self, model: CatBoostClassifier | None = None) -> np.ndarray:
        """
        Predict on big_features using quantized pool, cached view, or batched extraction.

        Priority order:
        1. Quantized CatBoost pool (if use_quantized_pool=True) - fastest
        2. Zero-copy pandas view with batched prediction
        3. Batch-by-batch extraction (fallback)

        Parameters
        ----------
        model : CatBoostClassifier, optional
            Model to use. Defaults to self.model.

        Returns
        -------
        np.ndarray
            Predictions (probabilities) for all samples in big_features.
        """
        if model is None:
            model = self.model
        if model is None:
            raise ValueError("No model available for prediction")

        n_total = len(self.big_features)

        # Option 1: Try quantized pool first (fastest)
        if self.config.use_quantized_pool:
            pool = self._get_quantized_pool()
            if pool is not None:
                logger.info(f"Predicting on {n_total:,} samples using quantized pool...")
                # predict_proba returns shape (n_samples, n_classes)
                return model.predict_proba(pool)[:, 1].astype(np.float32)

        # Option 2: Try cached zero-copy view
        features_view = self._get_big_features_for_prediction()
        if features_view is not None:
            logger.info(f"Predicting on {n_total:,} samples using cached view...")
            return predict_proba_batched(
                model,
                features_view,
                batch_size=self.config.prediction_batch_size,
                desc="Prediction",
            )

        # Fallback: batch-by-batch extraction (for when zero-copy view unavailable)
        logger.info(f"Predicting on {n_total:,} samples using batched extraction...")
        batch_size = self.config.prediction_batch_size
        all_preds = np.zeros(n_total, dtype=np.float32)

        for start in tqdmu(range(0, n_total, batch_size), desc="Batched prediction"):
            end = min(start + batch_size, n_total)
            batch_indices = np.arange(start, end)

            # Extract features for this batch only
            batch_features = extract_features_array(self.big_features, batch_indices, self.feature_cols)

            # Predict
            batch_preds = model.predict_proba(batch_features)[:, 1]
            all_preds[start:end] = batch_preds

            # Free memory
            del batch_features

        clean_ram()
        return all_preds

    def _get_tracked_rowid_probability(self) -> float | None:
        """Get predicted probability for the tracked rowid."""
        if self._tracked_rowid_index is None or self.model is None:
            return None

        try:
            features = extract_features_array(self.big_features, [self._tracked_rowid_index], self.feature_cols)
            prob = self.model.predict_proba(features)[0, 1]
            logger.info(f"Tracked rowid {self.config.track_rowid} (idx {self._tracked_rowid_index}): " f"P(flare)={prob:.6f}")
            return float(prob)
        except Exception as e:
            logger.warning(f"Failed to get probability for tracked rowid: {e}")
            return None

    # =========================================================================
    # Phase 0: Initialization
    # =========================================================================

    def initialize(self) -> None:
        """
        Phase 0: Initialize the pipeline.

        1. Stratified split of known flares into train/val/held-out
        2. Sample negatives from big_features
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

        # Stratified split of known flares
        n_flares = len(self.oos_features)
        train_ratio = self.config.n_train_flares_init / n_flares
        val_ratio = self.config.n_val_pool / n_flares
        held_out_ratio = self.config.n_held_out_flares / n_flares

        train_flare_indices, val_indices, held_out_flare_indices = stratified_flare_split(
            self.oos_features,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            held_out_ratio=held_out_ratio,
            random_state=self.random_state,
        )

        self.validation_pool = val_indices
        held_out_flare_indices = np.array(held_out_flare_indices)

        logger.info(
            f"Flares split (stratified): train={len(train_flare_indices)}, " f"val_pool={len(self.validation_pool)}, held_out={len(held_out_flare_indices)}"
        )

        # Sample negatives from big_features
        n_big = len(self.big_features)
        all_neg_indices = self.rng.permutation(n_big)

        train_neg_indices = all_neg_indices[: self.config.n_train_neg_init].tolist()
        held_out_neg_indices = all_neg_indices[self.config.n_train_neg_init : self.config.n_train_neg_init + self.config.n_held_out_neg]

        logger.info(f"Negatives sampled: train={len(train_neg_indices)}, " f"held_out={len(held_out_neg_indices)}")

        # Create held-out set
        self.held_out = HeldOutSet(
            flare_indices=held_out_flare_indices,
            negative_indices=held_out_neg_indices,
        )

        # Add flares to labeled_train
        for idx in train_flare_indices:
            self.labeled_train.append(
                LabeledSample(
                    index=idx,
                    label=1,
                    weight=1.0,
                    source="seed",
                    added_iter=0,
                    confidence=1.0,
                    consensus_score=1.0,
                    is_flare_source=True,
                )
            )

        # Add negatives to labeled_train
        for idx in train_neg_indices:
            self.labeled_train.append(
                LabeledSample(
                    index=idx,
                    label=0,
                    weight=1.0,
                    source="seed",
                    added_iter=0,
                    confidence=1.0,
                    consensus_score=1.0,
                    is_flare_source=False,
                )
            )

        logger.info(f"Initial labeled_train size: {len(self.labeled_train)}")

        # Train initial model
        logger.info("Training initial model...")
        features, labels, weights = self._build_training_data()
        self.model = train_model(features, labels, weights, config=self.config, random_state=self.random_state)

        # Compute baseline metrics (with iteration=0 to generate reports)
        val_recall = self._compute_val_metrics()
        held_out_metrics = self._compute_held_out_metrics(iteration=0)

        self.prev_val_recall = val_recall
        self.prev_held_out_recall = held_out_metrics["recall"]
        self.prev_held_out_precision = held_out_metrics["precision"]
        self.prev_held_out_ice = held_out_metrics.get("ice", 1.0)
        self.best_held_out_recall = held_out_metrics["recall"]
        self.best_held_out_ice = held_out_metrics.get("ice", 1.0)

        # Save initial checkpoint as best
        self.best_checkpoint = self._save_checkpoint(0, held_out_metrics)

        # Log initial metrics
        counts = self._count_by_source()
        eff_pos, eff_neg = self._compute_effective_sizes()

        initial_metrics = IterationMetrics(
            iteration=0,
            train_total=len(self.labeled_train),
            train_seed=counts["seed"],
            train_val_pool=counts["val_pool"],
            train_pseudo_pos=counts["pseudo_pos"],
            train_pseudo_neg=counts["pseudo_neg"],
            val_pool_remaining=len(self.validation_pool),
            effective_pos=eff_pos,
            effective_neg=eff_neg,
            val_recall=val_recall,
            held_out_recall=held_out_metrics["recall"],
            held_out_precision=held_out_metrics["precision"],
            held_out_auc=held_out_metrics["auc"],
            held_out_f1=held_out_metrics["f1"],
            held_out_logloss=held_out_metrics.get("logloss", 0.0),
            held_out_brier=held_out_metrics.get("brier", 0.0),
            held_out_ice=held_out_metrics.get("ice", 0.0),
            held_out_pr_auc=held_out_metrics.get("pr_auc", 0.0),
            enrichment_factor=0.0,
            estimated_flares_top10k=0.0,
            pseudo_pos_threshold=self.pseudo_pos_threshold,
            pseudo_neg_threshold=self.pseudo_neg_threshold,
            pseudo_pos_added=0,
            pseudo_neg_added=0,
            pseudo_removed=0,
            val_pool_moved=0,
            n_successful_iters=0,
            n_rollbacks_recent=0,
            elapsed_hours=self._get_elapsed_hours(),
            tracked_rowid_prob=self._get_tracked_rowid_probability(),
        )
        self.metrics_history.append(initial_metrics)

        logger.info("=" * 60)
        logger.info("ITERATION 0 COMPLETE")
        logger.info(f"  Train: {len(self.labeled_train)} samples")
        logger.info(f"  Val recall: {val_recall:.3f}")
        logger.info(
            f"  Held-out: recall={held_out_metrics['recall']:.3f}, "
            f"precision={held_out_metrics['precision']:.3f}, "
            f"ROC-AUC={held_out_metrics.get('auc', 0):.3f}, "
            f"ICE={held_out_metrics.get('ice', 0):.4f}"
        )
        logger.info(
            f"  PR-AUC={held_out_metrics.get('pr_auc', 0):.3f}, "
            f"Logloss={held_out_metrics.get('logloss', 0):.4f}, "
            f"Brier={held_out_metrics.get('brier', 0):.4f}"
        )
        logger.info("=" * 60)

    # =========================================================================
    # Main Loop Phases
    # =========================================================================

    def _phase1_validation_and_early_stopping(self, iteration: int) -> tuple[float, dict, bool, str | None]:
        """
        Phase 1: Validation and early stopping check.

        Returns:
            (val_recall, held_out_metrics, should_continue, stop_reason)
        """
        val_recall = self._compute_val_metrics()
        held_out_metrics = self._compute_held_out_metrics(iteration=iteration)

        current_recall = held_out_metrics["recall"]
        current_precision = held_out_metrics["precision"]
        current_ice = held_out_metrics.get("ice", 1.0)

        # Check for degradation using ICE (lower is better, so detect increases)
        ice_increased = current_ice > self.prev_held_out_ice + self.config.ice_increase_threshold

        if ice_increased:
            logger.warning("DEGRADATION DETECTED (ICE increased)!")
            logger.warning(f"  ICE: {self.prev_held_out_ice:.4f} -> {current_ice:.4f}")
            logger.warning(f"  (Recall: {self.prev_held_out_recall:.3f} -> {current_recall:.3f})")
            logger.warning(f"  (Precision: {self.prev_held_out_precision:.3f} -> {current_precision:.3f})")

            # Rollback
            if self.best_checkpoint is not None:
                logger.info(f"Rolling back to iteration {self.best_checkpoint.iteration}")
                self._load_checkpoint(self.best_checkpoint)

            # Tighten thresholds
            self.pseudo_pos_threshold = min(0.999, self.pseudo_pos_threshold + 0.005)
            self.pseudo_neg_threshold = max(0.01, self.pseudo_neg_threshold - 0.01)
            self.max_pseudo_pos_per_iter = max(3, self.max_pseudo_pos_per_iter - 2)

            logger.info(f"  New thresholds: pos>{self.pseudo_pos_threshold:.3f}, neg<{self.pseudo_neg_threshold:.3f}")

            self.n_successful_iters = 0
            self.rollback_history.append(iteration)

            # Recompute metrics after rollback
            val_recall = self._compute_val_metrics()
            held_out_metrics = self._compute_held_out_metrics()

            return val_recall, held_out_metrics, True, None  # Continue but skip this iteration

        # No degradation
        self.n_successful_iters += 1

        # Track best model by ICE (lower is better)
        if current_ice < self.best_held_out_ice:
            self.best_held_out_ice = current_ice
            self.best_held_out_recall = current_recall
            self.best_checkpoint = self._save_checkpoint(iteration, held_out_metrics)
            logger.info(f"New best model saved (ICE={current_ice:.4f}, recall={current_recall:.3f})")

        # Check for success
        enrichment, _ = self._compute_enrichment_factor() if iteration > 0 else (0.0, 0.0)
        if (
            current_recall >= self.config.target_recall
            and current_precision >= self.config.target_precision
            and enrichment >= self.config.target_enrichment
            and self.n_successful_iters >= self.config.min_successful_iters_for_success
        ):
            return val_recall, held_out_metrics, False, "SUCCESS"

        return val_recall, held_out_metrics, True, None

    def _phase2_hard_example_mining(self, val_recall: float) -> int:
        """
        Phase 2: Hard example mining from validation pool.

        Returns number of samples moved to training.
        """
        if val_recall <= self.prev_val_recall:
            return 0
        if len(self.validation_pool) <= 5:
            return 0
        if self.model is None:
            return 0

        # Get predictions on validation pool
        features = extract_features_array(self.oos_features, self.validation_pool, self.feature_cols)
        proba = self.model.predict_proba(features)[:, 1]

        # Find detected flares (P > 0.5)
        detected_mask = proba > 0.5
        if not np.any(detected_mask):
            return 0

        # Find hardest example (lowest P among detected)
        detected_indices = np.where(detected_mask)[0]
        detected_proba = proba[detected_mask]
        hardest_local_idx = detected_indices[np.argmin(detected_proba)]
        hardest_prob = proba[hardest_local_idx]

        # Move to training
        hardest_oos_idx = self.validation_pool[hardest_local_idx]
        self.validation_pool.pop(hardest_local_idx)

        self.labeled_train.append(
            LabeledSample(
                index=hardest_oos_idx,
                label=1,
                weight=0.5,  # Lower weight - not verified
                source="val_pool",
                added_iter=len(self.metrics_history),  # Current iteration
                confidence=hardest_prob,
                consensus_score=0.0,  # Not checked by bootstrap
                is_flare_source=True,
            )
        )

        logger.info(f"Hard example from val_pool: P={hardest_prob:.3f}, " f"val_pool remaining: {len(self.validation_pool)}")
        return 1

    def _phase3_train_bootstrap_models(self, iteration: int) -> tuple[list[CatBoostClassifier], list[np.ndarray]]:
        """
        Phase 3: Train bootstrap models for consensus estimation.

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

        for seed in tqdmu(range(self.config.n_bootstrap_models), desc="Bootstrap models"):
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
            )
            bootstrap_models.append(model)
            bootstrap_indices_list.append(bootstrap_indices)

        self.bootstrap_models = bootstrap_models
        self.bootstrap_indices_list = bootstrap_indices_list
        return bootstrap_models, bootstrap_indices_list

    def _phase4_predict_all(self) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        """
        Phase 4: Prediction on all big_features.

        Uses batched prediction to avoid memory issues. Predictions are made on ALL
        samples - held_out exclusion happens at the pseudo-label selection stage.

        Returns
        -------
        tuple[np.ndarray, list[np.ndarray], np.ndarray]
            (main_predictions, bootstrap_predictions, all_indices)
            all_indices is just np.arange(n_big) for full dataset
        """
        n_big = len(self.big_features)
        all_indices = np.arange(n_big)

        logger.info(f"Predicting on {n_big:,} samples (batched to avoid OOM)...")

        # Main model predictions
        main_preds = self._predict_big_features_batched(self.model)

        # Bootstrap model predictions
        bootstrap_preds = []
        for i, bm in enumerate(self.bootstrap_models):
            logger.info(f"Bootstrap model {i+1}/{len(self.bootstrap_models)}...")
            bp = self._predict_big_features_batched(bm)
            bootstrap_preds.append(bp)

        clean_ram()
        return main_preds, bootstrap_preds, all_indices

    def _phase5_pseudo_label_negatives(
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
            Maps position in predictions to index in big_features
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
        labeled_big_indices, _ = self._get_labeled_indices()

        # Build exclusion set: already labeled + held_out negatives
        held_out_set = set(self.held_out.negative_indices.tolist()) if self.held_out else set()
        exclusion_set = labeled_big_indices | held_out_set

        # Find candidates with low probability
        low_prob_positions = np.where(main_preds < self.pseudo_neg_threshold)[0]

        # Map to big_features indices and filter out exclusions
        neg_candidates = [(int(prediction_indices[pos]), pos) for pos in low_prob_positions if int(prediction_indices[pos]) not in exclusion_set]

        if not neg_candidates:
            return 0

        # Random subsample for efficiency
        if len(neg_candidates) > self.config.max_pseudo_neg_per_iter * 2:
            selected_indices = self.rng.choice(
                len(neg_candidates),
                size=self.config.max_pseudo_neg_per_iter * 2,
                replace=False,
            )
            neg_candidates = [neg_candidates[i] for i in selected_indices]

        # Filter by bootstrap consensus (majority)
        confirmed = []
        for big_idx, pos in neg_candidates:
            main_prob = main_preds[pos]
            bootstrap_probs = [bp[pos] for bp in bootstrap_preds]

            # Majority of bootstrap models agree it's low
            n_low = sum(p < self.config.neg_consensus_min_low_prob for p in bootstrap_probs)
            n_required = max(2, self.config.n_bootstrap_models // 2)  # Majority
            if n_low >= n_required:
                avg_prob = (main_prob + np.mean(bootstrap_probs)) / 2
                consensus = 1 - np.std(bootstrap_probs)
                confirmed.append(
                    {
                        "index": big_idx,
                        "confidence": 1 - avg_prob,
                        "consensus_score": consensus,
                    }
                )

        # Sort by confidence and take top
        confirmed.sort(key=lambda x: x["confidence"], reverse=True)
        confirmed = confirmed[: self.config.max_pseudo_neg_per_iter]

        # Compute weight based on successful iterations
        weight = min(
            1.0,
            self.config.initial_pseudo_neg_weight + self.n_successful_iters * self.config.weight_increment,
        )

        # Add to training
        for item in confirmed:
            self.labeled_train.append(
                LabeledSample(
                    index=item["index"],
                    label=0,
                    weight=weight,
                    source="pseudo_neg",
                    added_iter=iteration,
                    confidence=item["confidence"],
                    consensus_score=item["consensus_score"],
                    is_flare_source=False,
                )
            )

        logger.info(f"Pseudo-negatives added: {len(confirmed)}, weight={weight:.2f}")
        return len(confirmed)

    def _phase6_pseudo_label_positives(
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
            Maps position in predictions to index in big_features
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
        labeled_big_indices, _ = self._get_labeled_indices()

        # Build exclusion set: already labeled + held_out negatives
        held_out_set = set(self.held_out.negative_indices.tolist()) if self.held_out else set()
        exclusion_set = labeled_big_indices | held_out_set

        # Find candidates with high probability
        high_prob_positions = np.where(main_preds > self.pseudo_pos_threshold)[0]

        # Map to (big_idx, position) tuples and filter out exclusions
        pos_candidates = [(int(prediction_indices[pos]), pos) for pos in high_prob_positions if int(prediction_indices[pos]) not in exclusion_set]

        if not pos_candidates:
            return 0

        # Take top-K by probability
        pos_candidates = sorted(
            pos_candidates,
            key=lambda x: main_preds[x[1]],  # Sort by prob at position
            reverse=True,
        )[: self.max_pseudo_pos_per_iter * 5]

        # Strict filtering: ALL bootstrap models must agree
        confirmed = []
        for big_idx, pos in pos_candidates:
            main_prob = main_preds[pos]
            bootstrap_probs = [bp[pos] for bp in bootstrap_preds]

            # All bootstrap models must be confident
            all_high = all(p > self.config.consensus_threshold for p in bootstrap_probs)
            low_variance = np.std(bootstrap_probs) < 0.05

            if all_high and low_variance:
                avg_prob = (main_prob + np.mean(bootstrap_probs)) / 2
                consensus = 1 - np.std(bootstrap_probs)
                confirmed.append(
                    {
                        "index": big_idx,
                        "confidence": avg_prob,
                        "consensus_score": consensus,
                    }
                )

        # Sort by consensus and confidence
        confirmed.sort(key=lambda x: (x["consensus_score"], x["confidence"]), reverse=True)
        confirmed = confirmed[: self.max_pseudo_pos_per_iter]

        # Compute weight (lower than negatives)
        weight = min(
            0.8,
            self.config.initial_pseudo_pos_weight + self.n_successful_iters * self.config.weight_increment * 0.5,
        )

        # Add to training and plot
        for item in confirmed:
            sample_idx = item["index"]

            self.labeled_train.append(
                LabeledSample(
                    index=sample_idx,
                    label=1,
                    weight=weight,
                    source="pseudo_pos",
                    added_iter=iteration,
                    confidence=item["confidence"],
                    consensus_score=item["consensus_score"],
                    is_flare_source=False,
                )
            )

            # Log sample ID and row number
            sample_id = self.big_features[sample_idx, "id"] if "id" in self.big_features.columns else sample_idx
            logger.info(f"  Added pseudo_pos: id={sample_id}, row={sample_idx}, conf={item['confidence']:.4f}")

            # Plot the sample
            plot_sample(sample_idx, self.big_features, self.output_dir, f"pseudo_pos_iter{iteration:03d}", self.config, action="added")

        logger.info(f"Pseudo-positives added: {len(confirmed)}, weight={weight:.2f}")
        return len(confirmed)

    def _phase7_review_pseudo_labels(self) -> int:
        """
        Phase 7: Review and remove inconsistent pseudo-labels.

        Uses batching by source for efficiency - avoids calling model.predict
        for each sample individually.

        Now also reviews seed negatives if config.review_seed_negatives is True.

        Returns number of samples removed.
        """
        # Group samples by source for batched prediction
        oos_samples = []  # (list_index, sample) for samples from oos_features
        big_samples = []  # (list_index, sample) for samples from big_features

        for i, sample in enumerate(self.labeled_train):
            # Skip seed positives (known flares - ground truth)
            if sample.source == "seed" and sample.label == 1:
                continue

            # Handle seed negatives based on config
            if sample.source == "seed" and sample.label == 0:
                if not self.config.review_seed_negatives:
                    continue

            if sample.is_flare_source:
                oos_samples.append((i, sample))
            else:
                big_samples.append((i, sample))

        to_remove = []
        to_plot_removed = []  # (index, source_df, sample) for removed samples

        # Batch predict for oos_features samples
        if oos_samples:
            oos_indices = [s.index for _, s in oos_samples]
            oos_features = extract_features_array(self.oos_features, oos_indices, self.feature_cols)

            main_probs = self.model.predict_proba(oos_features)[:, 1]
            bootstrap_probs_all = np.array([bm.predict_proba(oos_features)[:, 1] for bm in self.bootstrap_models])  # Shape: (n_bootstrap, n_samples)

            for j, (i, sample) in enumerate(oos_samples):
                current_prob = main_probs[j]
                bootstrap_probs = bootstrap_probs_all[:, j]

                should_remove, reason = self._check_sample_for_removal(sample, current_prob, bootstrap_probs)
                if should_remove:
                    to_remove.append(i)
                    to_plot_removed.append((sample.index, self.oos_features, sample))
                    logger.debug(f"Removing {sample.source} from oos: {reason}")
                else:
                    # Update weight based on current confidence
                    self._update_sample_weight(sample, current_prob, bootstrap_probs)

        # Batch predict for big_features samples
        if big_samples:
            big_indices = [s.index for _, s in big_samples]
            big_features_batch = extract_features_array(self.big_features, big_indices, self.feature_cols)

            main_probs = self.model.predict_proba(big_features_batch)[:, 1]
            bootstrap_probs_all = np.array([bm.predict_proba(big_features_batch)[:, 1] for bm in self.bootstrap_models])

            for j, (i, sample) in enumerate(big_samples):
                current_prob = main_probs[j]
                bootstrap_probs = bootstrap_probs_all[:, j]

                should_remove, reason = self._check_sample_for_removal(sample, current_prob, bootstrap_probs)
                if should_remove:
                    to_remove.append(i)
                    to_plot_removed.append((sample.index, self.big_features, sample))
                    logger.debug(f"Removing {sample.source} from big: {reason}")
                else:
                    self._update_sample_weight(sample, current_prob, bootstrap_probs)

        # Plot removed samples
        for sample_idx, source_df, sample in to_plot_removed:
            sample_id = source_df[sample_idx, "id"] if "id" in source_df.columns else sample_idx
            logger.info(f"  Removed {sample.source}: id={sample_id}, row={sample_idx}, " f"original_conf={sample.confidence:.4f}")
            plot_sample(sample_idx, source_df, self.output_dir, f"removed_{sample.source}", self.config, action="removed")

        # Remove in reverse order to preserve indices
        for i in sorted(to_remove, reverse=True):
            self.labeled_train.pop(i)

        logger.info(f"Review: removed {len(to_remove)} pseudo-labels " f"(oos: {len(oos_samples)}, big: {len(big_samples)} reviewed)")
        return len(to_remove)

    def _check_sample_for_removal(
        self,
        sample: LabeledSample,
        current_prob: float,
        bootstrap_probs: np.ndarray,
    ) -> tuple[bool, str]:
        """
        Check if a sample should be removed based on current predictions.

        Returns (should_remove, reason).
        """
        cfg = self.config

        # Check pseudo-positives (and val_pool samples)
        if sample.label == 1 and sample.source in ["pseudo_pos", "val_pool"]:
            # Model changed its mind?
            if current_prob < cfg.pseudo_pos_removal_prob:
                return True, f"prob dropped: was {sample.confidence:.3f}, now {current_prob:.3f}"

            # Bootstrap diverged?
            bootstrap_std = np.std(bootstrap_probs)
            bootstrap_mean = np.mean(bootstrap_probs)
            if bootstrap_std > cfg.bootstrap_instability_std and bootstrap_mean < cfg.bootstrap_instability_mean:
                return True, f"unstable: std={bootstrap_std:.3f}, mean={bootstrap_mean:.3f}"

            return False, ""

        # Check pseudo-negatives
        if sample.label == 0 and sample.source == "pseudo_neg":
            # Model changed its mind? (less strict)
            if current_prob > cfg.pseudo_neg_promotion_prob:
                return True, f"prob rose: was {1-sample.confidence:.3f}, now {current_prob:.3f}"
            return False, ""

        # Check seed negatives (if review_seed_negatives is enabled)
        if sample.label == 0 and sample.source == "seed":
            # Very high threshold for removing seed samples - model must be very confident
            bootstrap_mean = np.mean(bootstrap_probs)
            if current_prob > cfg.seed_neg_removal_prob and bootstrap_mean > cfg.seed_neg_removal_bootstrap_mean:
                return True, f"seed neg looks like flare: P={current_prob:.3f}, bootstrap_mean={bootstrap_mean:.3f}"
            return False, ""

        return False, ""

    def _update_sample_weight(
        self,
        sample: LabeledSample,
        current_prob: float,
        bootstrap_probs: np.ndarray,
    ) -> None:
        """Update sample weight based on current prediction confidence."""
        if sample.label == 1 and sample.source in ["pseudo_pos", "val_pool"]:
            new_confidence = (current_prob + np.mean(bootstrap_probs)) / 2
            if sample.confidence > 0:
                sample.weight = min(1.0, sample.weight * (new_confidence / sample.confidence))

    def _phase8_balance_weights(self) -> dict[int, float] | None:
        """
        Phase 8: Balance class weights if needed.

        Returns class_weight dict or None.
        """
        eff_pos, eff_neg = self._compute_effective_sizes()
        logger.info(f"Effective train size: pos={eff_pos:.1f}, neg={eff_neg:.1f}")

        if eff_pos == 0:
            return None

        if eff_neg / eff_pos > 20:
            class_weight = {0: 1.0, 1: eff_neg / eff_pos / 10}
            logger.info(f"Class weight adjusted: {class_weight}")
            return class_weight

        return None

    def _phase9_retrain_model(self, class_weight: dict[int, float] | None) -> None:
        """Phase 9: Retrain main model with updated data."""
        features, labels, weights = self._build_training_data()
        self.model = train_model(
            features,
            labels,
            weights,
            class_weight=class_weight,
            config=self.config,
            random_state=self.random_state + len(self.metrics_history),
        )

    def _phase10_adaptive_thresholds(self) -> None:
        """Phase 10: Adjust thresholds based on stability."""
        cfg = self.config

        if self.n_successful_iters >= cfg.threshold_relax_successful_iters:
            # Model stable - relax thresholds
            self.pseudo_pos_threshold = max(
                cfg.min_pseudo_pos_threshold,
                self.pseudo_pos_threshold - cfg.threshold_relax_pos_delta,
            )
            self.pseudo_neg_threshold = min(
                cfg.max_pseudo_neg_threshold,
                self.pseudo_neg_threshold + cfg.threshold_relax_neg_delta,
            )
            self.max_pseudo_pos_per_iter = min(
                cfg.max_pseudo_pos_cap,
                self.max_pseudo_pos_per_iter + cfg.threshold_relax_max_pos_delta,
            )
            logger.info(f"Thresholds relaxed: pos>{self.pseudo_pos_threshold:.3f}, " f"neg<{self.pseudo_neg_threshold:.3f}")

        elif self.n_successful_iters == 0:
            # Just rolled back - tighten
            self.pseudo_pos_threshold = min(
                0.999,
                self.pseudo_pos_threshold + cfg.threshold_tighten_pos_delta,
            )
            self.pseudo_neg_threshold = max(
                0.01,
                self.pseudo_neg_threshold - cfg.threshold_tighten_neg_delta,
            )
            self.max_pseudo_pos_per_iter = max(
                cfg.min_pseudo_pos_per_iter,
                self.max_pseudo_pos_per_iter - cfg.threshold_tighten_max_pos_delta,
            )
            logger.info(f"Thresholds tightened: pos>{self.pseudo_pos_threshold:.3f}, " f"neg<{self.pseudo_neg_threshold:.3f}")

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
            logger.info("Computing enrichment (batched prediction)...")
            proba = self._predict_big_features_batched(self.model)
        else:
            proba = main_preds

        top_10k_indices = np.argsort(proba)[-10000:]
        top_10k_probs = proba[top_10k_indices]

        # Estimated flares in top-10K
        estimated_flares = float(np.sum(top_10k_probs))

        # Random baseline at assumed prevalence
        random_baseline = 10000 * self.config.assumed_prevalence

        enrichment = estimated_flares / random_baseline if random_baseline > 0 else 0.0

        return enrichment, estimated_flares

    def _check_stopping_criteria(
        self,
        iteration: int,
        held_out_metrics: dict,
        pseudo_pos_added: int,
    ) -> str | None:
        """
        Check stopping criteria. Returns reason string or None to continue.

        Priority order:
        1. Time-based (default: 5 hours)
        2. ICE-based (if target_ice configured)
        3. Plateau detection
        4. Data exhaustion
        5. Max iterations (optional)
        6. Instability
        """
        # 1. Time-based stopping (primary criterion)
        elapsed_hours = self._get_elapsed_hours()
        if elapsed_hours >= self.config.max_time_hours:
            logger.info(f"Time limit reached: {elapsed_hours:.2f} hours >= {self.config.max_time_hours} hours")
            return "TIME_LIMIT"

        # 2. ICE-based stopping (if target_ice is set)
        if self.config.target_ice is not None:
            current_ice = held_out_metrics.get("ice", 1.0)
            if current_ice <= self.config.target_ice:
                logger.info(f"ICE target reached: {current_ice:.4f} <= {self.config.target_ice}")
                return "ICE_TARGET_REACHED"

        # 3. Plateau check
        if len(self.metrics_history) >= 10:
            # Check both recall and ICE for plateau
            recent_recalls = [m.held_out_recall for m in self.metrics_history[-10:]]
            recent_ices = [m.held_out_ice for m in self.metrics_history[-10:] if m.held_out_ice > 0]

            recall_range = max(recent_recalls) - min(recent_recalls)
            ice_stable = len(recent_ices) >= 5 and (max(recent_ices) - min(recent_ices)) < 0.01

            if recall_range < 0.01 and ice_stable:
                recent_pos_added = [m.pseudo_pos_added for m in self.metrics_history[-5:]]
                if sum(recent_pos_added) == 0:
                    return "PLATEAU"

        # 4. Data exhaustion
        if len(self.validation_pool) <= 3:
            recent_pos_added = [m.pseudo_pos_added for m in self.metrics_history[-3:]]
            if sum(recent_pos_added) == 0:
                return "DATA_EXHAUSTED"

        # 5. Max iterations (optional - only if explicitly set)
        if self.config.max_iters is not None and iteration >= self.config.max_iters:
            return "MAX_ITERATIONS"

        # 6. Instability
        recent_rollbacks = sum(1 for r in self.rollback_history if r > iteration - 10)
        if recent_rollbacks > 3:
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
        max_iterations = self.config.max_iters if self.config.max_iters is not None else 100000

        for iteration in range(1, max_iterations + 1):
            elapsed = self._get_elapsed_hours()
            logger.info("=" * 60)
            logger.info(f"ITERATION {iteration} (elapsed: {elapsed:.2f} hours)")
            logger.info("=" * 60)

            # Phase 1: Validation and early stopping
            val_recall, held_out_metrics, should_continue, phase1_stop = self._phase1_validation_and_early_stopping(iteration)
            if phase1_stop:
                stop_reason = phase1_stop
                break
            if not should_continue:
                continue  # Skip rest of iteration (rollback happened)

            # Phase 2: Hard example mining
            val_pool_moved = self._phase2_hard_example_mining(val_recall)

            # Phase 3: Bootstrap models (stores models and indices in self for OOB)
            logger.info("Training bootstrap models...")
            self._phase3_train_bootstrap_models(iteration)

            # Phase 4: Predict on all data
            main_preds, bootstrap_preds, prediction_indices = self._phase4_predict_all()

            # Phase 5: Pseudo-label negatives
            pseudo_neg_added = self._phase5_pseudo_label_negatives(main_preds, bootstrap_preds, prediction_indices, iteration)

            # Phase 6: Pseudo-label positives
            pseudo_pos_added = self._phase6_pseudo_label_positives(main_preds, bootstrap_preds, prediction_indices, iteration)

            # Clear phase 4 predictions (no longer needed)
            del main_preds, bootstrap_preds, prediction_indices
            clean_ram()

            # Phase 7: Review old pseudo-labels
            pseudo_removed = self._phase7_review_pseudo_labels()

            # Phase 8: Balance weights
            class_weight = self._phase8_balance_weights()

            # Phase 9: Retrain model
            logger.info("Retraining main model...")
            self._phase9_retrain_model(class_weight)

            # Phase 10: Adaptive thresholds
            self._phase10_adaptive_thresholds()

            # Phase 11: Enrichment (computed AFTER retrain, but only every N iterations for speed)
            if self.config.enrichment_every_n_iters > 0 and iteration % self.config.enrichment_every_n_iters == 0:
                enrichment, estimated_flares_top10k = self._compute_enrichment_factor()
            else:
                # Use last known enrichment or 0
                enrichment = self.metrics_history[-1].enrichment_factor if self.metrics_history else 0.0
                estimated_flares_top10k = self.metrics_history[-1].estimated_flares_top10k if self.metrics_history else 0.0

            # Phase 11b: OOB metrics for stability monitoring
            if self.bootstrap_indices_list:
                features, labels, _ = self._build_training_data()
                oob_metrics = compute_oob_metrics(self.bootstrap_models, features, labels, self.bootstrap_indices_list)
                logger.info(
                    f"OOB metrics: recall={oob_metrics['oob_recall']:.3f}, "
                    f"precision={oob_metrics['oob_precision']:.3f}, "
                    f"coverage={oob_metrics['oob_coverage']:.2%}"
                )

                # Check OOB/held-out divergence
                divergence = abs(oob_metrics["oob_recall"] - held_out_metrics["recall"])
                if divergence > self.config.oob_divergence_warning:
                    logger.warning(f"OOB/held-out recall divergence: {divergence:.3f} — possible instability")

                del features, labels
                clean_ram()

            # Get tracked rowid probability
            tracked_prob = self._get_tracked_rowid_probability()

            # Phase 12: Logging and checkpointing
            counts = self._count_by_source()
            eff_pos, eff_neg = self._compute_effective_sizes()
            recent_rollbacks = sum(1 for r in self.rollback_history if r > iteration - 10)

            metrics = IterationMetrics(
                iteration=iteration,
                train_total=len(self.labeled_train),
                train_seed=counts["seed"],
                train_val_pool=counts["val_pool"],
                train_pseudo_pos=counts["pseudo_pos"],
                train_pseudo_neg=counts["pseudo_neg"],
                val_pool_remaining=len(self.validation_pool),
                effective_pos=eff_pos,
                effective_neg=eff_neg,
                val_recall=val_recall,
                held_out_recall=held_out_metrics["recall"],
                held_out_precision=held_out_metrics["precision"],
                held_out_auc=held_out_metrics["auc"],
                held_out_f1=held_out_metrics["f1"],
                held_out_logloss=held_out_metrics.get("logloss", 0.0),
                held_out_brier=held_out_metrics.get("brier", 0.0),
                held_out_ice=held_out_metrics.get("ice", 0.0),
                held_out_pr_auc=held_out_metrics.get("pr_auc", 0.0),
                enrichment_factor=enrichment,
                estimated_flares_top10k=estimated_flares_top10k,
                pseudo_pos_threshold=self.pseudo_pos_threshold,
                pseudo_neg_threshold=self.pseudo_neg_threshold,
                pseudo_pos_added=pseudo_pos_added,
                pseudo_neg_added=pseudo_neg_added,
                pseudo_removed=pseudo_removed,
                val_pool_moved=val_pool_moved,
                n_successful_iters=self.n_successful_iters,
                n_rollbacks_recent=recent_rollbacks,
                elapsed_hours=self._get_elapsed_hours(),
                tracked_rowid_prob=tracked_prob,
            )
            self.metrics_history.append(metrics)
            self._save_checkpoint(iteration, asdict(metrics))

            # Log summary
            logger.info("=" * 60)
            logger.info(f"ITERATION {iteration} COMPLETE")
            logger.info(f"  Train: {len(self.labeled_train)} (seed:{counts['seed']}, " f"pseudo_pos:{counts['pseudo_pos']}, pseudo_neg:{counts['pseudo_neg']})")
            logger.info(f"  Val recall: {val_recall:.3f}")
            logger.info(
                f"  Held-out: recall={held_out_metrics['recall']:.3f}, "
                f"precision={held_out_metrics['precision']:.3f}, "
                f"ROC-AUC={held_out_metrics.get('auc', 0):.3f}, "
                f"ICE={held_out_metrics.get('ice', 0):.4f}"
            )
            logger.info(
                f"  PR-AUC={held_out_metrics.get('pr_auc', 0):.3f}, "
                f"Logloss={held_out_metrics.get('logloss', 0):.4f}, "
                f"Brier={held_out_metrics.get('brier', 0):.4f}"
            )
            logger.info(f"  Enrichment: {enrichment:.1f}x")
            logger.info(f"  Elapsed: {self._get_elapsed_hours():.2f} hours")
            logger.info(f"  Successful iters: {self.n_successful_iters}")
            logger.info("=" * 60)

            # Update previous metrics
            self.prev_val_recall = val_recall
            self.prev_held_out_recall = held_out_metrics["recall"]
            self.prev_held_out_precision = held_out_metrics["precision"]
            self.prev_held_out_ice = held_out_metrics.get("ice", 1.0)

            # Check stopping criteria
            stop_reason = self._check_stopping_criteria(iteration, held_out_metrics, pseudo_pos_added)
            if stop_reason:
                break

        # Finalization
        return self._finalize(stop_reason or "TIME_LIMIT")

    def _finalize(self, stop_reason: str) -> dict:
        """Finalize pipeline and save results."""
        logger.info("=" * 60)
        logger.info("FINALIZATION")
        logger.info(f"Stop reason: {stop_reason}")
        logger.info("=" * 60)

        # Load best model
        if self.best_checkpoint is not None:
            logger.info(f"Loading best model from iteration {self.best_checkpoint.iteration}")
            self.model = joblib.load(self.best_checkpoint.model_path)

        # Save final model
        final_model_path = self.output_dir / "final_model.joblib"
        joblib.dump(self.model, final_model_path)
        logger.info(f"Final model saved to {final_model_path}")

        # Save labeled training set
        labeled_train_data = [asdict(s) for s in self.labeled_train]
        labeled_train_df = pl.DataFrame(labeled_train_data)
        labeled_train_path = self.output_dir / "labeled_train.parquet"
        labeled_train_df.write_parquet(labeled_train_path, compression="zstd")
        logger.info(f"Labeled train saved to {labeled_train_path}")

        # Generate candidate lists using batched prediction
        logger.info("Generating candidate lists (batched prediction)...")
        final_proba = self._predict_big_features_batched(self.model)

        # Add proba column to big_features for filtering
        big_with_proba = self.big_features.with_columns(pl.Series("proba", final_proba))

        candidates = {
            "high_purity": big_with_proba.filter(pl.col("proba") > 0.95),
            "balanced": big_with_proba.filter(pl.col("proba") > 0.80),
            "high_recall": big_with_proba.filter(pl.col("proba") > 0.50),
        }

        del big_with_proba, final_proba
        clean_ram()

        for name, df in candidates.items():
            path = self.output_dir / f"candidates_{name}.parquet"
            df.write_parquet(path, compression="zstd")
            logger.info(f"  {name}: {len(df):,} candidates saved to {path}")

        # Save metrics history
        metrics_path = self.output_dir / "metrics_history.json"
        with open(metrics_path, "w") as f:
            json.dump(
                [asdict(m) for m in self.metrics_history],
                f,
                indent=2,
                default=str,
            )
        logger.info(f"Metrics history saved to {metrics_path}")

        # Summary
        best_metrics = self.best_checkpoint.metrics if self.best_checkpoint else {}
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Best iteration: {self.best_checkpoint.iteration if self.best_checkpoint else 0}")
        logger.info(f"Best recall: {best_metrics.get('recall', 0):.3f}")
        logger.info(f"Best precision: {best_metrics.get('precision', 0):.3f}")
        logger.info(f"Final train size: {len(self.labeled_train)}")
        logger.info(f"Candidates (high purity): {len(candidates['high_purity']):,}")
        logger.info(f"Candidates (balanced): {len(candidates['balanced']):,}")
        logger.info(f"Candidates (high recall): {len(candidates['high_recall']):,}")
        logger.info("=" * 60)

        return {
            "final_model": self.model,
            "labeled_train": labeled_train_df,
            "candidates": candidates,
            "metrics_history": self.metrics_history,
            "best_iteration": self.best_checkpoint.iteration if self.best_checkpoint else 0,
            "stop_reason": stop_reason,
        }


# =============================================================================
# Convenience Function
# =============================================================================


def run_active_learning_pipeline(
    big_features: pl.DataFrame,
    oos_features: pl.DataFrame,
    config: PipelineConfig | None = None,
    output_dir: Path | str = "active_learning_output",
    random_state: int = 42,
) -> dict:
    """
    Run zero-expert self-training pipeline.

    Parameters
    ----------
    big_features : pl.DataFrame
        Large unlabeled dataset (~94M observations).
        Expected columns: feature columns, optionally 'id'.
    oos_features : pl.DataFrame
        Known flares dataset (99 samples with class=1).
        Expected columns: feature columns, 'class', optionally 'id'.
    config : PipelineConfig, optional
        Pipeline configuration. Uses defaults if not provided.
    output_dir : Path or str
        Directory for saving checkpoints and results.
    random_state : int
        Random seed for reproducibility.

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
    pipeline = ActiveLearningPipeline(
        big_features=big_features,
        oos_features=oos_features,
        config=config,
        output_dir=output_dir,
        random_state=random_state,
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
    # big_features = pl.read_parquet("path/to/big_features.parquet")
    # oos_features = pl.read_parquet("path/to/oos_features.parquet")

    # Run pipeline
    # results = run_active_learning_pipeline(
    #     big_features=big_features,
    #     oos_features=oos_features,
    #     output_dir="active_learning_output",
    # )

    print("Pipeline module loaded. Use run_active_learning_pipeline() to start.")

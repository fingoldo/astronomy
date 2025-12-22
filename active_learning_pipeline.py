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
"""

from __future__ import annotations

import json
import logging
import psutil
import time
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
# Module Constants
# =============================================================================

DEFAULT_STRATIFICATION_BINS = 5
MIN_SAMPLES_PER_STRATIFICATION_BIN = 10
MIN_BOOTSTRAP_CONSENSUS_MODELS = 2
FALLBACK_MAX_ITERATIONS = 100_000
MAX_PSEUDO_POS_THRESHOLD_BOUND = 0.999
MIN_PSEUDO_NEG_THRESHOLD_BOUND = 0.01


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


# =============================================================================
# Helper Functions for Improved Pipeline
# =============================================================================


def _random_split(
    n_samples: int,
    train_ratio: float,
    val_ratio: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random fallback split for stratified_flare_split."""
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(n_samples)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    return (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_val],
        shuffled[n_train + n_val :],
    )


def compute_consensus_score(bootstrap_probs: np.ndarray | list[float]) -> float:
    """
    Compute consensus score from bootstrap model predictions.

    The consensus score measures agreement among bootstrap models. Higher values
    indicate more stable, reliable predictions. Computed as 1 - std, giving a
    range of approximately [0.5, 1.0] for probability predictions.

    Parameters
    ----------
    bootstrap_probs : array-like
        Predicted probabilities from bootstrap models.

    Returns
    -------
    float
        Consensus score in [0.5, 1.0]. Higher = more agreement.
    """
    return 1.0 - float(np.std(bootstrap_probs))


def stratified_flare_split(
    known_flares: pl.DataFrame,
    train_ratio: float = 0.10,
    val_ratio: float = 0.40,
    held_out_ratio: float = 0.50,
    stratify_cols: list[str] | None = None,
    random_state: int = 42,
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
        return _random_split(n_samples, train_ratio, val_ratio, random_state)

    # Build stratification labels
    try:
        # Extract each column separately from polars to avoid shape issues
        strat_arrays = []
        for col in stratify_cols:
            col_data = known_flares[col].to_numpy()
            # Ensure 1D and float
            col_data = np.asarray(col_data, dtype=np.float64).flatten()
            strat_arrays.append(col_data)

        # Stack into 2D array (n_samples, n_cols)
        if len(strat_arrays) == 1:
            strat_data = strat_arrays[0].reshape(-1, 1)
        else:
            strat_data = np.column_stack(strat_arrays)

        # Use quantile binning to create strata
        n_bins = min(DEFAULT_STRATIFICATION_BINS, n_samples // MIN_SAMPLES_PER_STRATIFICATION_BIN)
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

        logger.info(f"Stratified split: train={len(train_idx)}, val={len(val_idx)}, held_out={len(held_out_idx)}")

        return train_idx, val_idx, held_out_idx

    except ValueError as e:
        logger.warning(f"Stratified split failed ({e}), falling back to random")
        return _random_split(n_samples, train_ratio, val_ratio, random_state)


def get_adaptive_curriculum_weight(
    confidence: float,
    current_recall: float,
    config: PipelineConfig | None = None,
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
    confidence : float
        Pseudo-label confidence in [0, 1]. For positives, this is P(flare).
        For negatives, this is 1 - P(flare).
    current_recall : float
        Current held-out recall in [0, 1]. Measures model maturity.
    config : PipelineConfig, optional
        Configuration with curriculum thresholds. If None, uses defaults.

    Returns
    -------
    float
        Sample weight in [0, 1]. Weight of 0 means "don't include this sample".

    References
    ----------
    - Bengio et al. (2009). Curriculum Learning. ICML.
    - Kumar et al. (2010). Self-Paced Learning for Latent Variable Models.
    """
    # Use config values or defaults
    if config is not None:
        phase1_recall = config.curriculum.phase1_recall
        phase2_recall = config.curriculum.phase2_recall
        phase1_conf = config.curriculum.phase1_conf
        phase2_conf = config.curriculum.phase2_conf
        phase3_conf = config.curriculum.phase3_conf
        phase2_weight = config.curriculum.phase2_weight
    else:
        # Defaults (matching original hardcoded values)
        phase1_recall = 0.5
        phase2_recall = 0.65
        phase1_conf = 0.95
        phase2_conf = 0.85
        phase3_conf = 0.70
        phase2_weight = 0.7

    if current_recall < phase1_recall:
        # Phase 1: Strict - only very confident pseudo-labels
        return 1.0 if confidence > phase1_conf else 0.0

    elif current_recall < phase2_recall:
        # Phase 2: Medium - expand with reduced weights
        if confidence > phase1_conf:
            return 1.0
        elif confidence > phase2_conf:
            return phase2_weight
        else:
            return 0.0

    else:
        # Phase 3: Mature - use gradient weights
        if confidence > phase1_conf:
            return 1.0
        elif confidence > phase3_conf:
            return confidence  # Linear weight
        else:
            return 0.0


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
    unlabeled_features : pl.DataFrame
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
        unlabeled_features: pl.DataFrame,
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
        self.unlabeled_features = unlabeled_features
        self.known_flares = known_flares
        self.feature_cols = feature_cols
        self.current_recall = current_recall
        self.config = config
        self._prepared_df: pl.DataFrame | None = None

    def add_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Build DataFrame from labeled samples.

        This method constructs the training DataFrame by extracting features
        from the appropriate source (known_flares or unlabeled_features) for each
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

        # Batch extract from known_flares
        if oos_samples:
            oos_indices = [s.index for _, s in oos_samples]
            oos_data = self.known_flares[oos_indices].select(self.feature_cols).to_dicts()
            for j, (i, sample) in enumerate(oos_samples):
                row_data = oos_data[j].copy()
                row_data["_label"] = sample.label
                row_data["_confidence"] = sample.confidence
                row_data["_source"] = sample.source
                weight = get_adaptive_curriculum_weight(sample.confidence, self.current_recall, self.config)
                row_data["_weight"] = sample.weight * weight if weight > 0 else 0.0
                rows[i] = row_data

        # Batch extract from unlabeled_features
        if big_samples:
            big_indices = [s.index for _, s in big_samples]
            big_data = self.unlabeled_features[big_indices].select(self.feature_cols).to_dicts()
            for j, (i, sample) in enumerate(big_samples):
                row_data = big_data[j].copy()
                row_data["_label"] = sample.label
                row_data["_confidence"] = sample.confidence
                row_data["_source"] = sample.source
                weight = get_adaptive_curriculum_weight(sample.confidence, self.current_recall, self.config)
                row_data["_weight"] = sample.weight * weight if weight > 0 else 0.0
                rows[i] = row_data

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
    unlabeled_features: pl.DataFrame,
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
    unlabeled_features : pl.DataFrame
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
        unlabeled_features=unlabeled_features,
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

    n_train_flares: int = 50  # Initial training flares
    n_train_neg_init: int = 1000  # Initial training negatives
    n_validation_flares: int = 25  # Validation set flares (for rollback/model selection)
    n_validation_neg: int = 5000  # Validation set negatives
    n_held_out_flares: int = 24  # Held-out flares (never used for decisions)
    n_held_out_neg: int = 5000  # Held-out negatives


@dataclass
class CatBoostConfig:
    """CatBoost model hyperparameters."""

    iterations: int = 1000
    depth: int = 8
    learning_rate: float = 0.1
    verbose: bool = False
    use_gpu: bool = True
    eval_fraction: float = 0.1  # Fraction for auto early stopping
    early_stopping_rounds: int = 50
    plot: bool = True  # Plot training progress
    loss_function: str = "Logloss"
    eval_metric: str = "Logloss"


@dataclass
class CurriculumConfig:
    """Curriculum learning phase thresholds."""

    phase1_recall: float = 0.5  # Below this: strict phase
    phase2_recall: float = 0.65  # Below this: medium phase
    phase1_conf: float = 0.95  # Required confidence in phase 1
    phase2_conf: float = 0.85  # Required confidence in phase 2
    phase3_conf: float = 0.70  # Required confidence in phase 3
    phase2_weight: float = 0.7  # Weight for medium-confidence samples in phase 2


@dataclass
class ThresholdConfig:
    """Pseudo-labeling thresholds and adaptive adjustment settings."""

    # Initial thresholds
    pseudo_pos_threshold: float = 0.99
    pseudo_neg_threshold: float = 0.05
    consensus_threshold: float = 0.95

    # Limits per iteration
    max_pseudo_pos_per_iter: int = 10
    max_pseudo_neg_per_iter: int = 100

    # Adaptive adjustments
    relax_successful_iters: int = 3  # Iters before relaxing
    relax_pos_delta: float = 0.005
    relax_neg_delta: float = 0.01
    relax_max_pos_delta: int = 1
    tighten_pos_delta: float = 0.01
    tighten_neg_delta: float = 0.02
    tighten_max_pos_delta: int = 3

    # Bounds
    min_pseudo_pos_threshold: float = 0.95
    max_pseudo_neg_threshold: float = 0.10
    min_pseudo_pos_per_iter: int = 3
    max_pseudo_pos_cap: int = 20

    # Review thresholds
    pseudo_pos_removal_prob: float = 0.3
    pseudo_neg_promotion_prob: float = 0.7
    bootstrap_instability_std: float = 0.2
    bootstrap_instability_mean: float = 0.7
    seed_neg_removal_prob: float = 0.9
    seed_neg_removal_bootstrap_mean: float = 0.85
    neg_consensus_min_low_prob: float = 0.1

    # Bootstrap
    n_bootstrap_models: int = 5
    bootstrap_variance_threshold: float = 0.05


@dataclass
class StoppingConfig:
    """Stopping criteria and plateau detection settings."""

    # Enrichment calculation
    top_k_candidates: int = 10000  # Number of top candidates for enrichment

    # Plateau detection
    plateau_window: int = 10  # Iterations to check for plateau
    plateau_recall_range: float = 0.01  # Min recall variation for plateau
    plateau_ice_range: float = 0.01  # Min ICE variation for plateau
    plateau_min_zero_pos_iters: int = 5  # Iterations with 0 pseudo_pos for plateau

    # Instability detection
    max_rollbacks_for_instability: int = 3  # Max rollbacks in window before stopping
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
    """

    # Nested configuration groups
    data: DataSplitConfig = field(default_factory=DataSplitConfig)
    catboost: CatBoostConfig = field(default_factory=CatBoostConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    stopping: StoppingConfig = field(default_factory=StoppingConfig)

    # Stopping criteria
    max_iters: int | None = None
    max_time_hours: float = 5.0
    target_ice: float | None = None

    # Success targets
    target_recall: float = 0.75
    target_precision: float = 0.60
    target_enrichment: float = 50.0
    min_successful_iters_for_success: int = 5

    # Sample weights
    initial_pseudo_pos_weight: float = 0.2
    initial_pseudo_neg_weight: float = 0.8
    weight_increment: float = 0.1
    max_pseudo_pos_weight: float = 0.8

    # Class balancing
    class_imbalance_threshold: float = 20.0
    class_weight_divisor: float = 10.0

    # Rollback
    ice_increase_threshold: float = 0.05
    oob_divergence_warning: float = 0.15

    # Enrichment
    assumed_prevalence: float = 0.001
    enrichment_every_n_iters: int = 5

    # Prediction
    prediction_batch_size: int = 100_000_000

    # Plotting
    plot_samples: bool = True
    plot_singlepoint_min_outlying_factor: float = 10.0

    # Misc
    review_seed_negatives: bool = True
    track_rowid: int | None = 55554273
    exclude_columns: set[str] = field(default_factory=lambda: {"id", "class", "ts", "index"})

    # mlframe integration
    use_mlframe: bool = False
    mlframe_models: list[str] = field(default_factory=lambda: ["cb"])

@dataclass(slots=True)
class LabeledSample:
    """A single labeled sample in the training set."""

    index: int  # Index in the source DataFrame (unlabeled_features or known_flares)
    label: int  # 0 or 1
    weight: float  # Sample weight for training
    source: SampleSource  # Origin of the sample (seed, pseudo_pos, pseudo_neg)
    added_iter: int  # Iteration when added
    confidence: float  # P(label) when added
    consensus_score: float  # Bootstrap consensus [0, 1]
    from_known_flares: bool = False  # True if from known_flares, False if from unlabeled_features


@dataclass(slots=True)
class SampleRemovalInfo:
    """Information about a sample marked for removal during pseudo-label review."""

    sample_index: int  # Index in the source DataFrame
    source_df: pl.DataFrame  # Source DataFrame (known_flares or unlabeled_features)
    dataset: object  # HuggingFace dataset for plotting (optional)
    sample: LabeledSample  # The labeled sample being removed


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
    """Truly held-out set - NEVER used for any training decisions.

    This set is only evaluated once at the very end (in _finalize) to provide
    honest, unbiased metrics that have not been used for model selection,
    rollback decisions, or threshold adjustments.
    """

    flare_indices: np.ndarray  # Indices in known_flares
    negative_indices: np.ndarray  # Indices in unlabeled_features


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
    """

    flare_indices: np.ndarray  # Indices in known_flares
    negative_indices: np.ndarray  # Indices in unlabeled_features


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
    task_type = "GPU" if config.catboost.use_gpu else "CPU"

    # Build model with calibration-focused loss and native eval_fraction
    model = CatBoostClassifier(
        iterations=config.catboost.iterations,
        depth=config.catboost.depth,
        learning_rate=config.catboost.learning_rate,
        random_seed=random_state,
        verbose=config.catboost.verbose,
        auto_class_weights=None,  # We handle weights manually
        loss_function=config.catboost.loss_function,  # Logloss for calibration
        eval_metric=config.catboost.eval_metric,  # Logloss instead of AUC
        task_type=task_type,
        early_stopping_rounds=config.catboost.early_stopping_rounds,
        eval_fraction=config.catboost.eval_fraction if config.catboost.eval_fraction > 0 else None,
    )

    model.fit(
        features,
        labels,
        sample_weight=sample_weights,
        plot=config.catboost.plot and plot_file is not None,
        plot_file=str(plot_file) if plot_file else None,
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
) -> np.ndarray:
    """Extract feature matrix for given indices."""
    subset = df[indices].select(feature_cols)
    return subset.to_numpy().astype(np.float32)


def plot_sample(
    sample_index: int,
    source_df: pl.DataFrame,
    output_dir: Path,
    prefix: str,
    config: PipelineConfig,
    action: str = "added",
    dataset=None,
) -> None:
    """
    Plot a sample in both raw and cleaned modes.

    Parameters
    ----------
    sample_index : int
        Index in the source DataFrame / dataset.
    source_df : pl.DataFrame
        Source DataFrame (unlabeled_features or known_flares). Used for ID lookup.
    output_dir : Path
        Base output directory.
    prefix : str
        Prefix for filenames (e.g., "pseudo_pos", "pseudo_neg").
    config : PipelineConfig
        Pipeline configuration.
    action : str
        Action description ("added" or "removed").
    dataset : HuggingFace Dataset, optional
        Original light curve dataset for plotting. If provided, uses
        dataset["target"][sample_index] for plotting instead of source_df.
    """
    if not VIEW_SERIES_AVAILABLE or not config.plot_samples:
        return

    if dataset is None:
        logger.debug(f"No dataset provided for plotting sample {sample_index}, skipping")
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

        # Get the light curve data from HuggingFace dataset
        lc_data = dataset["target"][sample_index]

        # Plot raw
        raw_file = plots_dir / f"{action}_{sample_id}_row{row_num}_raw.png"
        fig = view_series(
            lc_data,
            index=sample_index,
            backend=backend,
            plot_file=str(raw_file) if backend == "matplotlib" else None,
        )
        # Display in Jupyter if using plotly
        if backend == "plotly" and fig is not None:
            fig.show()

        # Plot cleaned
        cleaned_file = plots_dir / f"{action}_{sample_id}_row{row_num}_cleaned.png"
        fig = view_series(
            lc_data,
            index=sample_index,
            backend=backend,
            singlepoint_min_outlying_factor=config.plot_singlepoint_min_outlying_factor,
            plot_file=str(cleaned_file) if backend == "matplotlib" else None,
        )
        # Display in Jupyter if using plotly
        if backend == "plotly" and fig is not None:
            fig.show()

        logger.info(f"Plotted sample {sample_id} (row {row_num}) - {action}")

    except (IndexError, KeyError, ValueError, OSError) as e:
        logger.warning(f"Failed to plot sample {sample_index}: {e}")


def report_held_out_metrics(
    targets: np.ndarray,
    probs: np.ndarray,
    iteration: int,
    output_dir: Path,
    config: PipelineConfig,
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

            if isinstance(returned_metrics, dict) and returned_metrics:
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
        unlabeled_features: pl.DataFrame,
        known_flares: pl.DataFrame,
        config: PipelineConfig | None = None,
        output_dir: Path | str = "active_learning_output",
        random_state: int = 42,
        unlabeled_features_dataset=None,
        known_flares_dataset=None,
    ):
        """
        Initialize the pipeline.

        Parameters
        ----------
        unlabeled_features : pl.DataFrame
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
        unlabeled_features_dataset : HuggingFace Dataset, optional
            Original light curve data for unlabeled_features, used for plotting.
            Access pattern: dataset["target"][row_index]
        known_flares_dataset : HuggingFace Dataset, optional
            Original light curve data for known_flares, used for plotting.
            Access pattern: dataset["target"][row_index]
        """
        self.unlabeled_features = unlabeled_features
        self.known_flares = known_flares
        self.config = config or PipelineConfig()
        self.unlabeled_features_dataset = unlabeled_features_dataset
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
        self.feature_cols = get_feature_columns(self.known_flares, self.config.exclude_columns)
        logger.info(f"Using {len(self.feature_cols)} feature columns")

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
        self.n_successful_iters: int = 0
        self.rollback_history: list[int] = []  # Iteration numbers of rollbacks

        # Current thresholds (mutable)
        self.pseudo_pos_threshold = self.config.thresholds.pseudo_pos_threshold
        self.pseudo_neg_threshold = self.config.thresholds.pseudo_neg_threshold
        self.max_pseudo_pos_per_iter = self.config.thresholds.max_pseudo_pos_per_iter

        # Time tracking
        self.start_time: float | None = None

        # Cached feature views for efficiency (zero-copy pandas view of polars DataFrame)
        self._unlabeled_view: np.ndarray | None = None

        # Track rowid index mapping
        self._tracked_rowid_index: int | None = None
        if self.config.track_rowid is not None and "id" in self.unlabeled_features.columns:
            # Find the index of the tracked rowid
            mask = self.unlabeled_features["id"] == self.config.track_rowid
            indices = np.where(mask.to_numpy())[0]
            if len(indices) > 0:
                self._tracked_rowid_index = int(indices[0])
                logger.info(f"Tracking rowid {self.config.track_rowid} at index {self._tracked_rowid_index}")

        # Running counts for efficiency (avoid O(n) scans)
        self._source_counts: dict[SampleSource, int] = {
            SampleSource.SEED: 0,
            SampleSource.PSEUDO_POS: 0,
            SampleSource.PSEUDO_NEG: 0,
        }
        self._effective_pos: float = 0.0
        self._effective_neg: float = 0.0
        self._labeled_unlabeled_indices: set[int] = set()
        self._labeled_known_flare_indices: set[int] = set()

    def _validate_inputs(self) -> None:
        """Validate input DataFrames."""
        n_flares = len(self.known_flares)
        data = self.config.data
        required_flares = data.n_train_flares + data.n_validation_flares + data.n_held_out_flares
        if n_flares < required_flares:
            raise ValueError(f"Need at least {required_flares} known flares, got {n_flares}")

        required_negs = data.n_train_neg_init + data.n_validation_neg + data.n_held_out_neg
        if len(self.unlabeled_features) < required_negs:
            raise ValueError(f"unlabeled_features too small for required negative samples ({required_negs})")

        logger.info(f"unlabeled_features: {len(self.unlabeled_features):,} samples")
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
        return sample

    def _get_labeled_indices(self) -> tuple[set[int], set[int]]:
        """Get sets of already labeled indices (unlabeled_features, known_flares)."""
        return self._labeled_unlabeled_indices, self._labeled_known_flare_indices

    def _build_pseudo_label_exclusion_set(self) -> set[int]:
        """
        Build set of unlabeled_features indices to exclude from pseudo-labeling.

        Excludes:
        - Already labeled samples (from _labeled_unlabeled_indices)
        - Held-out negative samples (to preserve honest evaluation)

        Returns
        -------
        set[int]
            Indices in unlabeled_features that should not receive pseudo-labels.
        """
        labeled_big_indices, _ = self._get_labeled_indices()
        held_out_set = set(self.held_out.negative_indices.tolist()) if self.held_out else set()
        return labeled_big_indices | held_out_set

    def _build_training_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build feature matrix, labels, and weights from labeled_train (batched by source)."""
        # Separate by source for batched extraction
        oos_items = [(i, s) for i, s in enumerate(self.labeled_train) if s.from_known_flares]
        big_items = [(i, s) for i, s in enumerate(self.labeled_train) if not s.from_known_flares]

        n_samples = len(self.labeled_train)
        features = np.zeros((n_samples, len(self.feature_cols)), dtype=np.float32)
        labels = np.zeros(n_samples, dtype=np.int32)
        weights = np.zeros(n_samples, dtype=np.float32)

        # Batch extract from known_flares
        if oos_items:
            oos_indices = [s.index for _, s in oos_items]
            known_flares = extract_features_array(self.known_flares, oos_indices, self.feature_cols)
            for j, (i, sample) in enumerate(oos_items):
                features[i] = known_flares[j]
                labels[i] = sample.label
                weights[i] = sample.weight

        # Batch extract from unlabeled_features
        if big_items:
            big_indices = [s.index for _, s in big_items]
            unlabeled_features_batch = extract_features_array(self.unlabeled_features, big_indices, self.feature_cols)
            for j, (i, sample) in enumerate(big_items):
                features[i] = unlabeled_features_batch[j]
                labels[i] = sample.label
                weights[i] = sample.weight

        return features, labels, weights

    def _compute_metrics_for_set(
        self,
        flare_indices: np.ndarray,
        negative_indices: np.ndarray,
    ) -> dict[str, float]:
        """
        Compute classification metrics on a labeled set.

        Shared implementation used by both validation and held-out evaluation.

        Parameters
        ----------
        flare_indices : np.ndarray
            Indices of positive samples in known_flares.
        negative_indices : np.ndarray
            Indices of negative samples in unlabeled_features.

        Returns
        -------
        dict[str, float]
            Metrics: recall, precision, auc, f1, logloss, brier, ice, pr_auc
        """
        if self.model is None:
            return {"recall": 0.0, "precision": 0.0, "auc": 0.5, "f1": 0.0, "logloss": 1.0, "brier": 0.25, "ice": 0.5}

        # Flares from known_flares
        flare_features = extract_features_array(self.known_flares, flare_indices, self.feature_cols)
        flare_proba = self.model.predict_proba(flare_features)[:, 1]
        flare_labels = np.ones(len(flare_indices), dtype=np.int32)

        # Negatives from unlabeled_features
        neg_features = extract_features_array(self.unlabeled_features, negative_indices, self.feature_cols)
        neg_proba = self.model.predict_proba(neg_features)[:, 1]
        neg_labels = np.zeros(len(negative_indices), dtype=np.int32)

        # Combined
        all_proba = np.concatenate([flare_proba, neg_proba])
        all_preds = (all_proba >= 0.5).astype(int)
        all_labels = np.concatenate([flare_labels, neg_labels])

        # Basic metrics
        has_both_classes = len(np.unique(all_labels)) > 1
        metrics = {
            "recall": recall_score(all_labels, all_preds, zero_division=0),
            "precision": precision_score(all_labels, all_preds, zero_division=0),
            "auc": roc_auc_score(all_labels, all_proba) if has_both_classes else 0.5,
            "f1": f1_score(all_labels, all_preds, zero_division=0),
            "logloss": log_loss(all_labels, all_proba) if has_both_classes else 1.0,
            "brier": brier_score_loss(all_labels, all_proba),
            "ice": _compute_ice(all_labels, all_proba),
            "pr_auc": average_precision_score(all_labels, all_proba) if has_both_classes else 0.0,
        }

        return metrics

    def _compute_validation_metrics(self, iteration: int = 0) -> dict[str, float]:
        """
        Compute metrics on validation set (used for training decisions).

        This set IS used for rollback decisions, model selection, and threshold
        adjustments. Metrics from this set have model selection bias.
        """
        if self.validation is None or self.model is None:
            return {"recall": 0.0, "precision": 0.0, "auc": 0.5, "f1": 0.0, "logloss": 1.0, "brier": 0.25, "ice": 0.5}

        # Extract features ONCE and compute predictions
        flare_features = extract_features_array(self.known_flares, self.validation.flare_indices, self.feature_cols)
        neg_features = extract_features_array(self.unlabeled_features, self.validation.negative_indices, self.feature_cols)

        flare_proba = self.model.predict_proba(flare_features)[:, 1]
        neg_proba = self.model.predict_proba(neg_features)[:, 1]

        # Combine labels and predictions
        all_labels = np.concatenate([
            np.ones(len(self.validation.flare_indices), dtype=np.int32),
            np.zeros(len(self.validation.negative_indices), dtype=np.int32),
        ])
        all_proba = np.concatenate([flare_proba, neg_proba])
        all_preds = (all_proba >= 0.5).astype(int)

        # Compute metrics
        has_both_classes = len(np.unique(all_labels)) > 1
        metrics = {
            "recall": recall_score(all_labels, all_preds, zero_division=0),
            "precision": precision_score(all_labels, all_preds, zero_division=0),
            "auc": roc_auc_score(all_labels, all_proba) if has_both_classes else 0.5,
            "f1": f1_score(all_labels, all_preds, zero_division=0),
            "logloss": log_loss(all_labels, all_proba) if has_both_classes else 1.0,
            "brier": brier_score_loss(all_labels, all_proba),
            "ice": _compute_ice(all_labels, all_proba),
            "pr_auc": average_precision_score(all_labels, all_proba) if has_both_classes else 0.0,
        }

        # Add report_model_perf metrics if available (reuse extracted data)
        additional = report_held_out_metrics(all_labels, all_proba, iteration, self.output_dir, self.config)
        for key, value in additional.items():
            if key not in metrics:
                metrics[key] = value

        return metrics

    def _compute_held_out_metrics_final(self) -> dict[str, float]:
        """
        Compute metrics on the truly held-out set (ONLY called in _finalize).

        This method is called exactly once at the end of training to provide
        honest, unbiased evaluation metrics. The held-out set was NEVER used
        for any training decisions (rollback, model selection, thresholds).
        """
        if self.held_out is None or self.model is None:
            return {"recall": 0.0, "precision": 0.0, "auc": 0.5, "f1": 0.0, "logloss": 1.0, "brier": 0.25, "ice": 0.5}

        return self._compute_metrics_for_set(
            self.held_out.flare_indices,
            self.held_out.negative_indices,
        )

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

    def _reset_running_counts(self) -> None:
        """Reset all running counts to zero. Called before rebuilding from checkpoint."""
        self._source_counts = {
            SampleSource.SEED: 0,
            SampleSource.PSEUDO_POS: 0,
            SampleSource.PSEUDO_NEG: 0,
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
        """Count samples by source (returns cached counts)."""
        return self._source_counts.copy()

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

    def _get_unlabeled_features_for_prediction(self):
        """
        Get feature view for unlabeled_features using zero-copy pandas view.

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
        if self._unlabeled_view is not None:
            return self._unlabeled_view

        # Report RAM before
        ram_before = psutil.Process().memory_info().rss / 1024**3
        logger.info(f"Creating zero-copy view of unlabeled_features for prediction... (RAM: {ram_before:.2f} GB)")

        if MLFRAME_AVAILABLE and get_pandas_view_of_polars_df is not None:
            # Use zero-copy pandas view - do NOT call .values (causes copy!)
            self._unlabeled_view = get_pandas_view_of_polars_df(self.unlabeled_features.select(self.feature_cols))
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
        Predict on unlabeled_features using cached view or batched extraction.

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
            Predictions (probabilities) for all samples in unlabeled_features.
        """
        if model is None:
            model = self.model
        if model is None:
            raise ValueError("No model available for prediction")

        n_total = len(self.unlabeled_features)
        batch_size = self.config.prediction_batch_size

        # Option 1: Try cached zero-copy view
        features_view = self._get_unlabeled_features_for_prediction()
        if features_view is not None:
            # Smart bypass: if data fits in one batch, predict directly without batching
            if n_total <= batch_size:
                logger.info(f"Predicting on {n_total:,} samples (single pass)...")
                return model.predict_proba(features_view)[:, 1].astype(np.float32)
            else:
                logger.info(f"Predicting on {n_total:,} samples using cached view (batched)...")
                return predict_proba_batched(
                    model,
                    features_view,
                    batch_size=batch_size,
                    desc="Prediction",
                )

        # Fallback: batch-by-batch extraction (for when zero-copy view unavailable)
        logger.info(f"Predicting on {n_total:,} samples using batched extraction...")
        all_preds = np.zeros(n_total, dtype=np.float32)

        # Smart bypass for fallback too
        if n_total <= batch_size:
            all_features = extract_features_array(self.unlabeled_features, np.arange(n_total), self.feature_cols)
            return model.predict_proba(all_features)[:, 1].astype(np.float32)

        for start in tqdmu(range(0, n_total, batch_size), desc="Batched prediction"):
            end = min(start + batch_size, n_total)
            batch_indices = np.arange(start, end)

            # Extract features for this batch only
            batch_features = extract_features_array(self.unlabeled_features, batch_indices, self.feature_cols)

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
            features = extract_features_array(self.unlabeled_features, [self._tracked_rowid_index], self.feature_cols)
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
        2. Sample negatives from unlabeled_features
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
        )

        val_flare_indices = np.array(val_flare_indices)
        held_out_flare_indices = np.array(held_out_flare_indices)

        logger.info(
            f"Flares split (stratified): train={len(train_flare_indices)}, "
            f"validation={len(val_flare_indices)}, held_out={len(held_out_flare_indices)}"
        )

        # Sample negatives from unlabeled_features for train, validation, and held-out
        n_unlabeled = len(self.unlabeled_features)
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

        logger.info(
            f"Negatives sampled: train={len(train_neg_indices)}, "
            f"validation={len(val_neg_indices)}, held_out={len(held_out_neg_indices)}"
        )

        # Create validation set (used for rollback/model selection decisions)
        self.validation = ValidationSet(
            flare_indices=val_flare_indices,
            negative_indices=val_neg_indices,
        )

        # Create held-out set (truly honest - NEVER used for any decisions)
        self.held_out = HeldOutSet(
            flare_indices=held_out_flare_indices,
            negative_indices=held_out_neg_indices,
        )

        # Add flares to labeled_train
        for idx in train_flare_indices:
            self._add_sample(LabeledSample(
                index=idx,
                label=1,
                weight=1.0,
                source=SampleSource.SEED,
                added_iter=0,
                confidence=1.0,
                consensus_score=1.0,
                from_known_flares=True,
            ))

        # Add negatives to labeled_train
        for idx in train_neg_indices:
            self._add_sample(LabeledSample(
                index=idx,
                label=0,
                weight=1.0,
                source=SampleSource.SEED,
                added_iter=0,
                confidence=1.0,
                consensus_score=1.0,
                from_known_flares=False,
            ))

        logger.info(f"Initial labeled_train size: {len(self.labeled_train)}")

        # Train initial model
        logger.info("Training initial model...")
        features, labels, weights = self._build_training_data()
        self.model = train_model(features, labels, weights, config=self.config, random_state=self.random_state)

        # Compute baseline validation metrics (used for decisions)
        validation_metrics = self._compute_validation_metrics(iteration=0)

        self.prev_validation_recall = validation_metrics["recall"]
        self.prev_validation_precision = validation_metrics["precision"]
        self.prev_validation_ice = validation_metrics.get("ice", 1.0)
        self.best_validation_recall = validation_metrics["recall"]
        self.best_validation_ice = validation_metrics.get("ice", 1.0)

        # Save initial checkpoint as best
        self.best_checkpoint = self._save_checkpoint(0, validation_metrics)

        # Log initial metrics
        counts = self._count_by_source()
        eff_pos, eff_neg = self._compute_effective_sizes()

        initial_metrics = IterationMetrics(
            iteration=0,
            train_total=len(self.labeled_train),
            train_seed=counts[SampleSource.SEED],
            train_pseudo_pos=counts[SampleSource.PSEUDO_POS],
            train_pseudo_neg=counts[SampleSource.PSEUDO_NEG],
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
            tracked_rowid_prob=self._get_tracked_rowid_probability(),
        )
        self.metrics_history.append(initial_metrics)

        logger.info("=" * 60)
        logger.info("ITERATION 0 COMPLETE")
        logger.info(f"  Train: {len(self.labeled_train)} samples")
        logger.info(
            f"  Validation: R={validation_metrics['recall']:.3f}, P={validation_metrics['precision']:.3f}, "
            f"AUC={validation_metrics.get('auc', 0):.3f}, PR-AUC={validation_metrics.get('pr_auc', 0):.3f}, "
            f"ICE={validation_metrics.get('ice', 0):.4f}, LL={validation_metrics.get('logloss', 0):.4f}, "
            f"Brier={validation_metrics.get('brier', 0):.4f}"
        )
        logger.info("=" * 60)

    # =========================================================================
    # Main Loop Phases
    # =========================================================================

    def _validate_and_check_stopping(self, iteration: int) -> tuple[dict, bool, str | None]:
        """
        Validate model and check for early stopping or rollback.

        Uses the validation set (not held-out) for all training decisions.
        The held-out set is only evaluated once at the end in _finalize().

        Returns:
            (validation_metrics, rollback_occurred, stop_reason)
        """
        validation_metrics = self._compute_validation_metrics(iteration=iteration)

        current_recall = validation_metrics["recall"]
        current_precision = validation_metrics["precision"]
        current_ice = validation_metrics.get("ice", 1.0)

        # Check for degradation using ICE (lower is better, so detect increases)
        ice_increased = current_ice > self.prev_validation_ice + self.config.ice_increase_threshold

        if ice_increased:
            logger.warning("DEGRADATION DETECTED (validation ICE increased)!")
            logger.warning(f"  ICE: {self.prev_validation_ice:.4f} -> {current_ice:.4f}")
            logger.warning(f"  (Recall: {self.prev_validation_recall:.3f} -> {current_recall:.3f})")
            logger.warning(f"  (Precision: {self.prev_validation_precision:.3f} -> {current_precision:.3f})")

            # Rollback
            if self.best_checkpoint is not None:
                logger.info(f"Rolling back to iteration {self.best_checkpoint.iteration}")
                self._load_checkpoint(self.best_checkpoint)

            # Tighten thresholds after rollback
            self._tighten_thresholds()

            self.n_successful_iters = 0
            self.rollback_history.append(iteration)

            # Recompute metrics after rollback
            validation_metrics = self._compute_validation_metrics()

            return validation_metrics, True, None  # rollback_occurred=True, skip rest of iteration

        # No degradation
        self.n_successful_iters += 1

        # Track best model by ICE (lower is better)
        if current_ice < self.best_validation_ice:
            self.best_validation_ice = current_ice
            self.best_validation_recall = current_recall
            self.best_checkpoint = self._save_checkpoint(iteration, validation_metrics)
            logger.info(f"New best model saved (validation ICE={current_ice:.4f}, recall={current_recall:.3f})")

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

        for seed in tqdmu(range(self.config.thresholds.n_bootstrap_models), desc="Bootstrap models"):
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
        """
        n_unlabeled = len(self.unlabeled_features)
        all_indices = np.arange(n_unlabeled)

        logger.info(f"Predicting on {n_unlabeled:,} samples (batched to avoid OOM)...")

        # Main model predictions
        main_preds = self._predict_unlabeled_features_batched(self.model)

        # Bootstrap model predictions
        bootstrap_preds = []
        for i, bm in enumerate(self.bootstrap_models):
            logger.info(f"Bootstrap model {i+1}/{len(self.bootstrap_models)}...")
            bp = self._predict_unlabeled_features_batched(bm)
            bootstrap_preds.append(bp)

        clean_ram()
        return main_preds, bootstrap_preds, all_indices

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
            Maps position in predictions to index in unlabeled_features
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
        exclusion_set = self._build_pseudo_label_exclusion_set()

        # Find candidates with low probability
        low_prob_positions = np.where(main_preds < self.pseudo_neg_threshold)[0]

        # Map to unlabeled_features indices and filter out exclusions
        neg_candidates = [(int(prediction_indices[pos]), pos) for pos in low_prob_positions if int(prediction_indices[pos]) not in exclusion_set]

        if not neg_candidates:
            return 0

        # Random subsample for efficiency
        thresholds = self.config.thresholds
        if len(neg_candidates) > thresholds.max_pseudo_neg_per_iter * 2:
            selected_indices = self.rng.choice(
                len(neg_candidates),
                size=thresholds.max_pseudo_neg_per_iter * 2,
                replace=False,
            )
            neg_candidates = [neg_candidates[i] for i in selected_indices]

        # Filter by bootstrap consensus (majority)
        confirmed = []
        for big_idx, pos in neg_candidates:
            main_prob = main_preds[pos]
            bootstrap_probs = [bp[pos] for bp in bootstrap_preds]

            # Majority of bootstrap models agree it's low
            n_low = sum(p < thresholds.neg_consensus_min_low_prob for p in bootstrap_probs)
            n_required = max(MIN_BOOTSTRAP_CONSENSUS_MODELS, thresholds.n_bootstrap_models // 2)
            if n_low >= n_required:
                avg_prob = (main_prob + np.mean(bootstrap_probs)) / 2
                # Consensus score: 1 - std gives range [0.5, 1.0] for probabilities
                # (max std for probabilities is ~0.5 when half are 0 and half are 1)
                consensus = compute_consensus_score(bootstrap_probs)
                confirmed.append(
                    {
                        "index": big_idx,
                        "confidence": 1 - avg_prob,
                        "consensus_score": consensus,
                    }
                )

        # Sort by confidence and take top
        confirmed.sort(key=lambda x: x["confidence"], reverse=True)
        confirmed = confirmed[: thresholds.max_pseudo_neg_per_iter]

        # Compute weight based on successful iterations
        weight = min(
            1.0,
            self.config.initial_pseudo_neg_weight + self.n_successful_iters * self.config.weight_increment,
        )

        # Add to training
        for item in confirmed:
            self._add_sample(LabeledSample(
                index=item["index"],
                label=0,
                weight=weight,
                source=SampleSource.PSEUDO_NEG,
                added_iter=iteration,
                confidence=item["confidence"],
                consensus_score=item["consensus_score"],
                from_known_flares=False,
            ))

        logger.info(f"Pseudo-negatives added: {len(confirmed)}, weight={weight:.2f}")
        return len(confirmed)

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
            Maps position in predictions to index in unlabeled_features
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
        exclusion_set = self._build_pseudo_label_exclusion_set()

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
        thresholds = self.config.thresholds
        confirmed = []
        for big_idx, pos in pos_candidates:
            main_prob = main_preds[pos]
            bootstrap_probs = np.array([bp[pos] for bp in bootstrap_preds])

            # All bootstrap models must be confident
            all_high = all(p > thresholds.consensus_threshold for p in bootstrap_probs)
            if not all_high:
                continue

            # Compute stats once
            bootstrap_std = bootstrap_probs.std()
            if bootstrap_std >= thresholds.bootstrap_variance_threshold:
                continue

            bootstrap_mean = bootstrap_probs.mean()
            avg_prob = (main_prob + bootstrap_mean) / 2
            consensus = 1.0 - bootstrap_std

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
            self.config.max_pseudo_pos_weight,
            self.config.initial_pseudo_pos_weight + self.n_successful_iters * self.config.weight_increment * 0.5,
        )  # Note: max_pseudo_pos_weight, initial_pseudo_pos_weight, weight_increment are in PipelineConfig directly

        # Add to training and plot
        for item in confirmed:
            sample_idx = item["index"]

            self._add_sample(LabeledSample(
                index=sample_idx,
                label=1,
                weight=weight,
                source=SampleSource.PSEUDO_POS,
                added_iter=iteration,
                confidence=item["confidence"],
                consensus_score=item["consensus_score"],
                from_known_flares=False,
            ))

            # Log sample ID and row number
            sample_id = self.unlabeled_features[sample_idx, "id"] if "id" in self.unlabeled_features.columns else sample_idx
            logger.info(f"  Added pseudo_pos: id={sample_id}, row={sample_idx}, conf={item['confidence']:.4f}")

            # Plot the sample
            plot_sample(sample_idx, self.unlabeled_features, self.output_dir, f"pseudo_pos_iter{iteration:03d}", self.config, action="added", dataset=self.unlabeled_features_dataset)

        logger.info(f"Pseudo-positives added: {len(confirmed)}, weight={weight:.2f}")
        return len(confirmed)

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
        dataset,
        source_name: str,
    ) -> tuple[list[int], list[SampleRemovalInfo]]:
        """
        Review a batch of samples from a single source for removal.

        Parameters
        ----------
        samples : list[tuple[int, LabeledSample]]
            List of (list_index, sample) tuples to review.
        source_df : pl.DataFrame
            Source DataFrame (known_flares or unlabeled_features).
        dataset : HuggingFace Dataset or None
            Dataset for plotting.
        source_name : str
            Name for logging ("oos" or "big").

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

        to_remove: list[int] = []
        to_plot: list[SampleRemovalInfo] = []

        for j, (i, sample) in enumerate(samples):
            current_prob = main_probs[j]
            bootstrap_probs = bootstrap_probs_all[:, j]

            should_remove, reason = self._check_sample_for_removal(sample, current_prob, bootstrap_probs)
            if should_remove:
                to_remove.append(i)
                to_plot.append(SampleRemovalInfo(
                    sample_index=sample.index,
                    source_df=source_df,
                    dataset=dataset,
                    sample=sample,
                ))
                logger.debug(f"Removing {sample.source} from {source_name}: {reason}")
            else:
                self._update_sample_weight(sample, current_prob, bootstrap_probs)

        return to_remove, to_plot

    def _apply_removals(self, to_remove: list[int], to_plot: list[SampleRemovalInfo]) -> None:
        """
        Plot removed samples and remove them from labeled_train.

        Parameters
        ----------
        to_remove : list[int]
            Indices in labeled_train to remove.
        to_plot : list[SampleRemovalInfo]
            Removal info for plotting.
        """
        # Plot removed samples
        for info in to_plot:
            sample_id = info.source_df[info.sample_index, "id"] if "id" in info.source_df.columns else info.sample_index
            logger.info(f"  Removed {info.sample.source}: id={sample_id}, row={info.sample_index}, original_conf={info.sample.confidence:.4f}")
            plot_sample(info.sample_index, info.source_df, self.output_dir, f"removed_{info.sample.source}", self.config, action="removed", dataset=info.dataset)

        # Remove in reverse order to preserve indices
        for i in sorted(to_remove, reverse=True):
            self._remove_sample(i)

    def _review_pseudo_labels(self) -> int:
        """
        Review and remove pseudo-labels that have become inconsistent.

        Uses batching by source for efficiency. Also reviews seed negatives
        if config.review_seed_negatives is True.

        Returns number of samples removed.
        """
        # Collect samples to review
        known_flare_samples, unlabeled_samples = self._collect_reviewable_samples()

        # Review each batch
        flare_remove, flare_plot = self._review_sample_batch(
            known_flare_samples, self.known_flares, self.known_flares_dataset, "known_flares"
        )
        unlabeled_remove, unlabeled_plot = self._review_sample_batch(
            unlabeled_samples, self.unlabeled_features, self.unlabeled_features_dataset, "unlabeled"
        )

        # Combine and apply removals
        to_remove = flare_remove + unlabeled_remove
        to_plot = flare_plot + unlabeled_plot
        self._apply_removals(to_remove, to_plot)

        logger.info(f"Review: removed {len(to_remove)} pseudo-labels (known_flares: {len(known_flare_samples)}, unlabeled: {len(unlabeled_samples)} reviewed)")
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
        thresholds = self.config.thresholds

        # Check pseudo-positives
        if sample.label == 1 and sample.source == SampleSource.PSEUDO_POS:
            # Model changed its mind?
            if current_prob < thresholds.pseudo_pos_removal_prob:
                return True, f"prob dropped: was {sample.confidence:.3f}, now {current_prob:.3f}"

            # Bootstrap diverged?
            bootstrap_std = np.std(bootstrap_probs)
            bootstrap_mean = np.mean(bootstrap_probs)
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
            bootstrap_mean = np.mean(bootstrap_probs)
            if current_prob > thresholds.seed_neg_removal_prob and bootstrap_mean > thresholds.seed_neg_removal_bootstrap_mean:
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
        if sample.label == 1 and sample.source == SampleSource.PSEUDO_POS:
            new_confidence = (current_prob + np.mean(bootstrap_probs)) / 2
            if sample.confidence > 0:
                sample.weight = min(1.0, sample.weight * (new_confidence / sample.confidence))

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

    def _retrain_model(self, class_weight: dict[int, float] | None) -> None:
        """Retrain main model with updated labeled data."""
        features, labels, weights = self._build_training_data()
        self.model = train_model(
            features,
            labels,
            weights,
            class_weight=class_weight,
            config=self.config,
            random_state=self.random_state + len(self.metrics_history),
        )

    def _adjust_thresholds(self) -> None:
        """Adjust pseudo-labeling thresholds based on training stability."""
        thresholds = self.config.thresholds

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
            logger.info("Computing enrichment (batched prediction)...")
            proba = self._predict_unlabeled_features_batched(self.model)
        else:
            proba = main_preds

        top_k = self.config.stopping.top_k_candidates
        top_k_indices = np.argsort(proba)[-top_k:]
        top_k_probs = proba[top_k_indices]

        # Estimated flares in top-K
        estimated_flares = float(np.sum(top_k_probs))

        # Random baseline at assumed prevalence
        random_baseline = top_k * self.config.assumed_prevalence

        enrichment = estimated_flares / random_baseline if random_baseline > 0 else 0.0

        return enrichment, estimated_flares

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

        # 1. Time-based stopping (primary criterion)
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
            recent_recalls = [m.validation_recall for m in self.metrics_history[-stopping.plateau_window:]]
            recent_ices = [m.validation_ice for m in self.metrics_history[-stopping.plateau_window:] if m.validation_ice > 0]

            recall_range = max(recent_recalls) - min(recent_recalls)
            ice_stable = len(recent_ices) >= 5 and (max(recent_ices) - min(recent_ices)) < stopping.plateau_ice_range

            if recall_range < stopping.plateau_recall_range and ice_stable:
                recent_pos_added = [m.pseudo_pos_added for m in self.metrics_history[-stopping.plateau_min_zero_pos_iters:]]
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

            # Validate and check for early stopping
            validation_metrics, rollback_occurred, stop = self._validate_and_check_stopping(iteration)
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

            # Clear predictions (no longer needed)
            del main_preds, bootstrap_preds, prediction_indices
            clean_ram()

            # Review and remove inconsistent pseudo-labels
            pseudo_removed = self._review_pseudo_labels()

            # Balance class weights if needed
            class_weight = self._balance_class_weights()

            # Retrain model with updated data
            logger.info("Retraining main model...")
            self._retrain_model(class_weight)

            # Adjust thresholds based on stability
            self._adjust_thresholds()

            # Compute enrichment (periodically to save time)
            if self.config.enrichment_every_n_iters > 0 and iteration % self.config.enrichment_every_n_iters == 0:
                enrichment, estimated_flares_top10k = self._compute_enrichment_factor()
            else:
                # Use last known enrichment or 0
                enrichment = self.metrics_history[-1].enrichment_factor if self.metrics_history else 0.0
                estimated_flares_top10k = self.metrics_history[-1].estimated_flares_top10k if self.metrics_history else 0.0

            # Compute OOB metrics for stability monitoring
            if self.bootstrap_indices_list:
                features, labels, _ = self._build_training_data()
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
                train_seed=counts[SampleSource.SEED],
                train_pseudo_pos=counts[SampleSource.PSEUDO_POS],
                train_pseudo_neg=counts[SampleSource.PSEUDO_NEG],
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
                tracked_rowid_prob=tracked_prob,
            )
            self.metrics_history.append(metrics)
            self._save_checkpoint(iteration, asdict(metrics))

            # Log summary
            logger.info("=" * 60)
            logger.info(f"ITERATION {iteration} COMPLETE")
            logger.info(f"  Train: {len(self.labeled_train)} (seed:{counts[SampleSource.SEED]}, pseudo_pos:{counts[SampleSource.PSEUDO_POS]}, pseudo_neg:{counts[SampleSource.PSEUDO_NEG]})")
            logger.info(
                f"  Validation: R={validation_metrics['recall']:.3f}, P={validation_metrics['precision']:.3f}, "
                f"AUC={validation_metrics.get('auc', 0):.3f}, PR-AUC={validation_metrics.get('pr_auc', 0):.3f}, "
                f"ICE={validation_metrics.get('ice', 0):.4f}, LL={validation_metrics.get('logloss', 0):.4f}, "
                f"Brier={validation_metrics.get('brier', 0):.4f}"
            )
            logger.info(f"  Enrichment: {enrichment:.1f}x")
            logger.info(f"  Elapsed: {self._get_elapsed_hours():.2f} hours")
            logger.info(f"  Successful iters: {self.n_successful_iters}")
            logger.info("=" * 60)

            # Update previous metrics
            self.prev_validation_recall = validation_metrics["recall"]
            self.prev_validation_precision = validation_metrics["precision"]
            self.prev_validation_ice = validation_metrics.get("ice", 1.0)

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
        self.clear_caches()

        logger.info("Generating candidate lists (batched prediction)...")
        final_proba = self._predict_unlabeled_features_batched(self.model)

        pool_with_proba = self.unlabeled_features.with_columns(pl.Series("proba", final_proba))

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
        logger.info(f"HONEST held-out recall: {honest_metrics['recall']:.3f}")
        logger.info(f"HONEST held-out precision: {honest_metrics['precision']:.3f}")
        logger.info(f"HONEST held-out ICE: {honest_metrics['ice']:.4f}")
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
        logger.info(
            f"HONEST HELD-OUT: R={honest_metrics['recall']:.3f}, P={honest_metrics['precision']:.3f}, "
            f"AUC={honest_metrics['auc']:.3f}, PR-AUC={honest_metrics['pr_auc']:.3f}, "
            f"ICE={honest_metrics['ice']:.4f}, LL={honest_metrics['logloss']:.4f}, "
            f"Brier={honest_metrics['brier']:.4f}"
        )

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
    unlabeled_features: pl.DataFrame,
    known_flares: pl.DataFrame,
    config: PipelineConfig | None = None,
    output_dir: Path | str = "active_learning_output",
    random_state: int = 42,
    unlabeled_features_dataset=None,
    known_flares_dataset=None,
) -> dict:
    """
    Run zero-expert self-training pipeline.

    Parameters
    ----------
    unlabeled_features : pl.DataFrame
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
    unlabeled_features_dataset : HuggingFace Dataset, optional
        Original light curve data for unlabeled_features, used for plotting.
        Access pattern: dataset["target"][row_index]
    known_flares_dataset : HuggingFace Dataset, optional
        Original light curve data for known_flares, used for plotting.
        Access pattern: dataset["target"][row_index]

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
        unlabeled_features=unlabeled_features,
        known_flares=known_flares,
        config=config,
        output_dir=output_dir,
        random_state=random_state,
        unlabeled_features_dataset=unlabeled_features_dataset,
        known_flares_dataset=known_flares_dataset,
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
    # unlabeled_features = pl.read_parquet("path/to/unlabeled_features.parquet")
    # known_flares = pl.read_parquet("path/to/known_flares.parquet")

    # Run pipeline
    # results = run_active_learning_pipeline(
    #     unlabeled_features=unlabeled_features,
    #     known_flares=known_flares,
    #     output_dir="active_learning_output",
    # )

    print("Pipeline module loaded. Use run_active_learning_pipeline() to start.")

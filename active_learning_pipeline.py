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

import gc
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import recall_score, precision_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions for Improved Pipeline
# =============================================================================


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
        rng = np.random.default_rng(random_state)
        shuffled = rng.permutation(n_samples)

        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        train_idx = shuffled[:n_train].tolist()
        val_idx = shuffled[n_train:n_train + n_val].tolist()
        held_out_idx = shuffled[n_train + n_val:].tolist()

        return train_idx, val_idx, held_out_idx

    # Build stratification labels
    try:
        strat_data = oos_features.select(stratify_cols).to_numpy()

        # Use quantile binning to create strata
        n_bins = min(5, n_samples // 10)  # At least 10 samples per bin
        if n_bins < 2:
            raise ValueError("Too few samples for stratification")

        binner = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        strata = binner.fit_transform(strat_data)

        # Combine multiple columns into single stratum label
        if strata.shape[1] > 1:
            strata_labels = (strata[:, 0] * n_bins + strata[:, 1]).astype(int)
        else:
            strata_labels = strata[:, 0].astype(int)

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

        logger.info(f"Stratified split: train={len(train_idx)}, val={len(val_idx)}, "
                    f"held_out={len(held_out_idx)}")

        return train_idx.tolist(), val_idx.tolist(), held_out_idx.tolist()

    except Exception as e:
        logger.warning(f"Stratified split failed ({e}), falling back to random")
        rng = np.random.default_rng(random_state)
        shuffled = rng.permutation(n_samples)

        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        train_idx = shuffled[:n_train].tolist()
        val_idx = shuffled[n_train:n_train + n_val].tolist()
        held_out_idx = shuffled[n_train + n_val:].tolist()

        return train_idx, val_idx, held_out_idx


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
    hardness = [(idx, i, abs(probas[i] - 0.5))
                for i, idx in enumerate(val_pool_indices)
                if probas[i] > 0.5]  # Only consider detected flares

    if len(hardness) == 0:
        return []

    # Sort by hardness (ascending = hardest first)
    hardness.sort(key=lambda x: x[2])

    # Select from different strata
    n = len(hardness)
    selected_positions = [0]  # Hardest

    if n >= 3:
        selected_positions.append(n // 3)      # Medium-hard
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
    bootstrap_models: list,
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
    oob_proba[valid_mask] = (
        oob_predictions[valid_mask].sum(axis=1) / oob_counts[valid_mask]
    )

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
    max_iters: int = 50
    n_bootstrap_models: int = 5  # Increased for better OOB coverage

    # Sample weights
    initial_pseudo_pos_weight: float = 0.2
    initial_pseudo_neg_weight: float = 0.8
    weight_increment: float = 0.1

    # Rollback criteria
    recall_drop_threshold: float = 0.05
    precision_drop_threshold: float = 0.1

    # Success targets
    target_recall: float = 0.75
    target_precision: float = 0.60
    target_enrichment: float = 50.0

    # Prevalence assumption (for enrichment calculation)
    assumed_prevalence: float = 0.001  # 0.1%

    # Prediction batch size for large datasets
    prediction_batch_size: int = 100_000

    # CatBoost parameters
    catboost_iterations: int = 500
    catboost_depth: int = 6
    catboost_learning_rate: float = 0.1
    catboost_verbose: bool = False


@dataclass
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


@dataclass
class Checkpoint:
    """Model checkpoint with associated metadata."""

    iteration: int
    model_path: Path
    labeled_train: list[LabeledSample]
    metrics: dict
    config_snapshot: dict


@dataclass
class HeldOutSet:
    """Held-out evaluation set."""

    flare_indices: np.ndarray  # Indices in oos_features
    negative_indices: np.ndarray  # Indices in big_features


@dataclass
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

    # Enrichment
    enrichment_factor: float
    estimated_flares_top10k: float

    # Thresholds (current)
    pseudo_pos_threshold: float
    pseudo_neg_threshold: float

    # Changes this iteration
    pseudo_pos_added: int
    pseudo_neg_added: int
    pseudo_removed: int
    val_pool_moved: int

    # Stability
    n_successful_iters: int
    n_rollbacks_recent: int

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
) -> CatBoostClassifier:
    """
    Train a CatBoost classifier.

    This is a stub implementation. Replace with your own model training logic.

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

    model = CatBoostClassifier(
        iterations=config.catboost_iterations,
        depth=config.catboost_depth,
        learning_rate=config.catboost_learning_rate,
        random_seed=random_state,
        verbose=config.catboost_verbose,
        auto_class_weights=None,  # We handle weights manually
        loss_function="Logloss",
        eval_metric="AUC",
    )

    model.fit(features, labels, sample_weight=sample_weights)
    return model


def evaluate_model(
    model: CatBoostClassifier,
    features: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Evaluate model on a labeled set.

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
        Dictionary with recall, precision, auc, f1 metrics.
    """
    proba = model.predict_proba(features)[:, 1]
    preds = (proba >= threshold).astype(int)

    # Handle edge cases for metrics
    if len(np.unique(labels)) < 2:
        auc = 0.5  # Can't compute AUC with single class
    else:
        auc = roc_auc_score(labels, proba)

    return {
        "recall": recall_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "auc": auc,
        "f1": f1_score(labels, preds, zero_division=0),
    }


def predict_proba_batched(
    model: CatBoostClassifier,
    features: np.ndarray,
    batch_size: int = 100_000,
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
    for i in tqdm(range(n_batches), desc=desc, disable=n_batches <= 1):
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
    return [c for c in df.columns if c not in exclude and df[c].dtype in (pl.Float32, pl.Float64, pl.Int64, pl.Int32)]


def extract_features_array(
    df: pl.DataFrame,
    indices: np.ndarray | list[int],
    feature_cols: list[str],
) -> np.ndarray:
    """Extract feature matrix for given indices."""
    subset = df[indices].select(feature_cols)
    return subset.to_numpy().astype(np.float32)


# =============================================================================
# Pipeline Implementation
# =============================================================================


class ActiveLearningPipeline:
    """Zero-Expert Self-Training Pipeline for Flare Detection."""

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
        self.prev_held_out_recall: float = 0.0
        self.prev_held_out_precision: float = 0.0
        self.prev_val_recall: float = 0.0
        self.n_successful_iters: int = 0
        self.rollback_history: list[int] = []  # Iteration numbers of rollbacks

        # Current thresholds (mutable)
        self.pseudo_pos_threshold = self.config.pseudo_pos_threshold
        self.pseudo_neg_threshold = self.config.pseudo_neg_threshold
        self.max_pseudo_pos_per_iter = self.config.max_pseudo_pos_per_iter

    def _validate_inputs(self) -> None:
        """Validate input DataFrames."""
        n_flares = len(self.oos_features)
        required_flares = (
            self.config.n_train_flares_init
            + self.config.n_val_pool
            + self.config.n_held_out_flares
        )
        if n_flares < required_flares:
            raise ValueError(
                f"Need at least {required_flares} known flares, got {n_flares}"
            )

        if len(self.big_features) < self.config.n_train_neg_init + self.config.n_held_out_neg:
            raise ValueError(
                f"big_features too small for required negative samples"
            )

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
        """Build feature matrix, labels, and weights from labeled_train."""
        n_samples = len(self.labeled_train)
        features_list = []
        labels = np.zeros(n_samples, dtype=np.int32)
        weights = np.zeros(n_samples, dtype=np.float32)

        for i, sample in enumerate(self.labeled_train):
            if sample.is_flare_source:
                row_features = extract_features_array(
                    self.oos_features, [sample.index], self.feature_cols
                )
            else:
                row_features = extract_features_array(
                    self.big_features, [sample.index], self.feature_cols
                )
            features_list.append(row_features[0])
            labels[i] = sample.label
            weights[i] = sample.weight

        features = np.array(features_list, dtype=np.float32)
        return features, labels, weights

    def _compute_val_metrics(self) -> float:
        """Compute recall on validation pool (all are flares)."""
        if not self.validation_pool or self.model is None:
            return 0.0

        features = extract_features_array(
            self.oos_features, self.validation_pool, self.feature_cols
        )
        proba = self.model.predict_proba(features)[:, 1]
        preds = (proba >= 0.5).astype(int)
        # All validation pool samples are flares (label=1)
        labels = np.ones(len(self.validation_pool), dtype=np.int32)
        return recall_score(labels, preds, zero_division=0)

    def _compute_held_out_metrics(self) -> dict[str, float]:
        """Compute metrics on held-out set."""
        if self.held_out is None or self.model is None:
            return {"recall": 0.0, "precision": 0.0, "auc": 0.5, "f1": 0.0}

        # Flares from oos_features
        flare_features = extract_features_array(
            self.oos_features, self.held_out.flare_indices, self.feature_cols
        )
        flare_proba = self.model.predict_proba(flare_features)[:, 1]
        flare_preds = (flare_proba >= 0.5).astype(int)
        flare_labels = np.ones(len(self.held_out.flare_indices), dtype=np.int32)

        # Negatives from big_features
        neg_features = extract_features_array(
            self.big_features, self.held_out.negative_indices, self.feature_cols
        )
        neg_proba = self.model.predict_proba(neg_features)[:, 1]
        neg_preds = (neg_proba >= 0.5).astype(int)
        neg_labels = np.zeros(len(self.held_out.negative_indices), dtype=np.int32)

        # Combined metrics
        all_proba = np.concatenate([flare_proba, neg_proba])
        all_preds = np.concatenate([flare_preds, neg_preds])
        all_labels = np.concatenate([flare_labels, neg_labels])

        return evaluate_model.__wrapped__(
            None, None, None
        ) if False else {  # Dummy to avoid calling model
            "recall": recall_score(all_labels, all_preds, zero_division=0),
            "precision": precision_score(all_labels, all_preds, zero_division=0),
            "auc": roc_auc_score(all_labels, all_proba) if len(np.unique(all_labels)) > 1 else 0.5,
            "f1": f1_score(all_labels, all_preds, zero_division=0),
        }

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

    # =========================================================================
    # Phase 0: Initialization
    # =========================================================================

    def initialize(self) -> None:
        """
        Phase 0: Initialize the pipeline.

        1. Split known flares into train/val/held-out
        2. Sample negatives from big_features
        3. Train initial model
        4. Compute baseline metrics
        """
        logger.info("=" * 60)
        logger.info("PHASE 0: INITIALIZATION")
        logger.info("=" * 60)

        # Shuffle known flares
        n_flares = len(self.oos_features)
        all_flare_indices = self.rng.permutation(n_flares)

        # Split flares
        train_end = self.config.n_train_flares_init
        val_end = train_end + self.config.n_val_pool
        held_out_end = val_end + self.config.n_held_out_flares

        train_flare_indices = all_flare_indices[:train_end].tolist()
        self.validation_pool = all_flare_indices[train_end:val_end].tolist()
        held_out_flare_indices = all_flare_indices[val_end:held_out_end]

        logger.info(f"Flares split: train={len(train_flare_indices)}, "
                    f"val_pool={len(self.validation_pool)}, "
                    f"held_out={len(held_out_flare_indices)}")

        # Sample negatives from big_features
        n_big = len(self.big_features)
        all_neg_indices = self.rng.permutation(n_big)

        train_neg_indices = all_neg_indices[:self.config.n_train_neg_init].tolist()
        held_out_neg_indices = all_neg_indices[
            self.config.n_train_neg_init:
            self.config.n_train_neg_init + self.config.n_held_out_neg
        ]

        logger.info(f"Negatives sampled: train={len(train_neg_indices)}, "
                    f"held_out={len(held_out_neg_indices)}")

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
        self.model = train_model(
            features, labels, weights, config=self.config, random_state=self.random_state
        )

        # Compute baseline metrics
        val_recall = self._compute_val_metrics()
        held_out_metrics = self._compute_held_out_metrics()

        self.prev_val_recall = val_recall
        self.prev_held_out_recall = held_out_metrics["recall"]
        self.prev_held_out_precision = held_out_metrics["precision"]
        self.best_held_out_recall = held_out_metrics["recall"]

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
        )
        self.metrics_history.append(initial_metrics)

        logger.info("=" * 60)
        logger.info("ITERATION 0 COMPLETE")
        logger.info(f"  Train: {len(self.labeled_train)} samples")
        logger.info(f"  Val recall: {val_recall:.3f}")
        logger.info(f"  Held-out: recall={held_out_metrics['recall']:.3f}, "
                    f"precision={held_out_metrics['precision']:.3f}")
        logger.info("=" * 60)

    # =========================================================================
    # Main Loop Phases
    # =========================================================================

    def _phase1_validation_and_early_stopping(
        self, iteration: int
    ) -> tuple[float, dict, bool, str | None]:
        """
        Phase 1: Validation and early stopping check.

        Returns:
            (val_recall, held_out_metrics, should_continue, stop_reason)
        """
        val_recall = self._compute_val_metrics()
        held_out_metrics = self._compute_held_out_metrics()

        current_recall = held_out_metrics["recall"]
        current_precision = held_out_metrics["precision"]

        # Check for degradation
        recall_dropped = current_recall < self.prev_held_out_recall - self.config.recall_drop_threshold
        precision_dropped = current_precision < self.prev_held_out_precision - self.config.precision_drop_threshold

        if recall_dropped or precision_dropped:
            logger.warning("DEGRADATION DETECTED!")
            logger.warning(f"  Recall: {self.prev_held_out_recall:.3f} -> {current_recall:.3f}")
            logger.warning(f"  Precision: {self.prev_held_out_precision:.3f} -> {current_precision:.3f}")

            # Rollback
            if self.best_checkpoint is not None:
                logger.info(f"Rolling back to iteration {self.best_checkpoint.iteration}")
                self._load_checkpoint(self.best_checkpoint)

            # Tighten thresholds
            self.pseudo_pos_threshold = min(0.999, self.pseudo_pos_threshold + 0.005)
            self.pseudo_neg_threshold = max(0.01, self.pseudo_neg_threshold - 0.01)
            self.max_pseudo_pos_per_iter = max(3, self.max_pseudo_pos_per_iter - 2)

            logger.info(f"  New thresholds: pos>{self.pseudo_pos_threshold:.3f}, "
                        f"neg<{self.pseudo_neg_threshold:.3f}")

            self.n_successful_iters = 0
            self.rollback_history.append(iteration)

            # Recompute metrics after rollback
            val_recall = self._compute_val_metrics()
            held_out_metrics = self._compute_held_out_metrics()

            return val_recall, held_out_metrics, True, None  # Continue but skip this iteration

        # No degradation
        self.n_successful_iters += 1

        if current_recall > self.best_held_out_recall:
            self.best_held_out_recall = current_recall
            self.best_checkpoint = self._save_checkpoint(iteration, held_out_metrics)
            logger.info(f"New best model saved (recall={current_recall:.3f})")

        # Check for success
        enrichment, _ = self._compute_enrichment_factor() if iteration > 0 else (0.0, 0.0)
        if (
            current_recall >= self.config.target_recall
            and current_precision >= self.config.target_precision
            and enrichment >= self.config.target_enrichment
            and self.n_successful_iters >= 5
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
        features = extract_features_array(
            self.oos_features, self.validation_pool, self.feature_cols
        )
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

        logger.info(f"Hard example from val_pool: P={hardest_prob:.3f}, "
                    f"val_pool remaining: {len(self.validation_pool)}")
        return 1

    def _phase3_train_bootstrap_models(
        self, iteration: int
    ) -> tuple[list[CatBoostClassifier], list[np.ndarray]]:
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

        for seed in range(self.config.n_bootstrap_models):
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
        Phase 4: Prediction on all big_features (excluding held-out negatives).

        To prevent data leakage, we exclude held_out_neg indices from predictions.
        These samples are reserved for evaluation only and should never influence
        the pseudo-labeling process.

        Returns
        -------
        tuple[np.ndarray, list[np.ndarray], np.ndarray]
            (main_predictions, bootstrap_predictions, prediction_indices)
            prediction_indices maps position in predictions to index in big_features
        """
        logger.info("Predicting on full dataset (excluding held-out negatives)...")

        # Create mask excluding held-out negatives
        n_big = len(self.big_features)
        held_out_set = set(self.held_out.negative_indices.tolist()) if self.held_out else set()
        prediction_indices = np.array([i for i in range(n_big) if i not in held_out_set])

        logger.info(f"Predicting on {len(prediction_indices):,} samples "
                    f"(excluded {len(held_out_set):,} held-out negatives)")

        features_subset = extract_features_array(
            self.big_features,
            prediction_indices,
            self.feature_cols,
        )

        main_preds = predict_proba_batched(
            self.model,
            features_subset,
            batch_size=self.config.prediction_batch_size,
            desc="Main model",
        )

        bootstrap_preds = []
        for i, bm in enumerate(self.bootstrap_models):
            bp = predict_proba_batched(
                bm,
                features_subset,
                batch_size=self.config.prediction_batch_size,
                desc=f"Bootstrap {i+1}",
            )
            bootstrap_preds.append(bp)

        del features_subset
        gc.collect()

        return main_preds, bootstrap_preds, prediction_indices

    def _phase5_pseudo_label_negatives(
        self,
        main_preds: np.ndarray,
        bootstrap_preds: list[np.ndarray],
        prediction_indices: np.ndarray,
        iteration: int,
    ) -> int:
        """
        Phase 5: Aggressive pseudo-labeling of negatives.

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
        """
        labeled_big_indices, _ = self._get_labeled_indices()

        # Create reverse mapping: big_features_idx -> position in predictions
        big_idx_to_pos = {int(big_idx): pos for pos, big_idx in enumerate(prediction_indices)}

        # Find candidates with low probability (using prediction array positions)
        low_prob_positions = np.where(main_preds < self.pseudo_neg_threshold)[0]

        # Map to big_features indices and filter out already labeled
        # Store as (big_idx, position) tuples for later lookup
        neg_candidates = [
            (int(prediction_indices[pos]), pos) for pos in low_prob_positions
            if int(prediction_indices[pos]) not in labeled_big_indices
        ]

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
            n_low = sum(p < 0.1 for p in bootstrap_probs)
            n_required = max(2, self.config.n_bootstrap_models // 2)  # Majority
            if n_low >= n_required:
                avg_prob = (main_prob + np.mean(bootstrap_probs)) / 2
                consensus = 1 - np.std(bootstrap_probs)
                confirmed.append({
                    "index": big_idx,
                    "confidence": 1 - avg_prob,
                    "consensus_score": consensus,
                })

        # Sort by confidence and take top
        confirmed.sort(key=lambda x: x["confidence"], reverse=True)
        confirmed = confirmed[:self.config.max_pseudo_neg_per_iter]

        # Compute weight based on successful iterations
        weight = min(
            1.0,
            self.config.initial_pseudo_neg_weight
            + self.n_successful_iters * self.config.weight_increment,
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
        iteration: int,
    ) -> int:
        """
        Phase 6: Conservative pseudo-labeling of positives.

        Returns number of positives added.
        """
        labeled_big_indices, _ = self._get_labeled_indices()
        held_out_set = set(self.held_out.negative_indices.tolist()) if self.held_out else set()

        # Find candidates with high probability
        pos_candidates = np.where(main_preds > self.pseudo_pos_threshold)[0]
        pos_candidates = [
            idx for idx in pos_candidates
            if idx not in labeled_big_indices and idx not in held_out_set
        ]

        if not pos_candidates:
            return 0

        # Take top-K by probability
        pos_candidates = sorted(
            pos_candidates,
            key=lambda x: main_preds[x],
            reverse=True,
        )[:self.max_pseudo_pos_per_iter * 5]

        # Strict filtering: ALL bootstrap models must agree
        confirmed = []
        for idx in pos_candidates:
            main_prob = main_preds[idx]
            bootstrap_probs = [bp[idx] for bp in bootstrap_preds]

            # All bootstrap models must be confident
            all_high = all(p > self.config.consensus_threshold for p in bootstrap_probs)
            low_variance = np.std(bootstrap_probs) < 0.05

            if all_high and low_variance:
                avg_prob = (main_prob + np.mean(bootstrap_probs)) / 2
                consensus = 1 - np.std(bootstrap_probs)
                confirmed.append({
                    "index": idx,
                    "confidence": avg_prob,
                    "consensus_score": consensus,
                })

        # Sort by consensus and confidence
        confirmed.sort(key=lambda x: (x["consensus_score"], x["confidence"]), reverse=True)
        confirmed = confirmed[:self.max_pseudo_pos_per_iter]

        # Compute weight (lower than negatives)
        weight = min(
            0.8,
            self.config.initial_pseudo_pos_weight
            + self.n_successful_iters * self.config.weight_increment * 0.5,
        )

        # Add to training
        for item in confirmed:
            self.labeled_train.append(
                LabeledSample(
                    index=item["index"],
                    label=1,
                    weight=weight,
                    source="pseudo_pos",
                    added_iter=iteration,
                    confidence=item["confidence"],
                    consensus_score=item["consensus_score"],
                    is_flare_source=False,
                )
            )

        logger.info(f"Pseudo-positives added: {len(confirmed)}, weight={weight:.2f}")
        return len(confirmed)

    def _phase7_review_pseudo_labels(self) -> int:
        """
        Phase 7: Review and remove inconsistent pseudo-labels.

        Returns number of samples removed.
        """
        to_remove = []

        for i, sample in enumerate(self.labeled_train):
            # Skip seed samples (ground truth)
            if sample.source == "seed":
                continue

            # Get current prediction
            if sample.is_flare_source:
                features = extract_features_array(
                    self.oos_features, [sample.index], self.feature_cols
                )
            else:
                features = extract_features_array(
                    self.big_features, [sample.index], self.feature_cols
                )

            current_prob = self.model.predict_proba(features)[0, 1]
            bootstrap_probs = [
                bm.predict_proba(features)[0, 1] for bm in self.bootstrap_models
            ]

            # Check pseudo-positives
            if sample.label == 1 and sample.source in ["pseudo_pos", "val_pool"]:
                # Model changed its mind?
                if current_prob < 0.3:
                    to_remove.append(i)
                    logger.debug(f"Removing pseudo-pos: iter {sample.added_iter}, "
                                 f"was P={sample.confidence:.3f}, now P={current_prob:.3f}")
                    continue

                # Bootstrap diverged?
                if np.std(bootstrap_probs) > 0.2 and np.mean(bootstrap_probs) < 0.7:
                    to_remove.append(i)
                    logger.debug(f"Removing pseudo-pos: unstable, std={np.std(bootstrap_probs):.3f}")
                    continue

                # Update weight based on current confidence
                new_confidence = (current_prob + np.mean(bootstrap_probs)) / 2
                if sample.confidence > 0:
                    sample.weight = min(1.0, sample.weight * (new_confidence / sample.confidence))

            # Check pseudo-negatives
            elif sample.label == 0 and sample.source == "pseudo_neg":
                # Model changed its mind? (less strict)
                if current_prob > 0.7:
                    to_remove.append(i)
                    logger.debug(f"Removing pseudo-neg: iter {sample.added_iter}, "
                                 f"was P={1-sample.confidence:.3f}, now P={current_prob:.3f}")
                    continue

        # Remove in reverse order to preserve indices
        for i in sorted(to_remove, reverse=True):
            self.labeled_train.pop(i)

        logger.info(f"Review: removed {len(to_remove)} pseudo-labels")
        return len(to_remove)

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
        if self.n_successful_iters >= 3:
            # Model stable - relax thresholds
            self.pseudo_pos_threshold = max(0.95, self.pseudo_pos_threshold - 0.005)
            self.pseudo_neg_threshold = min(0.10, self.pseudo_neg_threshold + 0.01)
            self.max_pseudo_pos_per_iter = min(20, self.max_pseudo_pos_per_iter + 1)
            logger.info(f"Thresholds relaxed: pos>{self.pseudo_pos_threshold:.3f}, "
                        f"neg<{self.pseudo_neg_threshold:.3f}")

        elif self.n_successful_iters == 0:
            # Just rolled back - tighten
            self.pseudo_pos_threshold = min(0.999, self.pseudo_pos_threshold + 0.01)
            self.pseudo_neg_threshold = max(0.01, self.pseudo_neg_threshold - 0.02)
            self.max_pseudo_pos_per_iter = max(3, self.max_pseudo_pos_per_iter - 3)
            logger.info(f"Thresholds tightened: pos>{self.pseudo_pos_threshold:.3f}, "
                        f"neg<{self.pseudo_neg_threshold:.3f}")

    def _compute_enrichment_factor(
        self, main_preds: np.ndarray | None = None
    ) -> tuple[float, float]:
        """
        Phase 11: Compute enrichment factor.

        Parameters
        ----------
        main_preds : np.ndarray, optional
            Precomputed predictions. If None, will compute them.

        Returns
        -------
        tuple[float, float]
            (enrichment_factor, estimated_flares_in_top10k)
        """
        if self.model is None:
            return 0.0, 0.0

        # Use precomputed predictions or compute them
        if main_preds is None:
            features_full = extract_features_array(
                self.big_features,
                np.arange(len(self.big_features)),
                self.feature_cols,
            )
            proba = predict_proba_batched(
                self.model,
                features_full,
                batch_size=self.config.prediction_batch_size,
                desc="Enrichment calc",
            )
            del features_full
            gc.collect()
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
        """Check stopping criteria. Returns reason string or None to continue."""
        # 1. Plateau check
        if len(self.metrics_history) >= 10:
            recent_recalls = [m.held_out_recall for m in self.metrics_history[-10:]]
            recall_range = max(recent_recalls) - min(recent_recalls)
            if recall_range < 0.01:
                recent_pos_added = [m.pseudo_pos_added for m in self.metrics_history[-5:]]
                if sum(recent_pos_added) == 0:
                    return "PLATEAU"

        # 2. Data exhaustion
        if len(self.validation_pool) <= 3:
            recent_pos_added = [m.pseudo_pos_added for m in self.metrics_history[-3:]]
            if sum(recent_pos_added) == 0:
                return "DATA_EXHAUSTED"

        # 3. Max iterations
        if iteration >= self.config.max_iters:
            return "MAX_ITERATIONS"

        # 4. Instability
        recent_rollbacks = sum(
            1 for r in self.rollback_history
            if r > iteration - 10
        )
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

        for iteration in range(1, self.config.max_iters + 1):
            logger.info("=" * 60)
            logger.info(f"ITERATION {iteration}")
            logger.info("=" * 60)

            # Phase 1: Validation and early stopping
            val_recall, held_out_metrics, should_continue, phase1_stop = (
                self._phase1_validation_and_early_stopping(iteration)
            )
            if phase1_stop:
                stop_reason = phase1_stop
                break
            if not should_continue:
                continue  # Skip rest of iteration (rollback happened)

            # Phase 2: Hard example mining
            val_pool_moved = self._phase2_hard_example_mining(val_recall)

            # Phase 3: Bootstrap models
            logger.info("Training bootstrap models...")
            self._phase3_train_bootstrap_models(iteration)

            # Phase 4: Predict on all data
            main_preds, bootstrap_preds = self._phase4_predict_all()

            # Phase 5: Pseudo-label negatives
            pseudo_neg_added = self._phase5_pseudo_label_negatives(
                main_preds, bootstrap_preds, iteration
            )

            # Phase 6: Pseudo-label positives
            pseudo_pos_added = self._phase6_pseudo_label_positives(
                main_preds, bootstrap_preds, iteration
            )

            # Phase 7: Review old pseudo-labels
            pseudo_removed = self._phase7_review_pseudo_labels()

            # Phase 8: Balance weights
            class_weight = self._phase8_balance_weights()

            # Phase 9: Retrain model
            logger.info("Retraining main model...")
            self._phase9_retrain_model(class_weight)

            # Phase 10: Adaptive thresholds
            self._phase10_adaptive_thresholds()

            # Phase 11: Enrichment (using phase 4 predictions)
            enrichment, estimated_flares_top10k = self._compute_enrichment_factor(main_preds)

            # Clear large arrays
            del main_preds, bootstrap_preds
            gc.collect()

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
            )
            self.metrics_history.append(metrics)
            self._save_checkpoint(iteration, asdict(metrics))

            # Log summary
            logger.info("=" * 60)
            logger.info(f"ITERATION {iteration} COMPLETE")
            logger.info(f"  Train: {len(self.labeled_train)} "
                        f"(seed:{counts['seed']}, pseudo_pos:{counts['pseudo_pos']}, "
                        f"pseudo_neg:{counts['pseudo_neg']})")
            logger.info(f"  Val recall: {val_recall:.3f}")
            logger.info(f"  Held-out: recall={held_out_metrics['recall']:.3f}, "
                        f"precision={held_out_metrics['precision']:.3f}")
            logger.info(f"  Enrichment: {enrichment:.1f}x")
            logger.info(f"  Successful iters: {self.n_successful_iters}")
            logger.info("=" * 60)

            # Update previous metrics
            self.prev_val_recall = val_recall
            self.prev_held_out_recall = held_out_metrics["recall"]
            self.prev_held_out_precision = held_out_metrics["precision"]

            # Check stopping criteria
            stop_reason = self._check_stopping_criteria(
                iteration, held_out_metrics, pseudo_pos_added
            )
            if stop_reason:
                break

        # Finalization
        return self._finalize(stop_reason or "MAX_ITERATIONS")

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

        # Generate candidate lists
        logger.info("Generating candidate lists...")
        features_full = extract_features_array(
            self.big_features,
            np.arange(len(self.big_features)),
            self.feature_cols,
        )
        final_proba = predict_proba_batched(
            self.model,
            features_full,
            batch_size=self.config.prediction_batch_size,
            desc="Final predictions",
        )

        # Add proba column to big_features for filtering
        big_with_proba = self.big_features.with_columns(
            pl.Series("proba", final_proba)
        )

        candidates = {
            "high_purity": big_with_proba.filter(pl.col("proba") > 0.95),
            "balanced": big_with_proba.filter(pl.col("proba") > 0.80),
            "high_recall": big_with_proba.filter(pl.col("proba") > 0.50),
        }

        del big_with_proba, features_full, final_proba
        gc.collect()

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

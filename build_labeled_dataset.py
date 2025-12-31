"""
Build a fully labeled feature dataset from multiple label sources.

Combines labels from:
- known_flares (positive)
- freaky_held_out_indices (positive)
- forced_positive_indices (positive)
- forced_negative_indices (negative)
- expert_labels_file (positive and negative)
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl

from active_learning_pipeline import _load_expert_labels_file

logger = logging.getLogger(__name__)


def build_labeled_dataset(
    unlabeled_samples: pl.DataFrame,
    known_flares: pl.DataFrame,
    unlabeled_dataset=None,
    known_flares_dataset=None,
    freaky_held_out_indices: list[int] | None = None,
    forced_positive_indices: list[int] | None = None,
    forced_negative_indices: list[int] | None = None,
    expert_labels_file: str | Path | None = None,
) -> pl.DataFrame:
    """
    Build a fully labeled feature dataset from multiple label sources.

    Parameters
    ----------
    unlabeled_samples : pl.DataFrame
        Feature DataFrame for unlabeled samples (has row indices)
    known_flares : pl.DataFrame
        Feature DataFrame for known flares (all positive)
    unlabeled_dataset : optional
        Raw dataset (not used for features, only for reference)
    known_flares_dataset : optional
        Raw dataset for known flares (not used for features)
    freaky_held_out_indices : list[int], optional
        Indices in unlabeled_samples to treat as positive
    forced_positive_indices : list[int], optional
        Indices in unlabeled_samples to treat as positive
    forced_negative_indices : list[int], optional
        Indices in unlabeled_samples to treat as negative
    expert_labels_file : str or Path, optional
        Path to JSONL file with expert labels

    Returns
    -------
    pl.DataFrame
        Combined dataset with "class", "source", "_dataset" (0=unlabeled, 1=known_flares),
        and "_orig_idx" (original row index in source dataset) columns
    """
    labeled_rows = []

    # 1. Known flares - all positive
    if known_flares is not None and len(known_flares) > 0:
        known_with_class = known_flares.with_row_index("_orig_idx").with_columns([
            pl.lit(1).alias("class"),
            pl.lit("known_flare").alias("source"),
            pl.lit(1).alias("_dataset"),  # 1 = known_flares
        ])
        labeled_rows.append(known_with_class)
        logger.info(f"Added {len(known_flares)} known flares (positive)")

    # Collect all labels for unlabeled_samples
    pos_indices: set[int] = set()
    neg_indices: set[int] = set()

    # 2. Freaky held-out indices (positive)
    if freaky_held_out_indices:
        pos_indices.update(freaky_held_out_indices)
        logger.info(f"Added {len(freaky_held_out_indices)} freaky held-out indices (positive)")

    # 3. Forced positive indices
    if forced_positive_indices:
        pos_indices.update(forced_positive_indices)
        logger.info(f"Added {len(forced_positive_indices)} forced positive indices")

    # 4. Forced negative indices
    if forced_negative_indices:
        neg_indices.update(forced_negative_indices)
        logger.info(f"Added {len(forced_negative_indices)} forced negative indices")

    # 5. Expert labels (reuse existing function)
    if expert_labels_file:
        expert_pos, expert_neg = _load_expert_labels_file(str(expert_labels_file))
        if expert_pos:
            pos_indices.update(expert_pos)
            logger.info(f"Added {len(expert_pos)} expert positive labels")
        if expert_neg:
            neg_indices.update(expert_neg)
            logger.info(f"Added {len(expert_neg)} expert negative labels")

    # Check for conflicts between pos and neg
    conflicts = pos_indices & neg_indices
    if conflicts:
        logger.warning(f"Found {len(conflicts)} conflicting indices (in both pos and neg). Treating as positive.")
        neg_indices -= conflicts

    # Extract rows from unlabeled_samples
    # Assuming unlabeled_samples has a row index or we use positional index
    if "row_index" in unlabeled_samples.columns:
        index_col = "row_index"
        unlabeled_with_idx = unlabeled_samples.with_columns(
            pl.col(index_col).alias("_orig_idx")
        )
    elif "index" in unlabeled_samples.columns:
        index_col = "index"
        unlabeled_with_idx = unlabeled_samples.with_columns(
            pl.col(index_col).alias("_orig_idx")
        )
    else:
        # Add row index
        unlabeled_with_idx = unlabeled_samples.with_row_index("_orig_idx")
        index_col = "_orig_idx"

    # Get positive samples from unlabeled
    if pos_indices:
        pos_mask = unlabeled_with_idx[index_col].is_in(list(pos_indices))
        pos_rows = unlabeled_with_idx.filter(pos_mask).with_columns([
            pl.lit(1).alias("class"),
            pl.lit("unlabeled_pos").alias("source"),
            pl.lit(0).alias("_dataset"),  # 0 = unlabeled_samples
        ])
        if len(pos_rows) > 0:
            labeled_rows.append(pos_rows)
            logger.info(f"Extracted {len(pos_rows)} positive samples from unlabeled_samples")

    # Get negative samples from unlabeled
    if neg_indices:
        neg_mask = unlabeled_with_idx[index_col].is_in(list(neg_indices))
        neg_rows = unlabeled_with_idx.filter(neg_mask).with_columns([
            pl.lit(0).alias("class"),
            pl.lit("unlabeled_neg").alias("source"),
            pl.lit(0).alias("_dataset"),  # 0 = unlabeled_samples
        ])
        if len(neg_rows) > 0:
            labeled_rows.append(neg_rows)
            logger.info(f"Extracted {len(neg_rows)} negative samples from unlabeled_samples")

    # Combine all
    if not labeled_rows:
        logger.warning("No labeled samples found!")
        return pl.DataFrame()

    result = pl.concat(labeled_rows, how="diagonal")

    pos_count = result["class"].sum()
    neg_count = len(result) - pos_count
    logger.info(f"Built labeled dataset: {len(result)} samples ({pos_count} positive, {neg_count} negative)")

    return result


def _extract_sequence_from_hf_dataset(
    dataset,
    idx: int,
    sequence_cols: tuple[str, ...] = ("mjd", "mag", "magerr", "norm"),
) -> np.ndarray:
    """
    Extract a single sequence from a HuggingFace Dataset.

    Parameters
    ----------
    dataset : HuggingFace Dataset
        Dataset with columns like mjd, mag, magerr, norm (each is a list)
    idx : int
        Row index in the dataset
    sequence_cols : tuple[str, ...]
        Column names to stack into sequence

    Returns
    -------
    np.ndarray
        (seq_len, n_features) array
    """
    row = dataset[idx]
    # Each column is a list of values for the light curve
    arrays = [np.array(row[col], dtype=np.float32) for col in sequence_cols]
    return np.column_stack(arrays)


def prepare_recurrent_training_data(
    labeled_df: pl.DataFrame,
    unlabeled_dataset=None,
    known_flares_dataset=None,
    feature_cols: list[str] | None = None,
    sequence_cols: tuple[str, ...] = ("mjd", "mag", "magerr", "norm"),
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray] | None]:
    """
    Prepare training data for RecurrentClassifierWrapper from labeled_df.

    Parameters
    ----------
    labeled_df : pl.DataFrame
        Output from build_labeled_dataset()
    unlabeled_dataset : HuggingFace Dataset, optional
        Raw dataset with sequence columns (mjd, mag, etc.)
    known_flares_dataset : HuggingFace Dataset, optional
        Raw dataset for known flares with sequence columns
    feature_cols : list[str], optional
        Feature columns to use. If None, auto-detects.
    sequence_cols : tuple[str, ...], optional
        Columns to extract for sequences

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[np.ndarray] | None]
        (features, labels, sequences)
        - features: (n_samples, n_features) array
        - labels: (n_samples,) array
        - sequences: list of (seq_len, n_seq_features) arrays, or None if datasets not provided
    """
    # Auto-detect feature columns
    exclude_cols = {"class", "source", "_dataset", "_orig_idx", "row_index", "index", "_row_idx"}
    exclude_cols.update(sequence_cols)  # Exclude sequence columns if present

    if feature_cols is None:
        feature_cols = [c for c in labeled_df.columns if c not in exclude_cols]

    # Extract features and labels
    features = labeled_df.select(feature_cols).to_numpy().astype(np.float32)
    labels = labeled_df["class"].to_numpy().astype(np.int64)

    logger.info(f"Prepared features: {features.shape}, labels: {labels.shape}")

    # Extract sequences if datasets provided
    sequences = None
    if "_dataset" in labeled_df.columns and "_orig_idx" in labeled_df.columns:
        has_unlabeled = unlabeled_dataset is not None
        has_known = known_flares_dataset is not None

        if has_unlabeled:
            sequences = []
            dataset_flags = labeled_df["_dataset"].to_numpy()
            orig_indices = labeled_df["_orig_idx"].to_numpy()

            for ds_flag, orig_idx in zip(dataset_flags, orig_indices):
                if ds_flag == 0:  # unlabeled_samples
                    seq = _extract_sequence_from_hf_dataset(unlabeled_dataset, int(orig_idx), sequence_cols)
                elif ds_flag == 1 and has_known:  # known_flares
                    seq = _extract_sequence_from_hf_dataset(known_flares_dataset, int(orig_idx), sequence_cols)
                else:
                    # Fallback: create dummy sequence
                    seq = np.zeros((1, len(sequence_cols)), dtype=np.float32)
                sequences.append(seq)

            logger.info(f"Extracted {len(sequences)} sequences")

    return features, labels, sequences


def train_recurrent_classifier(
    labeled_df: pl.DataFrame,
    unlabeled_dataset=None,
    known_flares_dataset=None,
    input_mode: str = "features",  # "features", "sequence", "hybrid"
    feature_cols: list[str] | None = None,
    val_fraction: float = 0.1,
    random_state: int = 42,
    **config_kwargs,
):
    """
    Convenience function to train RecurrentClassifierWrapper from labeled_df.

    Parameters
    ----------
    labeled_df : pl.DataFrame
        Output from build_labeled_dataset()
    unlabeled_dataset : HuggingFace Dataset, optional
        Raw dataset with sequence columns (mjd, mag, etc.)
    known_flares_dataset : HuggingFace Dataset, optional
        Raw dataset for known flares
    input_mode : str
        One of "features", "sequence", "hybrid"
    feature_cols : list[str], optional
        Feature columns to use
    val_fraction : float
        Fraction of data for validation
    random_state : int
        Random seed
    **config_kwargs
        Additional RecurrentConfig parameters

    Returns
    -------
    RecurrentClassifierWrapper
        Trained classifier
    """
    from recurrent_classifier import RecurrentClassifierWrapper, RecurrentConfig, InputMode

    # Map string to InputMode
    mode_map = {
        "features": InputMode.FEATURES_ONLY,
        "sequence": InputMode.SEQUENCE_ONLY,
        "hybrid": InputMode.HYBRID,
    }
    input_mode_enum = mode_map.get(input_mode, InputMode.FEATURES_ONLY)

    # Prepare data
    need_sequences = input_mode_enum in (InputMode.SEQUENCE_ONLY, InputMode.HYBRID)
    features, labels, sequences = prepare_recurrent_training_data(
        labeled_df,
        unlabeled_dataset if need_sequences else None,
        known_flares_dataset if need_sequences else None,
        feature_cols=feature_cols,
    )

    # Train/val split
    np.random.seed(random_state)
    n_samples = len(labels)
    indices = np.random.permutation(n_samples)
    n_val = int(n_samples * val_fraction)

    train_idx = indices[n_val:]
    val_idx = indices[:n_val]

    X_train = features[train_idx]
    y_train = labels[train_idx]
    X_val = features[val_idx]
    y_val = labels[val_idx]

    seq_train = [sequences[i] for i in train_idx] if sequences else None
    seq_val = [sequences[i] for i in val_idx] if sequences else None

    # Configure and train
    config = RecurrentConfig(input_mode=input_mode_enum, **config_kwargs)
    wrapper = RecurrentClassifierWrapper(config, random_state=random_state)

    # Prepare eval_set based on mode
    if input_mode_enum == InputMode.FEATURES_ONLY:
        eval_set = (X_val, y_val)
    else:
        eval_set = (seq_val, X_val, y_val)

    wrapper.fit(
        features=X_train,
        labels=y_train,
        sequences=seq_train,
        eval_set=eval_set,
    )

    return wrapper


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    print("Usage:")
    print()
    print("# 1. Build labeled dataset")
    print("from build_labeled_dataset import build_labeled_dataset, train_recurrent_classifier")
    print()
    print("labeled_df = build_labeled_dataset(")
    print("    unlabeled_samples=unlabeled_samples,")
    print("    known_flares=known_flares,")
    print("    freaky_held_out_indices=[...],")
    print("    expert_labels_file='expert_labels.txt',")
    print(")")
    print()
    print("# 2a. Train FEATURES_ONLY classifier (CatBoost drop-in)")
    print("wrapper = train_recurrent_classifier(")
    print("    labeled_df, unlabeled_dataset=None, input_mode='features'")
    print(")")
    print()
    print("# 2b. Train HYBRID classifier (features + sequences)")
    print("wrapper = train_recurrent_classifier(")
    print("    labeled_df, unlabeled_dataset, known_flares_dataset, input_mode='hybrid'")
    print(")")

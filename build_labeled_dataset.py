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
        Combined dataset with "class" column (0 or 1) and "source" column
    """
    labeled_rows = []

    # 1. Known flares - all positive
    if known_flares is not None and len(known_flares) > 0:
        known_with_class = known_flares.with_columns([
            pl.lit(1).alias("class"),
            pl.lit("known_flare").alias("source"),
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
    elif "index" in unlabeled_samples.columns:
        index_col = "index"
    else:
        # Add row index
        unlabeled_samples = unlabeled_samples.with_row_index("_row_idx")
        index_col = "_row_idx"

    # Get positive samples from unlabeled
    if pos_indices:
        pos_mask = unlabeled_samples[index_col].is_in(list(pos_indices))
        pos_rows = unlabeled_samples.filter(pos_mask).with_columns([
            pl.lit(1).alias("class"),
            pl.lit("unlabeled_pos").alias("source"),
        ])
        if len(pos_rows) > 0:
            labeled_rows.append(pos_rows)
            logger.info(f"Extracted {len(pos_rows)} positive samples from unlabeled_samples")

    # Get negative samples from unlabeled
    if neg_indices:
        neg_mask = unlabeled_samples[index_col].is_in(list(neg_indices))
        neg_rows = unlabeled_samples.filter(neg_mask).with_columns([
            pl.lit(0).alias("class"),
            pl.lit("unlabeled_neg").alias("source"),
        ])
        if len(neg_rows) > 0:
            labeled_rows.append(neg_rows)
            logger.info(f"Extracted {len(neg_rows)} negative samples from unlabeled_samples")

    # Combine all
    if not labeled_rows:
        logger.warning("No labeled samples found!")
        return pl.DataFrame()

    result = pl.concat(labeled_rows, how="diagonal")

    # Clean up temporary index column if added
    if "_row_idx" in result.columns and index_col == "_row_idx":
        result = result.drop("_row_idx")

    pos_count = result["class"].sum()
    neg_count = len(result) - pos_count
    logger.info(f"Built labeled dataset: {len(result)} samples ({pos_count} positive, {neg_count} negative)")

    return result


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    print("Usage:")
    print("  from build_labeled_dataset import build_labeled_dataset")
    print("  labeled_df = build_labeled_dataset(")
    print("      unlabeled_samples=unlabeled_samples,")
    print("      known_flares=known_flares,")
    print("      freaky_held_out_indices=[...],")
    print("      forced_positive_indices=[...],")
    print("      forced_negative_indices=[...],")
    print("      expert_labels_file='expert_labels.txt',")
    print("  )")

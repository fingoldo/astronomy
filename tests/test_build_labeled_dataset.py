"""Tests for build_labeled_dataset module."""

import numpy as np
import polars as pl
import pytest

# Skip all tests in this module if catboost is not installed (dependency via active_learning_pipeline)
pytest.importorskip("catboost", reason="catboost not installed")

from build_labeled_dataset import (
    build_labeled_dataset,
    _extract_sequence_from_hf_dataset,
    prepare_recurrent_training_data,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def unlabeled_features() -> pl.DataFrame:
    """Generate sample unlabeled features DataFrame."""
    np.random.seed(42)
    n_rows = 100
    return pl.DataFrame({
        "row_index": list(range(n_rows)),
        "feature_1": np.random.randn(n_rows).tolist(),
        "feature_2": np.random.randn(n_rows).tolist(),
        "mag_mean": np.random.randn(n_rows).tolist(),
    })


@pytest.fixture
def known_flares_features() -> pl.DataFrame:
    """Generate sample known flares features DataFrame."""
    np.random.seed(123)
    n_rows = 20
    return pl.DataFrame({
        "feature_1": np.random.randn(n_rows).tolist(),
        "feature_2": np.random.randn(n_rows).tolist(),
        "mag_mean": np.random.randn(n_rows).tolist(),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Test: build_labeled_dataset
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildLabeledDataset:
    """Tests for build_labeled_dataset function."""

    def test_with_known_flares_only(self, known_flares_features):
        """Test building dataset with known flares only."""
        result = build_labeled_dataset(
            unlabeled_samples=pl.DataFrame({"row_index": [], "feature_1": []}),
            known_flares=known_flares_features,
        )

        assert len(result) == len(known_flares_features)
        assert "class" in result.columns
        assert "source" in result.columns
        assert result["class"].sum() == len(known_flares_features)  # All positive

    def test_with_forced_indices(self, unlabeled_features, known_flares_features):
        """Test building dataset with forced positive/negative indices."""
        forced_pos = [0, 1, 2]
        forced_neg = [10, 11, 12]

        result = build_labeled_dataset(
            unlabeled_samples=unlabeled_features,
            known_flares=known_flares_features,
            forced_positive_indices=forced_pos,
            forced_negative_indices=forced_neg,
        )

        assert len(result) == len(known_flares_features) + len(forced_pos) + len(forced_neg)
        assert "class" in result.columns
        assert "_dataset" in result.columns

    def test_conflict_resolution(self, unlabeled_features, known_flares_features):
        """Test that conflicts are resolved in favor of positives."""
        # Same index in both pos and neg
        conflicting_idx = [5]
        forced_pos = conflicting_idx
        forced_neg = conflicting_idx

        result = build_labeled_dataset(
            unlabeled_samples=unlabeled_features,
            known_flares=known_flares_features,
            forced_positive_indices=forced_pos,
            forced_negative_indices=forced_neg,
        )

        # Check that conflicting index is only included once (as positive)
        unlabeled_rows = result.filter(pl.col("_dataset") == 0)
        assert len(unlabeled_rows) == 1
        assert unlabeled_rows["class"][0] == 1  # Should be positive

    def test_empty_result_warning(self, caplog):
        """Test warning when no labeled samples found."""
        result = build_labeled_dataset(
            unlabeled_samples=pl.DataFrame({"row_index": [0, 1], "feature_1": [0.0, 0.0]}),
            known_flares=None,
        )

        assert len(result) == 0

    def test_preserves_feature_columns(self, unlabeled_features, known_flares_features):
        """Test that original feature columns are preserved."""
        forced_pos = [0]

        result = build_labeled_dataset(
            unlabeled_samples=unlabeled_features,
            known_flares=known_flares_features,
            forced_positive_indices=forced_pos,
        )

        # Check that feature columns from known_flares are present
        assert "feature_1" in result.columns
        assert "feature_2" in result.columns


# ─────────────────────────────────────────────────────────────────────────────
# Test: prepare_recurrent_training_data
# ─────────────────────────────────────────────────────────────────────────────


class TestPrepareRecurrentTrainingData:
    """Tests for prepare_recurrent_training_data function."""

    def test_extracts_features_and_labels(self, known_flares_features):
        """Test that features and labels are correctly extracted."""
        labeled_df = known_flares_features.with_columns([
            pl.lit(1).alias("class"),
            pl.lit("test").alias("source"),
        ])

        features, labels, sequences = prepare_recurrent_training_data(
            labeled_df=labeled_df,
            feature_cols=["feature_1", "feature_2"],
        )

        assert features.shape == (len(labeled_df), 2)
        assert labels.shape == (len(labeled_df),)
        assert sequences is None  # No datasets provided

    def test_handles_nan_values(self, known_flares_features):
        """Test that NaN values are replaced with 0."""
        # Add NaN values
        df_with_nan = known_flares_features.with_columns([
            pl.when(pl.col("feature_1") > 0).then(None).otherwise(pl.col("feature_1")).alias("feature_1"),
            pl.lit(1).alias("class"),
            pl.lit("test").alias("source"),
        ])

        features, labels, _ = prepare_recurrent_training_data(
            labeled_df=df_with_nan,
            feature_cols=["feature_1", "feature_2"],
        )

        # Should not have any NaN values
        assert not np.any(np.isnan(features))

    def test_auto_detects_feature_cols(self, known_flares_features):
        """Test that feature columns are auto-detected."""
        labeled_df = known_flares_features.with_columns([
            pl.lit(1).alias("class"),
            pl.lit("test").alias("source"),
            pl.lit(0).alias("_dataset"),
            pl.lit(0).alias("_orig_idx"),
        ])

        features, labels, _ = prepare_recurrent_training_data(labeled_df=labeled_df)

        # Should auto-detect feature columns (excluding metadata columns)
        assert features.shape[1] == 3  # feature_1, feature_2, mag_mean

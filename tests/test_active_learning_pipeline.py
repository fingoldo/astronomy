"""Tests for active_learning_pipeline module.

Tests key utility functions and helpers that can be tested in isolation
without requiring the full dataset setup.
"""

import numpy as np
import polars as pl
import pytest

# Skip all tests in this module if catboost is not installed
pytest.importorskip("catboost", reason="catboost not installed")

from active_learning_pipeline import (
    _random_split,
    stratified_flare_split,
    compute_bootstrap_stats,
    SampleSource,
    ExpertMode,
    DEFAULT_STRATIFICATION_BINS,
    MIN_SAMPLES_PER_STRATIFICATION_BIN,
    BOOTSTRAP_CONSENSUS_FRACTION,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Enums
# ─────────────────────────────────────────────────────────────────────────────


class TestEnums:
    """Tests for enum classes."""

    def test_sample_source_values(self):
        """Test SampleSource enum has expected values."""
        assert SampleSource.SEED == "seed"
        assert SampleSource.PSEUDO_POS == "pseudo_pos"
        assert SampleSource.PSEUDO_NEG == "pseudo_neg"
        assert SampleSource.FORCED_POS == "forced_pos"
        assert SampleSource.FORCED_NEG == "forced_neg"
        assert SampleSource.EXPERT_POS == "expert_pos"
        assert SampleSource.EXPERT_NEG == "expert_neg"

    def test_sample_source_is_string(self):
        """Test SampleSource values are JSON serializable."""
        # Because it inherits from str, it should be JSON serializable
        import json
        assert json.dumps(SampleSource.SEED) == '"seed"'

    def test_expert_mode_values(self):
        """Test ExpertMode enum has expected values."""
        assert ExpertMode.NO_EXPERT == "no_expert"
        assert ExpertMode.EXPERT == "expert"


# ─────────────────────────────────────────────────────────────────────────────
# Test: _random_split
# ─────────────────────────────────────────────────────────────────────────────


class TestRandomSplit:
    """Tests for _random_split function."""

    def test_basic_split(self):
        """Test basic split with default ratios."""
        all_idx = np.arange(100)
        train, val, held_out = _random_split(all_idx, 0.1, 0.4, 42)

        # Check sizes
        assert len(train) == 10
        assert len(val) == 40
        assert len(held_out) == 50

    def test_no_overlap(self):
        """Test that splits have no overlapping indices."""
        all_idx = np.arange(100)
        train, val, held_out = _random_split(all_idx, 0.2, 0.3, 42)

        all_split = np.concatenate([train, val, held_out])
        assert len(np.unique(all_split)) == len(all_split)

    def test_all_indices_present(self):
        """Test that all original indices are present in some split."""
        all_idx = np.arange(50)
        train, val, held_out = _random_split(all_idx, 0.2, 0.3, 42)

        all_split = set(train) | set(val) | set(held_out)
        assert all_split == set(all_idx)

    def test_reproducible_with_seed(self):
        """Test that same seed produces same split."""
        all_idx = np.arange(100)
        train1, val1, held_out1 = _random_split(all_idx, 0.1, 0.4, 42)
        train2, val2, held_out2 = _random_split(all_idx, 0.1, 0.4, 42)

        np.testing.assert_array_equal(train1, train2)
        np.testing.assert_array_equal(val1, val2)
        np.testing.assert_array_equal(held_out1, held_out2)

    def test_different_seeds_produce_different_splits(self):
        """Test that different seeds produce different splits."""
        all_idx = np.arange(100)
        train1, _, _ = _random_split(all_idx, 0.1, 0.4, 42)
        train2, _, _ = _random_split(all_idx, 0.1, 0.4, 123)

        # Very unlikely to be equal with different seeds
        assert not np.array_equal(train1, train2)


# ─────────────────────────────────────────────────────────────────────────────
# Test: stratified_flare_split
# ─────────────────────────────────────────────────────────────────────────────


class TestStratifiedFlareSplit:
    """Tests for stratified_flare_split function."""

    @pytest.fixture
    def sample_flares_df(self) -> pl.DataFrame:
        """Generate sample flares DataFrame with stratification columns."""
        np.random.seed(42)
        n_samples = 100
        return pl.DataFrame({
            "norm_amplitude_sigma": np.random.uniform(1.0, 10.0, n_samples).tolist(),
            "npoints": np.random.randint(20, 100, n_samples).tolist(),
            "feature_1": np.random.randn(n_samples).tolist(),
        })

    def test_basic_stratified_split(self, sample_flares_df):
        """Test basic stratified split."""
        train, val, held_out = stratified_flare_split(
            sample_flares_df,
            train_ratio=0.1,
            val_ratio=0.4,
            random_state=42,
        )

        total = len(train) + len(val) + len(held_out)
        assert total == len(sample_flares_df)

    def test_no_overlap(self, sample_flares_df):
        """Test that stratified splits have no overlap."""
        train, val, held_out = stratified_flare_split(
            sample_flares_df,
            train_ratio=0.2,
            val_ratio=0.3,
            random_state=42,
        )

        all_split = set(train) | set(val) | set(held_out)
        assert len(all_split) == len(train) + len(val) + len(held_out)

    def test_falls_back_without_strat_cols(self):
        """Test that it falls back to random split without stratification columns."""
        df = pl.DataFrame({
            "feature_1": np.random.randn(50).tolist(),
            "feature_2": np.random.randn(50).tolist(),
        })

        train, val, held_out = stratified_flare_split(
            df,
            train_ratio=0.1,
            val_ratio=0.4,
            random_state=42,
        )

        total = len(train) + len(val) + len(held_out)
        assert total == len(df)

    def test_exclude_indices(self, sample_flares_df):
        """Test that exclude_indices are respected."""
        exclude = [0, 1, 2, 3, 4]
        train, val, held_out = stratified_flare_split(
            sample_flares_df,
            train_ratio=0.1,
            val_ratio=0.4,
            exclude_indices=exclude,
            random_state=42,
        )

        all_split = set(train) | set(val) | set(held_out)

        # Excluded indices should not be in any split
        for idx in exclude:
            assert idx not in all_split

        # Total should be original minus excluded
        assert len(all_split) == len(sample_flares_df) - len(exclude)


# ─────────────────────────────────────────────────────────────────────────────
# Test: compute_bootstrap_stats
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeBootstrapStats:
    """Tests for compute_bootstrap_stats function."""

    def test_basic_stats(self):
        """Test basic bootstrap stats computation."""
        # Shape: (n_models, n_candidates)
        bootstrap_preds = np.array([
            [0.1, 0.9, 0.5],
            [0.2, 0.8, 0.6],
            [0.15, 0.85, 0.55],
        ], dtype=np.float32)

        means, stds, consensus = compute_bootstrap_stats(bootstrap_preds)

        assert means.shape == (3,)
        assert stds.shape == (3,)
        assert consensus.shape == (3,)

        # Check mean values
        np.testing.assert_allclose(means[0], 0.15, atol=1e-5)
        np.testing.assert_allclose(means[1], 0.85, atol=1e-5)

    def test_consensus_is_one_minus_std(self):
        """Test that consensus = 1 - std."""
        bootstrap_preds = np.random.rand(5, 10).astype(np.float32)
        means, stds, consensus = compute_bootstrap_stats(bootstrap_preds)

        np.testing.assert_allclose(consensus, 1.0 - stds, atol=1e-5)

    def test_high_agreement_means_high_consensus(self):
        """Test that high agreement produces high consensus."""
        # All models predict ~0.9
        high_agreement = np.array([
            [0.89, 0.91, 0.90],
            [0.90, 0.90, 0.89],
            [0.91, 0.89, 0.91],
        ], dtype=np.float32)

        _, _, consensus = compute_bootstrap_stats(high_agreement)

        # High agreement should mean high consensus (low std)
        assert all(c > 0.95 for c in consensus)

    def test_low_agreement_means_low_consensus(self):
        """Test that low agreement produces low consensus."""
        # Models disagree significantly
        low_agreement = np.array([
            [0.1, 0.9, 0.5],
            [0.9, 0.1, 0.5],
            [0.5, 0.5, 0.5],
        ], dtype=np.float32)

        _, stds, consensus = compute_bootstrap_stats(low_agreement)

        # At least some candidates should have low consensus
        assert any(c < 0.7 for c in consensus)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Module Constants
# ─────────────────────────────────────────────────────────────────────────────


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_stratification_bins_is_reasonable(self):
        """Test stratification bins is a reasonable value."""
        assert DEFAULT_STRATIFICATION_BINS >= 2
        assert DEFAULT_STRATIFICATION_BINS <= 20

    def test_min_samples_per_bin_is_positive(self):
        """Test minimum samples per bin is positive."""
        assert MIN_SAMPLES_PER_STRATIFICATION_BIN > 0

    def test_bootstrap_consensus_fraction_is_valid(self):
        """Test bootstrap consensus fraction is between 0 and 1."""
        assert 0 < BOOTSTRAP_CONSENSUS_FRACTION <= 1


# ─────────────────────────────────────────────────────────────────────────────
# Test: Expert Labels File Handling
# ─────────────────────────────────────────────────────────────────────────────


from active_learning_pipeline import _load_expert_labels_file


class TestExpertLabelsFileHandling:
    """Tests for expert labels file loading functions."""

    def test_load_nonexistent_file_returns_empty_sets(self, tmp_path):
        """Test loading a nonexistent file returns empty sets."""
        fake_path = tmp_path / "nonexistent_labels.txt"
        pos, neg = _load_expert_labels_file(str(fake_path))
        assert pos == set()
        assert neg == set()

    def test_load_valid_jsonl_file(self, tmp_path):
        """Test loading a valid JSONL file returns correct sets."""
        import json

        labels_file = tmp_path / "expert_labels.txt"
        records = [
            {"iter": 1, "ts": "2024-01-01T00:00:00Z", "pos": [1, 2, 3], "neg": [10, 20]},
            {"iter": 2, "ts": "2024-01-02T00:00:00Z", "pos": [4, 5], "neg": [30, 40, 50]},
        ]
        with open(labels_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        pos, neg = _load_expert_labels_file(str(labels_file))
        assert pos == {1, 2, 3, 4, 5}
        assert neg == {10, 20, 30, 40, 50}

    def test_load_file_with_empty_lines(self, tmp_path):
        """Test that empty lines are skipped."""
        import json

        labels_file = tmp_path / "expert_labels.txt"
        with open(labels_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"pos": [1], "neg": [2]}) + "\n")
            f.write("\n")  # empty line
            f.write("   \n")  # whitespace-only line
            f.write(json.dumps({"pos": [3], "neg": [4]}) + "\n")

        pos, neg = _load_expert_labels_file(str(labels_file))
        assert pos == {1, 3}
        assert neg == {2, 4}

    def test_load_file_with_malformed_json_skips_bad_lines(self, tmp_path):
        """Test that malformed JSON lines are skipped."""
        import json

        labels_file = tmp_path / "expert_labels.txt"
        with open(labels_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"pos": [1], "neg": [2]}) + "\n")
            f.write("not valid json\n")  # malformed
            f.write(json.dumps({"pos": [3], "neg": [4]}) + "\n")

        pos, neg = _load_expert_labels_file(str(labels_file))
        assert pos == {1, 3}
        assert neg == {2, 4}

    def test_load_file_deduplicates_indices(self, tmp_path):
        """Test that duplicate indices are deduplicated."""
        import json

        labels_file = tmp_path / "expert_labels.txt"
        records = [
            {"pos": [1, 2, 3], "neg": [10]},
            {"pos": [2, 3, 4], "neg": [10, 20]},  # 2, 3, 10 are duplicates
        ]
        with open(labels_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        pos, neg = _load_expert_labels_file(str(labels_file))
        assert pos == {1, 2, 3, 4}
        assert neg == {10, 20}

    def test_load_empty_file(self, tmp_path):
        """Test loading an empty file returns empty sets."""
        labels_file = tmp_path / "expert_labels.txt"
        labels_file.touch()

        pos, neg = _load_expert_labels_file(str(labels_file))
        assert pos == set()
        assert neg == set()


# ─────────────────────────────────────────────────────────────────────────────
# Test: Configuration Dataclasses
# ─────────────────────────────────────────────────────────────────────────────


from dataclasses import asdict
from active_learning_pipeline import PseudoLabelConfig


class TestConfigSerialization:
    """Tests for configuration dataclass serialization."""

    def test_pseudo_label_config_to_dict(self):
        """Test PseudoLabelConfig converts to dict correctly."""
        config = PseudoLabelConfig()
        config_dict = asdict(config)

        # Check expected keys exist
        assert "pos_threshold" in config_dict
        assert "neg_threshold" in config_dict
        assert "max_pseudo_pos_per_iter" in config_dict
        assert "max_pseudo_neg_per_iter" in config_dict

    def test_pseudo_label_config_roundtrip(self):
        """Test PseudoLabelConfig can be serialized and recreated."""
        original = PseudoLabelConfig(
            pos_threshold=0.95,
            neg_threshold=0.05,
            max_pseudo_pos_per_iter=100,
            max_pseudo_neg_per_iter=500,
        )
        config_dict = asdict(original)

        # Recreate from dict
        recreated = PseudoLabelConfig(**config_dict)
        assert recreated.pos_threshold == original.pos_threshold
        assert recreated.neg_threshold == original.neg_threshold
        assert recreated.max_pseudo_pos_per_iter == original.max_pseudo_pos_per_iter
        assert recreated.max_pseudo_neg_per_iter == original.max_pseudo_neg_per_iter

    def test_config_json_serializable(self):
        """Test that config dict is JSON serializable."""
        import json

        config = PseudoLabelConfig()
        config_dict = asdict(config)

        # Should not raise
        json_str = json.dumps(config_dict)
        assert isinstance(json_str, str)

        # Should roundtrip
        loaded = json.loads(json_str)
        assert loaded == config_dict

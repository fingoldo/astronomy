"""Tests for recurrent_classifier module using simulated data."""

import numpy as np
import pytest
import torch

from recurrent_classifier import (
    RecurrentClassifierWrapper,
    RecurrentConfig,
    InputMode,
    RNNType,
    LightCurveDataset,
    collate_fn,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: Simulated Data
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def simulated_sequences() -> list[np.ndarray]:
    """Generate variable-length sequences simulating light curves."""
    np.random.seed(42)
    sequences = []
    for _ in range(100):
        seq_len = np.random.randint(20, 100)
        mjd = np.sort(np.random.uniform(58000, 59000, seq_len))
        mag = 15.0 + np.random.randn(seq_len) * 0.5
        magerr = np.abs(np.random.randn(seq_len) * 0.1) + 0.01
        norm = (mag - np.median(mag)) / np.median(magerr)
        seq = np.column_stack([mjd, mag, magerr, norm]).astype(np.float32)
        sequences.append(seq)
    return sequences


@pytest.fixture
def simulated_features() -> np.ndarray:
    """Generate tabular features (like handcrafted statistical features)."""
    np.random.seed(42)
    return np.random.randn(100, 50).astype(np.float32)


@pytest.fixture
def simulated_labels() -> np.ndarray:
    """Generate binary labels with class imbalance (10% positive)."""
    np.random.seed(42)
    labels = np.zeros(100, dtype=np.int64)
    labels[:10] = 1
    np.random.shuffle(labels)
    return labels


@pytest.fixture
def simulated_weights() -> np.ndarray:
    """Generate sample weights."""
    np.random.seed(42)
    return np.random.uniform(0.5, 1.5, 100).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Test: InputMode.FEATURES_ONLY (CatBoost drop-in)
# ─────────────────────────────────────────────────────────────────────────────


class TestFeaturesOnlyMode:
    """Tests for FEATURES_ONLY mode - pure MLP on tabular features."""

    def test_fit_predict(self, simulated_features, simulated_labels):
        """Basic fit and predict."""
        config = RecurrentConfig(
            input_mode=InputMode.FEATURES_ONLY,
            max_epochs=2,
            batch_size=32,
        )
        wrapper = RecurrentClassifierWrapper(config)
        wrapper.fit(features=simulated_features, labels=simulated_labels)

        proba = wrapper.predict_proba(features=simulated_features)
        assert proba.shape == (100, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

        preds = wrapper.predict(features=simulated_features)
        assert preds.shape == (100,)
        assert set(preds).issubset({0, 1})

    def test_with_sample_weights(self, simulated_features, simulated_labels, simulated_weights):
        """Training with sample weights."""
        config = RecurrentConfig(input_mode=InputMode.FEATURES_ONLY, max_epochs=2)
        wrapper = RecurrentClassifierWrapper(config)
        wrapper.fit(
            features=simulated_features,
            labels=simulated_labels,
            sample_weight=simulated_weights,
        )
        proba = wrapper.predict_proba(features=simulated_features)
        assert proba.shape == (100, 2)

    def test_with_class_weight(self, simulated_features, simulated_labels):
        """Training with class weights for imbalance."""
        config = RecurrentConfig(input_mode=InputMode.FEATURES_ONLY, max_epochs=2)
        wrapper = RecurrentClassifierWrapper(config)
        wrapper.fit(
            features=simulated_features,
            labels=simulated_labels,
            class_weight={0: 1.0, 1: 9.0},
        )
        proba = wrapper.predict_proba(features=simulated_features)
        assert proba.shape == (100, 2)

    def test_rejects_missing_features(self, simulated_labels):
        """Should raise error if features missing."""
        config = RecurrentConfig(input_mode=InputMode.FEATURES_ONLY, max_epochs=1)
        wrapper = RecurrentClassifierWrapper(config)

        with pytest.raises(ValueError, match="features required"):
            wrapper.fit(labels=simulated_labels)


# ─────────────────────────────────────────────────────────────────────────────
# Test: InputMode.SEQUENCE_ONLY (Pure RNN)
# ─────────────────────────────────────────────────────────────────────────────


class TestSequenceOnlyMode:
    """Tests for SEQUENCE_ONLY mode - RNN on raw time series."""

    @pytest.mark.parametrize("rnn_type", [RNNType.LSTM, RNNType.GRU, RNNType.RNN])
    def test_all_rnn_types(self, simulated_sequences, simulated_labels, rnn_type):
        """Test all RNN architectures."""
        config = RecurrentConfig(
            input_mode=InputMode.SEQUENCE_ONLY,
            rnn_type=rnn_type,
            hidden_size=32,
            num_layers=1,
            max_epochs=2,
            batch_size=32,
        )
        wrapper = RecurrentClassifierWrapper(config)
        wrapper.fit(sequences=simulated_sequences, labels=simulated_labels)

        proba = wrapper.predict_proba(sequences=simulated_sequences)
        assert proba.shape == (100, 2)

    def test_bidirectional(self, simulated_sequences, simulated_labels):
        """Test bidirectional RNN."""
        config = RecurrentConfig(
            input_mode=InputMode.SEQUENCE_ONLY,
            rnn_type=RNNType.LSTM,
            hidden_size=32,
            num_layers=1,
            bidirectional=True,
            max_epochs=2,
        )
        wrapper = RecurrentClassifierWrapper(config)
        wrapper.fit(sequences=simulated_sequences, labels=simulated_labels)
        proba = wrapper.predict_proba(sequences=simulated_sequences)
        assert proba.shape == (100, 2)

    def test_with_attention(self, simulated_sequences, simulated_labels):
        """Test attention pooling."""
        config = RecurrentConfig(
            input_mode=InputMode.SEQUENCE_ONLY,
            hidden_size=32,
            num_layers=1,
            use_attention=True,
            max_epochs=2,
        )
        wrapper = RecurrentClassifierWrapper(config)
        wrapper.fit(sequences=simulated_sequences, labels=simulated_labels)
        proba = wrapper.predict_proba(sequences=simulated_sequences)
        assert proba.shape == (100, 2)

    def test_without_attention(self, simulated_sequences, simulated_labels):
        """Test last-hidden pooling (no attention)."""
        config = RecurrentConfig(
            input_mode=InputMode.SEQUENCE_ONLY,
            hidden_size=32,
            num_layers=1,
            use_attention=False,
            max_epochs=2,
        )
        wrapper = RecurrentClassifierWrapper(config)
        wrapper.fit(sequences=simulated_sequences, labels=simulated_labels)
        proba = wrapper.predict_proba(sequences=simulated_sequences)
        assert proba.shape == (100, 2)

    def test_variable_length_handling(self):
        """Test handling of highly variable sequence lengths."""
        sequences = [
            np.random.randn(length, 4).astype(np.float32)
            for length in [5, 10, 50, 100, 200]
        ]
        labels = np.array([0, 1, 0, 1, 0], dtype=np.int64)

        config = RecurrentConfig(
            input_mode=InputMode.SEQUENCE_ONLY,
            hidden_size=16,
            num_layers=1,
            max_epochs=2,
            batch_size=5,
        )
        wrapper = RecurrentClassifierWrapper(config)
        wrapper.fit(sequences=sequences, labels=labels)
        proba = wrapper.predict_proba(sequences=sequences)
        assert proba.shape == (5, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Test: InputMode.HYBRID (Sequence + Features)
# ─────────────────────────────────────────────────────────────────────────────


class TestHybridMode:
    """Tests for HYBRID mode - RNN + handcrafted features."""

    def test_fit_predict(self, simulated_sequences, simulated_features, simulated_labels):
        """Basic fit and predict with both inputs."""
        config = RecurrentConfig(
            input_mode=InputMode.HYBRID,
            rnn_type=RNNType.GRU,
            hidden_size=32,
            num_layers=1,
            max_epochs=2,
        )
        wrapper = RecurrentClassifierWrapper(config)
        wrapper.fit(
            sequences=simulated_sequences,
            features=simulated_features,
            labels=simulated_labels,
        )

        proba = wrapper.predict_proba(
            sequences=simulated_sequences,
            features=simulated_features,
        )
        assert proba.shape == (100, 2)

    def test_with_all_options(
        self, simulated_sequences, simulated_features, simulated_labels, simulated_weights
    ):
        """Test with sample weights, class weights, and validation set."""
        config = RecurrentConfig(
            input_mode=InputMode.HYBRID,
            rnn_type=RNNType.LSTM,
            hidden_size=32,
            num_layers=1,
            bidirectional=True,
            use_attention=True,
            max_epochs=3,
        )
        wrapper = RecurrentClassifierWrapper(config)

        # Split for validation
        train_seqs, val_seqs = simulated_sequences[:80], simulated_sequences[80:]
        train_feats, val_feats = simulated_features[:80], simulated_features[80:]
        train_labels, val_labels = simulated_labels[:80], simulated_labels[80:]
        train_weights = simulated_weights[:80]

        wrapper.fit(
            sequences=train_seqs,
            features=train_feats,
            labels=train_labels,
            sample_weight=train_weights,
            eval_set=(val_seqs, val_feats, val_labels),
            class_weight={0: 1.0, 1: 9.0},
        )

        proba = wrapper.predict_proba(sequences=val_seqs, features=val_feats)
        assert proba.shape == (20, 2)

    def test_requires_both_inputs(self, simulated_sequences, simulated_features, simulated_labels):
        """Should raise error if missing required input."""
        config = RecurrentConfig(input_mode=InputMode.HYBRID, max_epochs=1)
        wrapper = RecurrentClassifierWrapper(config)

        with pytest.raises(ValueError, match="both features and sequences required"):
            wrapper.fit(features=simulated_features, labels=simulated_labels)

        with pytest.raises(ValueError, match="both features and sequences required"):
            wrapper.fit(sequences=simulated_sequences, labels=simulated_labels)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Serialization (save/load)
# ─────────────────────────────────────────────────────────────────────────────


class TestSerialization:
    """Tests for model save/load functionality."""

    def test_save_load_features_only(self, simulated_features, simulated_labels, tmp_path):
        """Save and load FEATURES_ONLY model."""
        config = RecurrentConfig(input_mode=InputMode.FEATURES_ONLY, max_epochs=2)
        wrapper = RecurrentClassifierWrapper(config)
        wrapper.fit(features=simulated_features, labels=simulated_labels)

        original_proba = wrapper.predict_proba(features=simulated_features)

        # Save and load
        model_path = tmp_path / "model.pt"
        wrapper.save(model_path)
        loaded = RecurrentClassifierWrapper.load(model_path)

        loaded_proba = loaded.predict_proba(features=simulated_features)
        np.testing.assert_allclose(original_proba, loaded_proba, rtol=1e-5)

    def test_save_load_sequence_only(self, simulated_sequences, simulated_labels, tmp_path):
        """Save and load SEQUENCE_ONLY model."""
        config = RecurrentConfig(
            input_mode=InputMode.SEQUENCE_ONLY,
            hidden_size=32,
            num_layers=1,
            max_epochs=2,
        )
        wrapper = RecurrentClassifierWrapper(config)
        wrapper.fit(sequences=simulated_sequences, labels=simulated_labels)

        original_proba = wrapper.predict_proba(sequences=simulated_sequences)

        model_path = tmp_path / "model.pt"
        wrapper.save(model_path)
        loaded = RecurrentClassifierWrapper.load(model_path)

        loaded_proba = loaded.predict_proba(sequences=simulated_sequences)
        np.testing.assert_allclose(original_proba, loaded_proba, rtol=1e-5)

    def test_save_load_hybrid(
        self, simulated_sequences, simulated_features, simulated_labels, tmp_path
    ):
        """Save and load HYBRID model."""
        config = RecurrentConfig(
            input_mode=InputMode.HYBRID,
            hidden_size=32,
            num_layers=1,
            max_epochs=2,
        )
        wrapper = RecurrentClassifierWrapper(config)
        wrapper.fit(
            sequences=simulated_sequences,
            features=simulated_features,
            labels=simulated_labels,
        )

        original_proba = wrapper.predict_proba(
            sequences=simulated_sequences,
            features=simulated_features,
        )

        model_path = tmp_path / "model.pt"
        wrapper.save(model_path)
        loaded = RecurrentClassifierWrapper.load(model_path)

        loaded_proba = loaded.predict_proba(
            sequences=simulated_sequences,
            features=simulated_features,
        )
        np.testing.assert_allclose(original_proba, loaded_proba, rtol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Dataset and Collate Function
# ─────────────────────────────────────────────────────────────────────────────


class TestDataset:
    """Tests for LightCurveDataset and collate_fn."""

    def test_dataset_length(self, simulated_sequences, simulated_features, simulated_labels):
        """Dataset returns correct length."""
        dataset = LightCurveDataset(
            sequences=simulated_sequences,
            aux_features=simulated_features,
            labels=simulated_labels,
        )
        assert len(dataset) == 100

    def test_collate_padding(self):
        """Collate function pads sequences correctly."""
        sequences = [
            np.random.randn(10, 4).astype(np.float32),
            np.random.randn(20, 4).astype(np.float32),
            np.random.randn(15, 4).astype(np.float32),
        ]
        labels = np.array([0, 1, 0], dtype=np.int64)

        dataset = LightCurveDataset(sequences=sequences, aux_features=None, labels=labels)
        batch = [dataset[i] for i in range(3)]
        collated = collate_fn(batch)

        assert collated["sequences"].shape == (3, 20, 4)  # Padded to max length
        assert collated["lengths"].tolist() == [10, 20, 15]
        assert collated["labels"].shape == (3,)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Configuration Validation
# ─────────────────────────────────────────────────────────────────────────────


class TestConfig:
    """Tests for RecurrentConfig validation."""

    def test_invalid_dropout(self):
        """Should reject invalid dropout values."""
        with pytest.raises(ValueError, match="dropout must be in"):
            RecurrentConfig(dropout=-0.1)

        with pytest.raises(ValueError, match="dropout must be in"):
            RecurrentConfig(dropout=1.0)

    def test_valid_config(self):
        """Valid configs should work."""
        config = RecurrentConfig(
            input_mode=InputMode.HYBRID,
            rnn_type=RNNType.LSTM,
            hidden_size=256,
            dropout=0.5,
        )
        assert config.hidden_size == 256
        assert config.rnn_type == RNNType.LSTM


# ─────────────────────────────────────────────────────────────────────────────
# Test: Prediction Caching
# ─────────────────────────────────────────────────────────────────────────────


class TestCaching:
    """Tests for prediction caching behavior."""

    def test_cache_hit(self, simulated_features, simulated_labels):
        """Same input should return cached result."""
        config = RecurrentConfig(input_mode=InputMode.FEATURES_ONLY, max_epochs=2)
        wrapper = RecurrentClassifierWrapper(config)
        wrapper.fit(features=simulated_features, labels=simulated_labels)

        # First call - cache miss
        proba1 = wrapper.predict_proba(features=simulated_features)

        # Second call - should be cached
        proba2 = wrapper.predict_proba(features=simulated_features)

        np.testing.assert_array_equal(proba1, proba2)

    def test_cache_cleared_on_fit(self, simulated_features, simulated_labels):
        """Cache should be cleared when fit() is called."""
        config = RecurrentConfig(input_mode=InputMode.FEATURES_ONLY, max_epochs=2)
        wrapper = RecurrentClassifierWrapper(config)
        wrapper.fit(features=simulated_features, labels=simulated_labels)

        proba1 = wrapper.predict_proba(features=simulated_features)

        # Re-fit clears cache
        wrapper.fit(features=simulated_features, labels=simulated_labels)

        # Results may differ due to different random initialization
        proba2 = wrapper.predict_proba(features=simulated_features)
        # Just check shape is correct - values may differ
        assert proba2.shape == proba1.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

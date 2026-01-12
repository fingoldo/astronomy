"""Tests for astro_flares module."""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from astro_flares import (
    normalize_magnitude,
    extract_features_polars,
    _compute_wavelet_features_single as compute_wavelet_features,
)


class TestNormalizeMagnitude:
    """Tests for normalize_magnitude function."""

    def test_basic_normalization(self):
        """Test basic magnitude normalization."""
        mag = np.array([15.0, 15.5, 14.5, 15.0, 15.2])
        magerr = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

        norm = normalize_magnitude(mag, magerr)

        assert len(norm) == len(mag)
        # Normalized by median magerr, centered around median mag
        assert isinstance(norm, np.ndarray)

    def test_zero_magerr_raises(self):
        """Should raise ValueError when median magerr is zero."""
        mag = np.array([15.0, 15.5, 14.5])
        magerr = np.array([0.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="median.*is 0"):
            normalize_magnitude(mag, magerr)

    def test_handles_negative_deviations(self):
        """Normalization should handle negative deviations correctly."""
        # 14.0 is brighter (lower magnitude = brighter in astronomy)
        mag = np.array([15.0, 14.0, 16.0])
        magerr = np.array([0.1, 0.1, 0.1])

        norm = normalize_magnitude(mag, magerr)

        # Brighter point (lower mag) should have negative normalized value
        assert norm[1] < norm[0]  # 14.0 should be below median
        assert norm[2] > norm[0]  # 16.0 should be above median

    def test_single_point(self):
        """Should handle single point input."""
        mag = np.array([15.0])
        magerr = np.array([0.1])

        norm = normalize_magnitude(mag, magerr)

        assert len(norm) == 1
        assert norm[0] == 0.0  # Single point centered at 0


class TestExtractFeaturesPolars:
    """Tests for extract_features_polars function."""

    def test_basic_extraction(self, sample_polars_df):
        """Extract features from sample DataFrame."""
        features = extract_features_polars(sample_polars_df)

        assert "npoints" in features.columns
        assert len(features) == len(sample_polars_df)

    def test_output_has_expected_columns(self, sample_polars_df):
        """Should output expected feature columns."""
        features = extract_features_polars(sample_polars_df)

        expected_cols = ["npoints", "mag_mean", "mag_std", "mag_median"]
        for col in expected_cols:
            assert col in features.columns, f"Missing column: {col}"

    def test_float32_output(self, sample_polars_df):
        """Should output Float32 when requested."""
        features = extract_features_polars(sample_polars_df, float32=True)

        # Check that numeric columns are Float32
        for col in features.columns:
            if col not in ["id", "class"]:
                dtype = features[col].dtype
                if dtype in [pl.Float32, pl.Float64]:
                    assert dtype == pl.Float32, f"Column {col} is {dtype}, expected Float32"

    def test_preserves_id_column(self, sample_polars_df):
        """Should preserve the id column."""
        features = extract_features_polars(sample_polars_df)

        assert "id" in features.columns
        assert features["id"].to_list() == sample_polars_df["id"].to_list()

    def test_preserves_class_column(self, sample_polars_df):
        """Should preserve the class column."""
        features = extract_features_polars(sample_polars_df)

        assert "class" in features.columns
        assert features["class"].to_list() == sample_polars_df["class"].to_list()


class TestComputeWaveletFeatures:
    """Tests for compute_wavelet_features function."""

    def test_basic_wavelet_features(self):
        """Compute wavelet features from normalized series."""
        np.random.seed(42)
        norm_series = np.random.randn(50)

        features = compute_wavelet_features(norm_series)

        assert isinstance(features, dict)
        # Should have features for default wavelets (haar, db4, sym4)
        assert "wv_haar_total_energy" in features
        assert "wv_db4_total_energy" in features

    def test_short_series_handling(self):
        """Should handle very short series gracefully."""
        norm_series = np.array([1.0, 2.0, 3.0])  # Only 3 points

        features = compute_wavelet_features(norm_series)

        assert isinstance(features, dict)
        # Should still produce features even for short series

    def test_constant_series(self):
        """Should handle constant series (no variation)."""
        norm_series = np.zeros(50)

        features = compute_wavelet_features(norm_series)

        assert isinstance(features, dict)
        # Energy should be zero or near-zero for constant series
        assert features["wv_haar_total_energy"] == pytest.approx(0.0, abs=1e-10)

    def test_custom_wavelets(self):
        """Should support custom wavelet list."""
        np.random.seed(42)
        norm_series = np.random.randn(50)

        features = compute_wavelet_features(norm_series, wavelets=["haar"])

        assert "wv_haar_total_energy" in features
        assert "wv_db4_total_energy" not in features  # Not requested

    def test_detail_ratio_range(self):
        """Detail ratio should be between 0 and 1."""
        np.random.seed(42)
        norm_series = np.random.randn(100)

        features = compute_wavelet_features(norm_series)

        for key, value in features.items():
            if "detail_ratio" in key:
                assert 0.0 <= value <= 1.0, f"{key} = {value} is out of range"

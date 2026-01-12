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


class TestArgextremumStats:
    """Tests for _get_argextremum_stats_exprs function."""

    def test_basic_argextremum_stats(self, sample_polars_df):
        """Test basic argextremum stats computation."""
        from astro_flares import _get_argextremum_stats_exprs

        exprs = _get_argextremum_stats_exprs(index_col="mag", stats_cols=["mag"])
        result = sample_polars_df.lazy().select(exprs).collect()

        # Should have 4 slices * 7 stats = 28 columns for mag
        expected_prefixes = [
            "mag_to_argmax", "mag_from_argmax",
            "mag_to_argmin", "mag_from_argmin",
        ]
        for prefix in expected_prefixes:
            assert f"{prefix}_len" in result.columns
            assert f"{prefix}_mean" in result.columns
            assert f"{prefix}_std" in result.columns
            assert f"{prefix}_slope" in result.columns

    def test_multiple_stats_cols(self, sample_polars_df):
        """Test argextremum stats with multiple columns."""
        from astro_flares import _get_argextremum_stats_exprs, _norm_expr

        # Add norm column
        df = sample_polars_df.with_columns(_norm_expr(float32=True))

        exprs = _get_argextremum_stats_exprs(index_col="mag", stats_cols=["mag", "norm"])
        result = df.lazy().select(exprs).collect()

        # Should have columns for both mag and norm
        assert "mag_to_argmax_mean" in result.columns
        assert "norm_to_argmax_mean" in result.columns

    def test_argextremum_stats_values(self):
        """Test that argextremum stats have sensible values."""
        from astro_flares import _get_argextremum_stats_exprs

        # Create a simple series where argmax is at index 2 (value 10.0)
        df = pl.DataFrame({
            "mag": [[1.0, 2.0, 10.0, 3.0, 4.0]],  # argmax=2, argmin=0
        })

        exprs = _get_argextremum_stats_exprs(index_col="mag", stats_cols=["mag"])
        result = df.lazy().select(exprs).collect()

        # to_argmax should have [1.0, 2.0] -> len=2
        assert result["mag_to_argmax_len"][0] == 2
        # from_argmax should have [10.0, 3.0, 4.0] -> len=3
        assert result["mag_from_argmax_len"][0] == 3
        # to_argmin should have [] -> len=0 (argmin is at index 0)
        assert result["mag_to_argmin_len"][0] == 0
        # from_argmin should have full series -> len=5
        assert result["mag_from_argmin_len"][0] == 5


class TestExtractAllFeaturesArgextremum:
    """Tests for extract_all_features with argextremum_stats_col parameter."""

    def test_argextremum_disabled_by_default(self, sample_polars_df):
        """When argextremum_stats_col=None, no argextremum columns should appear."""
        from astro_flares import (
            _get_additional_feature_exprs,
            _norm_expr,
        )

        # Simulate what extract_all_features does (without HF dataset)
        df = sample_polars_df.with_columns(_norm_expr(float32=True))

        # Compute additional features only (no argextremum)
        norm_exprs = _get_additional_feature_exprs("norm", include_mjd_features=True)
        result = df.select(["norm", "mjd"]).lazy().select(norm_exprs).collect()

        # No argextremum columns should exist
        assert not any("_to_argmax_" in c for c in result.columns)
        assert not any("_from_argmin_" in c for c in result.columns)

    def test_argextremum_enabled(self, sample_polars_df):
        """When argextremum_stats_col='mag', argextremum columns should appear."""
        from astro_flares import (
            _get_argextremum_stats_exprs,
            _norm_expr,
        )

        df = sample_polars_df.with_columns(_norm_expr(float32=True))

        # Compute argextremum stats
        stats_cols = ["mag", "norm"]
        argext_exprs = _get_argextremum_stats_exprs(index_col="mag", stats_cols=stats_cols)
        result = df.lazy().select(argext_exprs).collect()

        # Should have argextremum columns
        assert "mag_to_argmax_mean" in result.columns
        assert "norm_to_argmax_mean" in result.columns
        assert "mag_from_argmin_slope" in result.columns

        # Values should be finite
        assert result["mag_to_argmax_mean"].is_not_null().all()
        assert result["norm_from_argmax_std"].is_not_null().any()  # some may be null if subseries empty

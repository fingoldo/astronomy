"""Shared pytest fixtures for Astronomy ML tests."""

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def sample_light_curve_data() -> dict:
    """Generate a sample light curve data dict."""
    np.random.seed(42)
    n_points = 50
    return {
        "mjd": list(np.sort(np.random.uniform(58000, 59000, n_points))),
        "mag": list(15.0 + np.random.randn(n_points) * 0.5),
        "magerr": list(np.abs(np.random.randn(n_points) * 0.1) + 0.01),
        "class": 0,
        "id": "TEST_001",
    }


@pytest.fixture
def sample_polars_df() -> pl.DataFrame:
    """Generate a sample Polars DataFrame with list columns."""
    np.random.seed(42)
    n_rows = 10
    return pl.DataFrame({
        "id": [f"ID_{i}" for i in range(n_rows)],
        "class": [0] * 5 + [1] * 5,
        "mjd": [list(np.sort(np.random.uniform(58000, 59000, 30))) for _ in range(n_rows)],
        "mag": [list(15.0 + np.random.randn(30) * 0.5) for _ in range(n_rows)],
        "magerr": [list(np.abs(np.random.randn(30) * 0.1) + 0.01) for _ in range(n_rows)],
    })


@pytest.fixture
def feature_df() -> pl.DataFrame:
    """Generate a sample feature DataFrame."""
    np.random.seed(42)
    n_rows = 100
    return pl.DataFrame({
        "id": [f"ID_{i}" for i in range(n_rows)],
        "class": [0] * 90 + [1] * 10,
        "npoints": np.random.randint(20, 100, n_rows).tolist(),
        "mag_mean": np.random.randn(n_rows).tolist(),
        "mag_std": np.abs(np.random.randn(n_rows)).tolist(),
        "norm_skewness": np.random.randn(n_rows).tolist(),
        "norm_kurtosis": np.random.randn(n_rows).tolist(),
    })


@pytest.fixture
def temp_output_dir(tmp_path):
    """Provide a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir

"""Profiling tests for feature extraction performance in astro_flares.

This module profiles the time cost of each feature group in extract_all_features
to identify performance bottlenecks.

Usage:
    pytest tests/test_feature_profiling.py -v -s

The -s flag is needed to see printed timing results.
"""

import time
from functools import wraps
from typing import Callable

import numpy as np
import polars as pl
import pytest

from astro_flares import (
    extract_features_polars,
    _get_additional_feature_exprs,
    _get_argextremum_stats_exprs,
    compute_fraction_features,
    _compute_wavelet_features_single,
    _clean_single_outlier_native,
    _norm_expr,
    DEFAULT_WAVELETS,
    DEFAULT_ENGINE,
)


# =============================================================================
# Test Configuration
# =============================================================================

# Number of synthetic light curves to generate
N_SAMPLES = 1000

# Points per light curve (realistic range)
MIN_POINTS = 20
MAX_POINTS = 150

# Number of iterations for timing (average)
N_ITERATIONS = 3


# =============================================================================
# Timing Utilities
# =============================================================================


class TimingResult:
    """Store timing results for a feature group."""

    def __init__(self, name: str):
        self.name = name
        self.times: list[float] = []

    def add(self, elapsed: float) -> None:
        self.times.append(elapsed)

    @property
    def mean(self) -> float:
        return np.mean(self.times) if self.times else 0.0

    @property
    def std(self) -> float:
        return np.std(self.times) if len(self.times) > 1 else 0.0

    @property
    def total(self) -> float:
        return sum(self.times)


def time_function(func: Callable, *args, **kwargs) -> tuple[float, any]:
    """Time a function call and return (elapsed_seconds, result)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return elapsed, result


# =============================================================================
# Synthetic Data Generation
# =============================================================================


def generate_synthetic_light_curves(
    n_samples: int = N_SAMPLES,
    min_points: int = MIN_POINTS,
    max_points: int = MAX_POINTS,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate synthetic light curve data similar to ZTF M-dwarf flares.

    Parameters
    ----------
    n_samples : int
        Number of light curves to generate.
    min_points : int
        Minimum points per light curve.
    max_points : int
        Maximum points per light curve.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns: id, class, mjd, mag, magerr (list columns).
    """
    np.random.seed(seed)

    ids = [f"SYNTH_{i:06d}" for i in range(n_samples)]
    classes = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]).tolist()

    mjd_list = []
    mag_list = []
    magerr_list = []

    for i in range(n_samples):
        n_points = np.random.randint(min_points, max_points + 1)

        # Irregular time sampling (typical of survey data)
        mjd_start = np.random.uniform(58000, 59000)
        mjd_span = np.random.uniform(50, 500)  # days
        mjd = np.sort(mjd_start + np.random.uniform(0, mjd_span, n_points))

        # Base magnitude with some variation
        base_mag = np.random.uniform(14, 18)
        scatter = np.random.uniform(0.1, 0.5)

        # Generate magnitude with optional flare
        mag = np.random.normal(base_mag, scatter, n_points)

        # Add flare for positive class
        if classes[i] == 1:
            flare_idx = np.random.randint(n_points // 4, 3 * n_points // 4)
            flare_width = np.random.randint(3, min(10, n_points // 3))
            flare_depth = np.random.uniform(0.5, 2.0)
            for j in range(max(0, flare_idx - flare_width), min(n_points, flare_idx + flare_width)):
                dist = abs(j - flare_idx)
                mag[j] -= flare_depth * np.exp(-dist / (flare_width / 2))

        # Magnitude errors (typical survey errors)
        magerr = np.abs(np.random.normal(0.02, 0.01, n_points)) + 0.01

        mjd_list.append(mjd.tolist())
        mag_list.append(mag.tolist())
        magerr_list.append(magerr.tolist())

    return pl.DataFrame({
        "id": ids,
        "class": classes,
        "mjd": mjd_list,
        "mag": mag_list,
        "magerr": magerr_list,
    })


def add_derived_columns(df: pl.DataFrame, float32: bool = True) -> pl.DataFrame:
    """Add norm and velocity columns to the DataFrame."""
    # Add norm column
    df = df.with_columns(_norm_expr(float32))

    # Add velocity column
    velocity_expr = (
        pl.col("mag").list.eval(pl.element().diff().drop_nulls())
        / pl.col("mjd").list.eval(pl.element().diff().drop_nulls())
    )
    if float32:
        velocity_expr = velocity_expr.list.eval(pl.element().cast(pl.Float32))
    df = df.with_columns(velocity_expr.alias("velocity"))

    return df


# =============================================================================
# Profiling Tests
# =============================================================================


class TestFeatureExtractionProfiling:
    """Profile each feature extraction group for performance analysis."""

    @pytest.fixture(scope="class")
    def synthetic_data(self) -> pl.DataFrame:
        """Generate synthetic data once for all tests."""
        df = generate_synthetic_light_curves(N_SAMPLES)
        df = add_derived_columns(df)
        return df

    @pytest.fixture(scope="class")
    def timing_results(self) -> dict[str, TimingResult]:
        """Store timing results across tests."""
        return {}

    def _profile_feature_group(
        self,
        name: str,
        func: Callable,
        timing_results: dict[str, TimingResult],
        n_iterations: int = N_ITERATIONS,
    ) -> TimingResult:
        """Profile a feature extraction function multiple times."""
        result = TimingResult(name)
        for _ in range(n_iterations):
            elapsed, _ = time_function(func)
            result.add(elapsed)
        timing_results[name] = result
        return result

    def test_01_outlier_detection(self, synthetic_data, timing_results):
        """Profile outlier detection and cleaning."""
        df = synthetic_data.select(["mag", "magerr", "mjd"])

        def run_outlier_detection():
            return _clean_single_outlier_native(df, od_col="mag", od_iqr=40.0)

        result = self._profile_feature_group(
            "1. Outlier Detection",
            run_outlier_detection,
            timing_results,
        )
        print(f"\n  Outlier Detection: {result.mean:.4f}s (+/- {result.std:.4f}s)")

    def test_02_main_features_mag(self, synthetic_data, timing_results):
        """Profile main statistical features for mag column."""
        df = synthetic_data.select(["mag"])

        def run_main_features_mag():
            return extract_features_polars(df, float32=True, engine=DEFAULT_ENGINE)

        result = self._profile_feature_group(
            "2. Main Features (mag)",
            run_main_features_mag,
            timing_results,
        )
        print(f"\n  Main Features (mag): {result.mean:.4f}s (+/- {result.std:.4f}s)")

    def test_03_main_features_magerr(self, synthetic_data, timing_results):
        """Profile main statistical features for magerr column."""
        df = synthetic_data.select(["magerr"])

        def run_main_features_magerr():
            return extract_features_polars(df, float32=True, engine=DEFAULT_ENGINE)

        result = self._profile_feature_group(
            "3. Main Features (magerr)",
            run_main_features_magerr,
            timing_results,
        )
        print(f"\n  Main Features (magerr): {result.mean:.4f}s (+/- {result.std:.4f}s)")

    def test_04_main_features_norm(self, synthetic_data, timing_results):
        """Profile main statistical features for norm column."""
        df = synthetic_data.select(["norm"])

        def run_main_features_norm():
            return extract_features_polars(df, float32=True, engine=DEFAULT_ENGINE)

        result = self._profile_feature_group(
            "4. Main Features (norm)",
            run_main_features_norm,
            timing_results,
        )
        print(f"\n  Main Features (norm): {result.mean:.4f}s (+/- {result.std:.4f}s)")

    def test_05_main_features_velocity(self, synthetic_data, timing_results):
        """Profile main statistical features for velocity column."""
        df = synthetic_data.select(["velocity"])

        def run_main_features_velocity():
            return extract_features_polars(df, float32=True, engine=DEFAULT_ENGINE)

        result = self._profile_feature_group(
            "5. Main Features (velocity)",
            run_main_features_velocity,
            timing_results,
        )
        print(f"\n  Main Features (velocity): {result.mean:.4f}s (+/- {result.std:.4f}s)")

    def test_06_main_features_mjd(self, synthetic_data, timing_results):
        """Profile main statistical features for mjd column."""
        df = synthetic_data.select(["mjd"]).with_columns(
            pl.col("mjd").list.eval(pl.element().cast(pl.Float32))
        )

        def run_main_features_mjd():
            return extract_features_polars(df, float32=True, engine=DEFAULT_ENGINE)

        result = self._profile_feature_group(
            "6. Main Features (mjd)",
            run_main_features_mjd,
            timing_results,
        )
        print(f"\n  Main Features (mjd): {result.mean:.4f}s (+/- {result.std:.4f}s)")

    def test_07_additional_features_mag(self, synthetic_data, timing_results):
        """Profile additional features for mag column."""
        df = synthetic_data.select(["mag"])

        def run_additional_mag():
            exprs = _get_additional_feature_exprs("mag", include_mjd_features=False)
            return df.lazy().select(exprs).collect(engine=DEFAULT_ENGINE)

        result = self._profile_feature_group(
            "7. Additional Features (mag)",
            run_additional_mag,
            timing_results,
        )
        print(f"\n  Additional Features (mag): {result.mean:.4f}s (+/- {result.std:.4f}s)")

    def test_08_additional_features_norm(self, synthetic_data, timing_results):
        """Profile additional features for norm column with mjd."""
        df = synthetic_data.select(["norm", "mjd"])

        def run_additional_norm():
            exprs = _get_additional_feature_exprs("norm", include_mjd_features=True)
            return df.lazy().select(exprs).collect(engine=DEFAULT_ENGINE)

        result = self._profile_feature_group(
            "8. Additional Features (norm+mjd)",
            run_additional_norm,
            timing_results,
        )
        print(f"\n  Additional Features (norm+mjd): {result.mean:.4f}s (+/- {result.std:.4f}s)")

    def test_09_additional_features_velocity(self, synthetic_data, timing_results):
        """Profile additional features for velocity column."""
        df = synthetic_data.select(["velocity"])

        def run_additional_velocity():
            exprs = _get_additional_feature_exprs("velocity", prefix="vel", include_mjd_features=False)
            return df.lazy().select(exprs).collect(engine=DEFAULT_ENGINE)

        result = self._profile_feature_group(
            "9. Additional Features (velocity)",
            run_additional_velocity,
            timing_results,
        )
        print(f"\n  Additional Features (velocity): {result.mean:.4f}s (+/- {result.std:.4f}s)")

    def test_10_argextremum_argmin_only(self, synthetic_data, timing_results):
        """Profile argextremum stats (argmin only - default)."""
        df = synthetic_data.select(["mag", "norm", "velocity"])

        def run_argextremum_argmin():
            exprs = _get_argextremum_stats_exprs(
                index_col="mag",
                stats_cols=["mag", "norm", "velocity"],
                compute_additional=True,
                compute_argmin_stats=True,
                compute_argmax_stats=False,
            )
            return df.lazy().select(exprs).collect(engine=DEFAULT_ENGINE)

        result = self._profile_feature_group(
            "10. Argextremum (argmin only)",
            run_argextremum_argmin,
            timing_results,
        )
        print(f"\n  Argextremum (argmin only): {result.mean:.4f}s (+/- {result.std:.4f}s)")

    def test_11_argextremum_both(self, synthetic_data, timing_results):
        """Profile argextremum stats (both argmin and argmax)."""
        df = synthetic_data.select(["mag", "norm", "velocity"])

        def run_argextremum_both():
            exprs = _get_argextremum_stats_exprs(
                index_col="mag",
                stats_cols=["mag", "norm", "velocity"],
                compute_additional=True,
                compute_argmin_stats=True,
                compute_argmax_stats=True,
            )
            return df.lazy().select(exprs).collect(engine=DEFAULT_ENGINE)

        result = self._profile_feature_group(
            "11. Argextremum (both)",
            run_argextremum_both,
            timing_results,
        )
        print(f"\n  Argextremum (argmin+argmax): {result.mean:.4f}s (+/- {result.std:.4f}s)")

    def test_12_fraction_features(self, synthetic_data, timing_results):
        """Profile fraction features computation."""
        # First need to generate the required columns
        df_mag = synthetic_data.select(["mag"])
        exprs = _get_additional_feature_exprs("mag", prefix="norm", include_mjd_features=False)
        additional = df_mag.lazy().select(exprs).collect(engine=DEFAULT_ENGINE)
        additional = additional.with_columns(pl.lit(50).alias("npoints"))

        def run_fraction_features():
            return compute_fraction_features(additional)

        result = self._profile_feature_group(
            "12. Fraction Features",
            run_fraction_features,
            timing_results,
        )
        print(f"\n  Fraction Features: {result.mean:.4f}s (+/- {result.std:.4f}s)")

    def test_13_wavelet_features(self, synthetic_data, timing_results):
        """Profile wavelet feature extraction (row-by-row)."""
        # Get sample data for wavelet processing
        sample_size = min(100, len(synthetic_data))  # Limit for wavelet (slow)
        df_sample = synthetic_data.head(sample_size)

        def run_wavelet_features():
            results = []
            for i in range(sample_size):
                row = df_sample.row(i, named=True)
                mjd_arr = np.array(row["mjd"], dtype=np.float64)
                mag_arr = np.array(row["mag"], dtype=np.float64)
                magerr_arr = np.array(row["magerr"], dtype=np.float64)

                # Normalize
                med_err = np.median(magerr_arr)
                if med_err > 0:
                    norm = (mag_arr - np.median(mag_arr)) / med_err
                else:
                    norm = mag_arr - np.median(mag_arr)

                features = _compute_wavelet_features_single(
                    norm, mjd_arr,
                    wavelets=DEFAULT_WAVELETS,
                    max_level=6,
                    interpolate=True,
                    n_interp_points=64,
                    prefix="norm"
                )
                results.append(features)
            return pl.DataFrame(results)

        result = self._profile_feature_group(
            f"13. Wavelet Features (n={sample_size})",
            run_wavelet_features,
            timing_results,
            n_iterations=1,  # Only 1 iteration (slow)
        )
        # Extrapolate to full dataset
        extrapolated = result.mean * (N_SAMPLES / sample_size)
        print(f"\n  Wavelet Features ({sample_size} samples): {result.mean:.4f}s")
        print(f"  Wavelet Features (extrapolated to {N_SAMPLES}): {extrapolated:.4f}s")

    def test_99_summary(self, timing_results):
        """Print summary of all timing results."""
        print("\n" + "=" * 70)
        print("FEATURE EXTRACTION TIMING SUMMARY")
        print(f"Dataset: {N_SAMPLES} synthetic light curves")
        print("=" * 70)

        if not timing_results:
            print("No timing results collected.")
            return

        # Sort by mean time (descending)
        sorted_results = sorted(
            timing_results.values(),
            key=lambda r: r.mean,
            reverse=True,
        )

        total_time = sum(r.mean for r in sorted_results)

        print(f"\n{'Feature Group':<45} {'Time (s)':>10} {'% Total':>10}")
        print("-" * 70)

        for result in sorted_results:
            pct = (result.mean / total_time * 100) if total_time > 0 else 0
            print(f"{result.name:<45} {result.mean:>10.4f} {pct:>9.1f}%")

        print("-" * 70)
        print(f"{'TOTAL':<45} {total_time:>10.4f} {'100.0':>9}%")
        print("=" * 70)

        # Top 3 bottlenecks
        print("\nTOP 3 BOTTLENECKS:")
        for i, result in enumerate(sorted_results[:3], 1):
            pct = (result.mean / total_time * 100) if total_time > 0 else 0
            print(f"  {i}. {result.name}: {result.mean:.4f}s ({pct:.1f}%)")

        print()


# =============================================================================
# Quick Profiling Function (for manual testing)
# =============================================================================


def profile_all_features(n_samples: int = 1000, verbose: bool = True) -> dict[str, float]:
    """Run profiling on all feature groups and return timing dict.

    This function can be called directly for quick profiling outside pytest.

    Parameters
    ----------
    n_samples : int
        Number of synthetic samples to generate.
    verbose : bool
        If True, print results as we go.

    Returns
    -------
    dict[str, float]
        Dictionary mapping feature group name to mean time in seconds.
    """
    if verbose:
        print(f"Generating {n_samples} synthetic light curves...")

    df = generate_synthetic_light_curves(n_samples)
    df = add_derived_columns(df)

    results = {}

    # 1. Outlier detection
    if verbose:
        print("Profiling outlier detection...", end=" ", flush=True)
    start = time.perf_counter()
    _clean_single_outlier_native(df.select(["mag", "magerr", "mjd"]), od_col="mag", od_iqr=40.0)
    results["outlier_detection"] = time.perf_counter() - start
    if verbose:
        print(f"{results['outlier_detection']:.4f}s")

    # 2-6. Main features
    for col in ["mag", "magerr", "norm", "velocity", "mjd"]:
        if verbose:
            print(f"Profiling main features ({col})...", end=" ", flush=True)
        col_df = df.select([col])
        if col == "mjd":
            col_df = col_df.with_columns(pl.col(col).list.eval(pl.element().cast(pl.Float32)))
        start = time.perf_counter()
        extract_features_polars(col_df, float32=True, engine=DEFAULT_ENGINE)
        results[f"main_{col}"] = time.perf_counter() - start
        if verbose:
            print(f"{results[f'main_{col}']:.4f}s")

    # 7-9. Additional features
    for col, include_mjd in [("mag", False), ("norm", True), ("velocity", False)]:
        if verbose:
            print(f"Profiling additional features ({col})...", end=" ", flush=True)
        cols_needed = [col] + (["mjd"] if include_mjd else [])
        col_df = df.select(cols_needed)
        prefix = "vel" if col == "velocity" else None
        exprs = _get_additional_feature_exprs(col, prefix=prefix, include_mjd_features=include_mjd)
        start = time.perf_counter()
        col_df.lazy().select(exprs).collect(engine=DEFAULT_ENGINE)
        results[f"additional_{col}"] = time.perf_counter() - start
        if verbose:
            print(f"{results[f'additional_{col}']:.4f}s")

    # 10. Argextremum (argmin only)
    if verbose:
        print("Profiling argextremum (argmin)...", end=" ", flush=True)
    exprs = _get_argextremum_stats_exprs(
        index_col="mag", stats_cols=["mag", "norm", "velocity"],
        compute_additional=True, compute_argmin_stats=True, compute_argmax_stats=False,
    )
    start = time.perf_counter()
    df.select(["mag", "norm", "velocity"]).lazy().select(exprs).collect(engine=DEFAULT_ENGINE)
    results["argextremum_argmin"] = time.perf_counter() - start
    if verbose:
        print(f"{results['argextremum_argmin']:.4f}s")

    # 11. Wavelet features (limited sample)
    wavelet_sample = min(100, n_samples)
    if verbose:
        print(f"Profiling wavelet features ({wavelet_sample} samples)...", end=" ", flush=True)
    start = time.perf_counter()
    for i in range(wavelet_sample):
        row = df.row(i, named=True)
        mjd_arr = np.array(row["mjd"], dtype=np.float64)
        mag_arr = np.array(row["mag"], dtype=np.float64)
        magerr_arr = np.array(row["magerr"], dtype=np.float64)
        med_err = np.median(magerr_arr)
        norm = (mag_arr - np.median(mag_arr)) / med_err if med_err > 0 else mag_arr - np.median(mag_arr)
        _compute_wavelet_features_single(norm, mjd_arr, DEFAULT_WAVELETS, 6, True, 64, "norm")
    wavelet_time = time.perf_counter() - start
    results["wavelet"] = wavelet_time
    results["wavelet_extrapolated"] = wavelet_time * (n_samples / wavelet_sample)
    if verbose:
        print(f"{wavelet_time:.4f}s (extrapolated: {results['wavelet_extrapolated']:.4f}s)")

    if verbose:
        print("\n" + "=" * 50)
        print("SUMMARY (sorted by time)")
        print("=" * 50)
        sorted_items = sorted(results.items(), key=lambda x: x[1], reverse=True)
        total = sum(v for k, v in sorted_items if not k.endswith("_extrapolated"))
        for name, t in sorted_items:
            if not name.endswith("_extrapolated"):
                print(f"  {name:<30}: {t:>8.4f}s ({t/total*100:>5.1f}%)")
        print(f"  {'TOTAL':<30}: {total:>8.4f}s")

    return results


if __name__ == "__main__":
    profile_all_features(n_samples=1000)

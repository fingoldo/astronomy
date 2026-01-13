"""Detailed profiling of wavelet feature extraction.

This module profiles individual wavelets and computation steps to identify
the most expensive parts of wavelet feature extraction.

Usage:
    python tests/test_wavelet_profiling.py

    # Or via pytest:
    pytest tests/test_wavelet_profiling.py -v -s
"""

import time
import warnings
from collections import defaultdict

import numpy as np
import pywt
from scipy.stats import skew, kurtosis

from astro_flares import (
    DEFAULT_WAVELETS,
    MIN_WAVELET_SEQUENCE_LENGTH,
    EPSILON,
)


# =============================================================================
# Profiling Configuration
# =============================================================================

N_SAMPLES = 500  # Number of synthetic light curves
N_INTERP_POINTS = 64  # Interpolation grid size
MAX_LEVEL = 6  # Max wavelet decomposition level
MIN_POINTS = 20
MAX_POINTS = 150


# =============================================================================
# Synthetic Data
# =============================================================================


def generate_synthetic_series(n_samples: int, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate synthetic light curves (norm, mjd) pairs."""
    np.random.seed(seed)
    data = []

    for _ in range(n_samples):
        n_points = np.random.randint(MIN_POINTS, MAX_POINTS + 1)

        # Time array
        mjd_start = np.random.uniform(58000, 59000)
        mjd_span = np.random.uniform(50, 500)
        mjd = np.sort(mjd_start + np.random.uniform(0, mjd_span, n_points))

        # Normalized magnitude (simulate flare-like structure)
        norm = np.random.randn(n_points) * 0.5
        if np.random.random() < 0.1:  # 10% flares
            flare_idx = np.random.randint(n_points // 4, 3 * n_points // 4)
            flare_width = np.random.randint(3, min(10, n_points // 3))
            flare_depth = np.random.uniform(2, 5)
            for j in range(max(0, flare_idx - flare_width), min(n_points, flare_idx + flare_width)):
                dist = abs(j - flare_idx)
                norm[j] -= flare_depth * np.exp(-dist / (flare_width / 2))

        data.append((norm, mjd))

    return data


# =============================================================================
# Profiling Functions
# =============================================================================


def profile_interpolation(data: list[tuple[np.ndarray, np.ndarray]], n_interp_points: int = N_INTERP_POINTS) -> float:
    """Profile interpolation step only."""
    start = time.perf_counter()

    for norm, mjd in data:
        if len(mjd) < 2:
            continue
        mjd_clean = np.nan_to_num(mjd, nan=0.0, posinf=0.0, neginf=0.0)
        t_min, t_max = mjd_clean.min(), mjd_clean.max()
        if t_max > t_min:
            t_regular = np.linspace(t_min, t_max, n_interp_points)
            _ = np.interp(t_regular, mjd_clean, norm)

    return time.perf_counter() - start


def profile_dwt_only(interpolated_data: list[np.ndarray], wavelet: str, max_level: int = MAX_LEVEL) -> float:
    """Profile DWT decomposition only (no feature computation)."""
    start = time.perf_counter()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*boundary effects.*", module="pywt")
        for norm in interpolated_data:
            if len(norm) < MIN_WAVELET_SEQUENCE_LENGTH:
                continue
            actual_max_level = min(max_level, pywt.dwt_max_level(len(norm), wavelet))
            if actual_max_level < 1:
                actual_max_level = 1
            _ = pywt.wavedec(norm, wavelet, level=actual_max_level)

    return time.perf_counter() - start


def profile_global_features(coeffs_list: list[list[np.ndarray]], wavelet: str) -> float:
    """Profile global feature computation (total_energy, detail_ratio, etc.)."""
    start = time.perf_counter()

    for coeffs in coeffs_list:
        if coeffs is None:
            continue
        approx_energy = np.sum(coeffs[0] ** 2)
        detail_energies = [np.sum(c**2) for c in coeffs[1:]]
        total_detail_energy = sum(detail_energies)
        total_energy = approx_energy + total_detail_energy

        # Global features
        _ = float(total_energy)
        _ = float(total_detail_energy / (total_energy + EPSILON))
        _ = float(max(np.max(np.abs(c)) for c in coeffs[1:]) if coeffs[1:] else 0.0)

        # Entropy
        rel_energies = [e / (total_energy + EPSILON) for e in detail_energies]
        _ = -sum(p * np.log(p + EPSILON) for p in rel_energies if p > EPSILON)

        # Detail/approx ratio
        _ = float(total_detail_energy / (approx_energy + EPSILON))

        # Dominant level
        _ = float(np.argmax(detail_energies) + 1 if detail_energies else 0)

    return time.perf_counter() - start


def profile_per_level_features(coeffs_list: list[list[np.ndarray]], max_level: int = MAX_LEVEL) -> float:
    """Profile per-level feature computation (energy, stats per level)."""
    start = time.perf_counter()

    for coeffs in coeffs_list:
        if coeffs is None:
            continue

        approx_energy = np.sum(coeffs[0] ** 2)
        detail_energies = [np.sum(c**2) for c in coeffs[1:]]
        total_energy = approx_energy + sum(detail_energies)
        details = coeffs[1:]

        for lvl in range(1, max_level + 1):
            lvl_idx = lvl - 1
            if lvl_idx < len(detail_energies):
                d = details[lvl_idx]
                d_energy = detail_energies[lvl_idx]

                # Energy features
                _ = float(d_energy)
                _ = float(d_energy / (total_energy + EPSILON))

                # Statistical features
                if len(d) > 1:
                    d_mean = np.mean(d)
                    d_std = np.std(d)
                    d_median = np.median(d)

                    _ = float(d_mean)
                    _ = float(d_std)
                    _ = float(skew(d))
                    _ = float(kurtosis(d))
                    _ = float(np.mean(np.abs(d - d_median)))
                    _ = float(np.mean(np.abs(d) > 2 * d_std) if d_std > EPSILON else 0.0)

    return time.perf_counter() - start


def profile_scipy_stats_only(coeffs_list: list[list[np.ndarray]], max_level: int = MAX_LEVEL) -> float:
    """Profile scipy.stats skew/kurtosis computation only."""
    start = time.perf_counter()

    for coeffs in coeffs_list:
        if coeffs is None:
            continue
        details = coeffs[1:]

        for lvl_idx in range(min(max_level, len(details))):
            d = details[lvl_idx]
            if len(d) > 1:
                _ = float(skew(d))
                _ = float(kurtosis(d))

    return time.perf_counter() - start


def profile_full_wavelet(data: list[tuple[np.ndarray, np.ndarray]], wavelet: str) -> float:
    """Profile full wavelet feature extraction for a single wavelet."""
    start = time.perf_counter()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*boundary effects.*", module="pywt")

        for norm, mjd in data:
            if len(norm) < MIN_WAVELET_SEQUENCE_LENGTH:
                continue

            # Clean
            norm = np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)

            # Interpolate
            if len(mjd) >= 2:
                mjd_clean = np.nan_to_num(mjd, nan=0.0, posinf=0.0, neginf=0.0)
                t_min, t_max = mjd_clean.min(), mjd_clean.max()
                if t_max > t_min:
                    t_regular = np.linspace(t_min, t_max, N_INTERP_POINTS)
                    norm = np.interp(t_regular, mjd_clean, norm)

            # DWT
            actual_max_level = min(MAX_LEVEL, pywt.dwt_max_level(len(norm), wavelet))
            if actual_max_level < 1:
                actual_max_level = 1
            coeffs = pywt.wavedec(norm, wavelet, level=actual_max_level)

            # Global features
            approx_energy = np.sum(coeffs[0] ** 2)
            detail_energies = [np.sum(c**2) for c in coeffs[1:]]
            total_detail_energy = sum(detail_energies)
            total_energy = approx_energy + total_detail_energy

            _ = float(total_energy)
            _ = float(total_detail_energy / (total_energy + EPSILON))
            _ = float(max(np.max(np.abs(c)) for c in coeffs[1:]) if coeffs[1:] else 0.0)

            rel_energies = [e / (total_energy + EPSILON) for e in detail_energies]
            _ = -sum(p * np.log(p + EPSILON) for p in rel_energies if p > EPSILON)
            _ = float(total_detail_energy / (approx_energy + EPSILON))
            _ = float(np.argmax(detail_energies) + 1 if detail_energies else 0)

            # Per-level
            details = coeffs[1:]
            for lvl in range(1, MAX_LEVEL + 1):
                lvl_idx = lvl - 1
                if lvl_idx < len(detail_energies):
                    d = details[lvl_idx]
                    d_energy = detail_energies[lvl_idx]

                    _ = float(d_energy)
                    _ = float(d_energy / (total_energy + EPSILON))

                    if len(d) > 1:
                        d_mean = np.mean(d)
                        d_std = np.std(d)
                        d_median = np.median(d)

                        _ = float(d_mean)
                        _ = float(d_std)
                        _ = float(skew(d))
                        _ = float(kurtosis(d))
                        _ = float(np.mean(np.abs(d - d_median)))
                        _ = float(np.mean(np.abs(d) > 2 * d_std) if d_std > EPSILON else 0.0)

    return time.perf_counter() - start


# =============================================================================
# Main Profiling
# =============================================================================


def run_detailed_wavelet_profiling(n_samples: int = N_SAMPLES, verbose: bool = True):
    """Run detailed profiling of wavelet feature extraction."""

    if verbose:
        print(f"\n{'='*70}")
        print("DETAILED WAVELET FEATURE PROFILING")
        print(f"{'='*70}")
        print(f"Samples: {n_samples}, Interpolation points: {N_INTERP_POINTS}, Max level: {MAX_LEVEL}")
        print(f"Wavelets: {DEFAULT_WAVELETS}")
        print()

    # Generate data
    if verbose:
        print("Generating synthetic data...", end=" ", flush=True)
    data = generate_synthetic_series(n_samples)
    if verbose:
        print("done")

    # ==========================================================================
    # 1. Profile interpolation (shared across all wavelets)
    # ==========================================================================
    if verbose:
        print("\n--- Step 1: Interpolation ---")
    interp_time = profile_interpolation(data)
    if verbose:
        print(f"  Interpolation: {interp_time:.4f}s ({interp_time/n_samples*1000:.3f}ms/sample)")

    # Pre-interpolate data for subsequent tests
    interpolated = []
    for norm, mjd in data:
        if len(mjd) >= 2:
            mjd_clean = np.nan_to_num(mjd, nan=0.0, posinf=0.0, neginf=0.0)
            norm_clean = np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)
            t_min, t_max = mjd_clean.min(), mjd_clean.max()
            if t_max > t_min:
                t_regular = np.linspace(t_min, t_max, N_INTERP_POINTS)
                interpolated.append(np.interp(t_regular, mjd_clean, norm_clean))
            else:
                interpolated.append(norm_clean)
        else:
            interpolated.append(np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0))

    # ==========================================================================
    # 2. Profile DWT per wavelet
    # ==========================================================================
    if verbose:
        print("\n--- Step 2: DWT Decomposition (per wavelet) ---")

    dwt_times = {}
    coeffs_by_wavelet = {}

    for wav in DEFAULT_WAVELETS:
        dwt_time = profile_dwt_only(interpolated, wav)
        dwt_times[wav] = dwt_time

        # Store coefficients for feature computation
        coeffs_list = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*boundary effects.*", module="pywt")
            for norm in interpolated:
                if len(norm) < MIN_WAVELET_SEQUENCE_LENGTH:
                    coeffs_list.append(None)
                    continue
                actual_max_level = min(MAX_LEVEL, pywt.dwt_max_level(len(norm), wav))
                if actual_max_level < 1:
                    actual_max_level = 1
                coeffs_list.append(pywt.wavedec(norm, wav, level=actual_max_level))
        coeffs_by_wavelet[wav] = coeffs_list

        if verbose:
            print(f"  {wav:8s}: {dwt_time:.4f}s ({dwt_time/n_samples*1000:.3f}ms/sample)")

    # ==========================================================================
    # 3. Profile feature computation stages
    # ==========================================================================
    if verbose:
        print("\n--- Step 3: Feature Computation Breakdown ---")

    # Use first wavelet's coefficients for breakdown
    test_wav = DEFAULT_WAVELETS[0]
    test_coeffs = coeffs_by_wavelet[test_wav]

    global_time = profile_global_features(test_coeffs, test_wav)
    per_level_time = profile_per_level_features(test_coeffs)
    scipy_time = profile_scipy_stats_only(test_coeffs)

    if verbose:
        print(f"  Global features (energy, entropy, etc.): {global_time:.4f}s")
        print(f"  Per-level features (all): {per_level_time:.4f}s")
        print(f"    - scipy skew/kurtosis only: {scipy_time:.4f}s ({scipy_time/per_level_time*100:.1f}% of per-level)")

    # ==========================================================================
    # 4. Profile full wavelet extraction per wavelet
    # ==========================================================================
    if verbose:
        print("\n--- Step 4: Full Wavelet Extraction (per wavelet) ---")

    full_times = {}
    for wav in DEFAULT_WAVELETS:
        full_time = profile_full_wavelet(data, wav)
        full_times[wav] = full_time
        if verbose:
            print(f"  {wav:8s}: {full_time:.4f}s ({full_time/n_samples*1000:.3f}ms/sample)")

    # ==========================================================================
    # Summary
    # ==========================================================================
    if verbose:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")

        total_full = sum(full_times.values())

        print(f"\n{'Wavelet':<12} {'Time (s)':>10} {'ms/sample':>12} {'% Total':>10}")
        print("-" * 50)

        for wav in sorted(full_times.keys(), key=lambda w: full_times[w], reverse=True):
            t = full_times[wav]
            pct = t / total_full * 100
            print(f"{wav:<12} {t:>10.4f} {t/n_samples*1000:>12.3f} {pct:>9.1f}%")

        print("-" * 50)
        print(f"{'TOTAL':<12} {total_full:>10.4f} {total_full/n_samples*1000:>12.3f} {'100.0':>9}%")

        # Breakdown
        print(f"\n{'='*70}")
        print("TIME BREAKDOWN (using haar as reference)")
        print(f"{'='*70}")

        haar_interp = interp_time
        haar_dwt = dwt_times['haar']
        haar_features = full_times['haar'] - haar_interp - haar_dwt

        print(f"\n{'Component':<30} {'Time (s)':>10} {'% of Total':>12}")
        print("-" * 55)
        print(f"{'Interpolation (shared)':<30} {haar_interp:>10.4f} {haar_interp/full_times['haar']*100:>11.1f}%")
        print(f"{'DWT decomposition':<30} {haar_dwt:>10.4f} {haar_dwt/full_times['haar']*100:>11.1f}%")
        print(f"{'Feature computation':<30} {haar_features:>10.4f} {haar_features/full_times['haar']*100:>11.1f}%")
        print(f"{'  - scipy skew/kurtosis':<30} {scipy_time:>10.4f} {scipy_time/full_times['haar']*100:>11.1f}%")

        print(f"\n{'='*70}")
        print("OPTIMIZATION RECOMMENDATIONS")
        print(f"{'='*70}")

        # Identify most expensive wavelet
        slowest_wav = max(full_times.keys(), key=lambda w: full_times[w])
        fastest_wav = min(full_times.keys(), key=lambda w: full_times[w])

        print(f"\n1. Slowest wavelet: {slowest_wav} ({full_times[slowest_wav]:.4f}s)")
        print(f"   Fastest wavelet: {fastest_wav} ({full_times[fastest_wav]:.4f}s)")
        print(f"   Ratio: {full_times[slowest_wav]/full_times[fastest_wav]:.2f}x")

        print(f"\n2. scipy.stats (skew/kurtosis) takes {scipy_time/full_times['haar']*100:.1f}% of feature time")
        print(f"   Consider: numba-based skew/kurtosis for speedup")

        if haar_interp > haar_dwt:
            print(f"\n3. Interpolation ({haar_interp:.4f}s) > DWT ({haar_dwt:.4f}s)")
            print(f"   Consider: Pre-interpolate all data once before wavelet loop")

        print()

    return {
        "interpolation": interp_time,
        "dwt_times": dwt_times,
        "full_times": full_times,
        "global_features": global_time,
        "per_level_features": per_level_time,
        "scipy_stats": scipy_time,
    }


# =============================================================================
# Pytest integration
# =============================================================================


class TestWaveletProfiling:
    """Pytest tests for wavelet profiling."""

    def test_profile_wavelets(self):
        """Run wavelet profiling and verify results."""
        results = run_detailed_wavelet_profiling(n_samples=200, verbose=True)

        # Basic sanity checks
        assert results["interpolation"] > 0
        assert all(t > 0 for t in results["dwt_times"].values())
        assert all(t > 0 for t in results["full_times"].values())

        # Verify all default wavelets were profiled
        for wav in DEFAULT_WAVELETS:
            assert wav in results["dwt_times"]
            assert wav in results["full_times"]


if __name__ == "__main__":
    run_detailed_wavelet_profiling(n_samples=500)

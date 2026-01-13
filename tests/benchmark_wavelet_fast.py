"""Benchmark fast wavelet feature extraction using numba."""

import time
import warnings

import numpy as np
import pywt
from numba import njit
from numba.typed import List as NumbaList

EPSILON = 1e-10


@njit(cache=True)
def _compute_array_stats_numba(arr):
    """Compute mean, std, skewness, kurtosis, median."""
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    if n == 1:
        return arr[0], 0.0, 0.0, 0.0, arr[0]

    total = 0.0
    for i in range(n):
        total += arr[i]
    mean = total / n

    m2 = 0.0
    m3 = 0.0
    m4 = 0.0
    for i in range(n):
        diff = arr[i] - mean
        diff2 = diff * diff
        m2 += diff2
        m3 += diff2 * diff
        m4 += diff2 * diff2

    variance = m2 / n
    std = np.sqrt(variance)

    if std > 1e-10:
        skewness = (m3 / n) / (std ** 3)
    else:
        skewness = 0.0

    if variance > 1e-10:
        kurtosis_val = (m4 / n) / (variance ** 2) - 3.0
    else:
        kurtosis_val = 0.0

    sorted_arr = np.sort(arr.copy())
    if n % 2 == 0:
        median = (sorted_arr[n // 2 - 1] + sorted_arr[n // 2]) / 2.0
    else:
        median = sorted_arr[n // 2]

    return mean, std, skewness, kurtosis_val, median


@njit(cache=True)
def _compute_wavelet_stats_numba(coeffs_approx, coeffs_details, max_level):
    """
    Compute all wavelet statistics in one numba call.

    Returns flat array:
    - 6 global: total_energy, detail_ratio, max_detail, entropy, detail_approx_ratio, dominant_level
    - 8 per level * max_level: energy, rel_energy, mean, std, skewness, kurtosis, mad, frac_above_2std
    """
    EPSILON = 1e-10
    n_global = 6
    n_per_level = 8
    n_features = n_global + n_per_level * max_level
    features = np.zeros(n_features, dtype=np.float64)

    # Energies
    approx_energy = np.sum(coeffs_approx ** 2)

    n_details = len(coeffs_details)
    detail_energies = np.zeros(n_details, dtype=np.float64)
    for i in range(n_details):
        detail_energies[i] = np.sum(coeffs_details[i] ** 2)

    total_detail_energy = np.sum(detail_energies)
    total_energy = approx_energy + total_detail_energy

    # Global features
    features[0] = total_energy
    features[1] = total_detail_energy / (total_energy + EPSILON)

    # max_detail
    max_detail = 0.0
    for i in range(n_details):
        max_abs = np.max(np.abs(coeffs_details[i]))
        if max_abs > max_detail:
            max_detail = max_abs
    features[2] = max_detail

    # entropy
    entropy_val = 0.0
    for i in range(n_details):
        p = detail_energies[i] / (total_energy + EPSILON)
        if p > EPSILON:
            entropy_val -= p * np.log(p + EPSILON)
    features[3] = entropy_val

    # detail_approx_ratio
    features[4] = total_detail_energy / (approx_energy + EPSILON)

    # dominant_level
    dominant = 0
    max_energy = 0.0
    for i in range(n_details):
        if detail_energies[i] > max_energy:
            max_energy = detail_energies[i]
            dominant = i + 1
    features[5] = float(dominant)

    # Per-level features
    for lvl in range(max_level):
        base_idx = n_global + lvl * n_per_level

        if lvl < n_details:
            d = coeffs_details[lvl]
            d_energy = detail_energies[lvl]

            features[base_idx + 0] = d_energy
            features[base_idx + 1] = d_energy / (total_energy + EPSILON)

            if len(d) > 1:
                d_mean, d_std, d_skew, d_kurt, d_median = _compute_array_stats_numba(d)
                features[base_idx + 2] = d_mean
                features[base_idx + 3] = d_std
                features[base_idx + 4] = d_skew
                features[base_idx + 5] = d_kurt

                # MAD
                mad = 0.0
                for i in range(len(d)):
                    mad += np.abs(d[i] - d_median)
                mad /= len(d)
                features[base_idx + 6] = mad

                # frac_above_2std
                if d_std > EPSILON:
                    count = 0
                    threshold = 2 * d_std
                    for i in range(len(d)):
                        if np.abs(d[i]) > threshold:
                            count += 1
                    features[base_idx + 7] = float(count) / len(d)
            elif len(d) == 1:
                features[base_idx + 2] = d[0]

    return features


def compute_wavelet_features_fast(
    norm_series: np.ndarray,
    mjd: np.ndarray | None,
    wavelets: list[str],
    max_level: int = 6,
    interpolate: bool = True,
    n_interp_points: int = 64,
    prefix: str = "",
) -> dict[str, float]:
    """Fast wavelet feature computation using numba."""

    MIN_WAVELET_SEQUENCE_LENGTH = 8
    n_global = 6
    n_per_level = 8

    # Pre-compute feature names
    pfx = f"{prefix}_" if prefix else ""
    global_names = ["total_energy", "detail_ratio", "max_detail", "entropy", "detail_approx_ratio", "dominant_level"]
    level_names = ["energy", "rel_energy", "mean", "std", "skewness", "kurtosis", "mad", "frac_above_2std"]

    all_names = {}
    for wav in wavelets:
        wav_names = []
        for name in global_names:
            wav_names.append(f"{pfx}wv_{wav}_{name}")
        for lvl in range(1, max_level + 1):
            for name in level_names:
                wav_names.append(f"{pfx}wv_{wav}_d{lvl}_{name}")
        all_names[wav] = wav_names

    features = {}

    # Handle edge cases
    if len(norm_series) < MIN_WAVELET_SEQUENCE_LENGTH:
        for wav in wavelets:
            for name in all_names[wav]:
                features[name] = 0.0
        return features

    # Clean and interpolate
    norm_series = np.nan_to_num(norm_series, nan=0.0, posinf=0.0, neginf=0.0)
    if mjd is not None and interpolate and len(mjd) >= 2:
        mjd_clean = np.nan_to_num(mjd, nan=0.0, posinf=0.0, neginf=0.0)
        t_min, t_max = mjd_clean.min(), mjd_clean.max()
        if t_max > t_min:
            t_regular = np.linspace(t_min, t_max, n_interp_points)
            norm_series = np.interp(t_regular, mjd_clean, norm_series)

    # Process each wavelet
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*boundary effects.*", module="pywt")

        for wav in wavelets:
            try:
                actual_max_level = min(max_level, pywt.dwt_max_level(len(norm_series), wav))
                if actual_max_level < 1:
                    actual_max_level = 1

                coeffs = pywt.wavedec(norm_series, wav, level=actual_max_level)

                # Convert to typed list for numba
                coeffs_approx = coeffs[0].astype(np.float64)
                coeffs_details = NumbaList()
                for c in coeffs[1:]:
                    coeffs_details.append(c.astype(np.float64))

                # Call numba function
                feat_array = _compute_wavelet_stats_numba(coeffs_approx, coeffs_details, max_level)

                # Map to dict
                names = all_names[wav]
                for i, name in enumerate(names):
                    features[name] = float(feat_array[i])

            except (ValueError, RuntimeError):
                for name in all_names[wav]:
                    features[name] = 0.0

    return features


def main():
    """Run benchmark."""
    # Warmup
    test_arr = np.random.randn(64).astype(np.float64)
    _ = _compute_array_stats_numba(test_arr)

    test_list = NumbaList()
    test_list.append(test_arr)
    test_list.append(test_arr[:32].copy())
    _ = _compute_wavelet_stats_numba(test_arr, test_list, 6)

    # Generate test data
    np.random.seed(42)
    N_SAMPLES = 500
    wavelets = ["haar", "db4", "db6", "coif3", "sym4"]
    data = []
    for _ in range(N_SAMPLES):
        n_points = np.random.randint(20, 150)
        mjd = np.sort(np.random.uniform(58000, 59000, n_points))
        norm = np.random.randn(n_points) * 0.5
        data.append((norm, mjd))

    print("=" * 60)
    print("BENCHMARK: Fast Wavelet Feature Extraction")
    print("=" * 60)
    print(f"Samples: {N_SAMPLES}, Wavelets: {wavelets}")
    print()

    # Warmup actual function
    _ = compute_wavelet_features_fast(data[0][0], data[0][1], wavelets, 6, True, 64, "norm")

    # Benchmark FAST version
    start = time.perf_counter()
    for norm, mjd in data:
        _ = compute_wavelet_features_fast(norm, mjd, wavelets, 6, True, 64, "norm")
    fast_time = time.perf_counter() - start

    print(f"FAST version (numba stats + numba features):")
    print(f"  Total: {fast_time:.4f}s")
    print(f"  Per sample: {fast_time/N_SAMPLES*1000:.3f}ms")

    # Compare with current astro_flares implementation
    from astro_flares import _compute_wavelet_features_single

    # Warmup
    _ = _compute_wavelet_features_single(data[0][0], data[0][1], wavelets, 6, True, 64, "norm")

    start = time.perf_counter()
    for norm, mjd in data:
        _ = _compute_wavelet_features_single(norm, mjd, wavelets, 6, True, 64, "norm")
    current_time = time.perf_counter() - start

    print()
    print(f"CURRENT version (numba stats only):")
    print(f"  Total: {current_time:.4f}s")
    print(f"  Per sample: {current_time/N_SAMPLES*1000:.3f}ms")

    print()
    print("=" * 60)
    speedup = current_time / fast_time
    print(f"SPEEDUP: {speedup:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()

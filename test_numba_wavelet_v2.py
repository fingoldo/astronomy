"""
Full numba wavelet implementation: all 5 wavelets + statistics in one call.
Compares with existing pywt-based implementation.
"""

import numpy as np
import pywt
from numba import njit
import time
import sys
sys.path.insert(0, r'C:\Users\TheLocalCommander\Machine Learning\Astronomy')
from numba.typed import List as NumbaList


# =============================================================================
# HARDCODED FILTER COEFFICIENTS (from pywt)
# =============================================================================

# Haar
HAAR_LO = np.array([0.7071067811865476, 0.7071067811865476], dtype=np.float64)
HAAR_HI = np.array([-0.7071067811865476, 0.7071067811865476], dtype=np.float64)

# Daubechies 4
DB4_LO = np.array([
    -0.010597401785069032, 0.0328830116668852,
    0.030841381835560764, -0.18703481171909309,
    -0.027983769416859854, 0.6308807679298589,
    0.7148465705529157, 0.2303778133088965
], dtype=np.float64)
DB4_HI = np.array([
    -0.2303778133088965, 0.7148465705529157,
    -0.6308807679298589, -0.027983769416859854,
    0.18703481171909309, 0.030841381835560764,
    -0.0328830116668852, -0.010597401785069032
], dtype=np.float64)

# Daubechies 6
DB6_LO = np.array([
    -0.0010773010853084796, 0.004777257510945511,
    0.0005538422011614961, -0.03158203931748603,
    0.027522865530305727, 0.09750160558732304,
    -0.12976686756726194, -0.22626469396543983,
    0.31525035170919763, 0.7511339080210954,
    0.49462389039845306, 0.11154074335010947
], dtype=np.float64)
DB6_HI = np.array([
    -0.11154074335010947, 0.49462389039845306,
    -0.7511339080210954, 0.31525035170919763,
    0.22626469396543983, -0.12976686756726194,
    -0.09750160558732304, 0.027522865530305727,
    0.03158203931748603, 0.0005538422011614961,
    -0.004777257510945511, -0.0010773010853084796
], dtype=np.float64)

# Coiflet 3
COIF3_LO = np.array([
    -3.459977319727278e-05, -7.0983302506379e-05,
    0.0004662169598204029, 0.0011175187708306303,
    -0.0025745176881367972, -0.009007976136730624,
    0.015880544863669452, 0.03455502757329774,
    -0.08230192710629983, -0.07179982161915484,
    0.42848347637737, 0.7937772226260872,
    0.40517690240911824, -0.06112339000297255,
    -0.06577191128146936, 0.023452696142077168,
    0.007782596425672746, -0.003793512864380802
], dtype=np.float64)
COIF3_HI = np.array([
    0.003793512864380802, 0.007782596425672746,
    -0.023452696142077168, -0.06577191128146936,
    0.06112339000297255, 0.40517690240911824,
    -0.7937772226260872, 0.42848347637737,
    0.07179982161915484, -0.08230192710629983,
    -0.03455502757329774, 0.015880544863669452,
    0.009007976136730624, -0.0025745176881367972,
    -0.0011175187708306303, 0.0004662169598204029,
    7.0983302506379e-05, -3.459977319727278e-05
], dtype=np.float64)

# Symlet 4
SYM4_LO = np.array([
    -0.07576571478927333, -0.02963552764599851,
    0.49761866763201545, 0.8037387518059161,
    0.29785779560527736, -0.09921954357684722,
    -0.012603967262037833, 0.0322231006040427
], dtype=np.float64)
SYM4_HI = np.array([
    -0.0322231006040427, -0.012603967262037833,
    0.09921954357684722, 0.29785779560527736,
    -0.8037387518059161, 0.49761866763201545,
    0.02963552764599851, -0.07576571478927333
], dtype=np.float64)

# Constants
N_WAVELETS = 5
N_GLOBAL = 6  # total_energy, detail_ratio, max_detail, entropy, detail_approx_ratio, dominant_level
N_PER_LEVEL = 8  # energy, rel_energy, mean, std, skewness, kurtosis, mad, frac_above_2std
EPSILON = 1e-10


# =============================================================================
# Core numba functions
# =============================================================================

@njit(cache=True, fastmath=True)
def _symmetric_reflect(idx, n):
    """Symmetric boundary reflection."""
    if idx < 0:
        idx = -idx - 1
    if idx >= n:
        idx = 2 * n - 1 - idx
    while idx < 0 or idx >= n:
        if idx < 0:
            idx = -idx - 1
        if idx >= n:
            idx = 2 * n - 1 - idx
    return idx


@njit(cache=True, fastmath=True)
def _dwt_level(signal, lo_filter, hi_filter, approx_out, detail_out):
    """
    Single level DWT into pre-allocated buffers.
    Returns output length.
    """
    n = len(signal)
    flen = len(lo_filter)
    out_len = (n + flen - 1) // 2

    for i in range(out_len):
        lo_sum = 0.0
        hi_sum = 0.0

        for j in range(flen):
            sig_idx = 2 * i + 1 - (flen - 1) + j
            sig_idx = _symmetric_reflect(sig_idx, n)
            lo_sum += signal[sig_idx] * lo_filter[flen - 1 - j]
            hi_sum += signal[sig_idx] * hi_filter[flen - 1 - j]

        approx_out[i] = lo_sum
        detail_out[i] = hi_sum

    return out_len


@njit(cache=True, fastmath=True)
def _compute_array_stats(arr):
    """Compute mean, std, skewness, kurtosis, median for an array."""
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    if n == 1:
        return arr[0], 0.0, 0.0, 0.0, arr[0]

    # Mean
    total = 0.0
    for i in range(n):
        total += arr[i]
    mean = total / n

    # Moments
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

    if std > EPSILON:
        skewness = (m3 / n) / (std ** 3)
    else:
        skewness = 0.0

    if variance > EPSILON:
        kurtosis = (m4 / n) / (variance ** 2) - 3.0
    else:
        kurtosis = 0.0

    # Median
    sorted_arr = np.sort(arr.copy())
    if n % 2 == 0:
        median = (sorted_arr[n // 2 - 1] + sorted_arr[n // 2]) / 2.0
    else:
        median = sorted_arr[n // 2]

    return mean, std, skewness, kurtosis, median


@njit(cache=True, fastmath=True)
def _compute_wavelet_stats_v2(
    approx,
    approx_len,
    details,
    detail_lengths,
    max_level,
    result,
    offset
):
    """
    Compute wavelet statistics and write to result[offset:].

    Uses 2D details array and length array instead of list.
    """
    # Compute approx energy
    approx_energy = 0.0
    for i in range(approx_len):
        approx_energy += approx[i] * approx[i]

    # Detail energies
    n_details = 0
    for lvl in range(max_level):
        if detail_lengths[lvl] > 0:
            n_details = lvl + 1

    detail_energies = np.zeros(max_level, dtype=np.float64)
    for lvl in range(n_details):
        dlen = detail_lengths[lvl]
        energy = 0.0
        for i in range(dlen):
            energy += details[lvl, i] * details[lvl, i]
        detail_energies[lvl] = energy

    total_detail_energy = 0.0
    for lvl in range(n_details):
        total_detail_energy += detail_energies[lvl]

    total_energy = approx_energy + total_detail_energy

    # Global features
    result[offset + 0] = total_energy
    result[offset + 1] = total_detail_energy / (total_energy + EPSILON)  # detail_ratio

    # max_detail
    max_detail = 0.0
    for lvl in range(n_details):
        dlen = detail_lengths[lvl]
        for i in range(dlen):
            val = np.abs(details[lvl, i])
            if val > max_detail:
                max_detail = val
    result[offset + 2] = max_detail

    # entropy
    entropy_val = 0.0
    for lvl in range(n_details):
        p = detail_energies[lvl] / (total_energy + EPSILON)
        if p > EPSILON:
            entropy_val -= p * np.log(p + EPSILON)
    result[offset + 3] = entropy_val

    # detail_approx_ratio
    result[offset + 4] = total_detail_energy / (approx_energy + EPSILON)

    # dominant_level (1-indexed)
    dominant = 0
    max_energy = 0.0
    for lvl in range(n_details):
        if detail_energies[lvl] > max_energy:
            max_energy = detail_energies[lvl]
            dominant = lvl + 1
    result[offset + 5] = float(dominant)

    # Per-level features
    for lvl in range(max_level):
        base_idx = offset + N_GLOBAL + lvl * N_PER_LEVEL

        if lvl < n_details and detail_lengths[lvl] > 0:
            dlen = detail_lengths[lvl]
            d_energy = detail_energies[lvl]

            # Extract detail coefficients for this level
            d = details[lvl, :dlen].copy()

            result[base_idx + 0] = d_energy  # energy
            result[base_idx + 1] = d_energy / (total_energy + EPSILON)  # rel_energy

            if dlen > 1:
                d_mean, d_std, d_skew, d_kurt, d_median = _compute_array_stats(d)
                result[base_idx + 2] = d_mean
                result[base_idx + 3] = d_std
                result[base_idx + 4] = d_skew
                result[base_idx + 5] = d_kurt

                # MAD
                mad = 0.0
                for i in range(dlen):
                    mad += np.abs(d[i] - d_median)
                mad /= dlen
                result[base_idx + 6] = mad

                # frac_above_2std
                if d_std > EPSILON:
                    count = 0
                    threshold = 2 * d_std
                    for i in range(dlen):
                        if np.abs(d[i]) > threshold:
                            count += 1
                    result[base_idx + 7] = float(count) / dlen
            elif dlen == 1:
                result[base_idx + 2] = d[0]


@njit(cache=True, fastmath=True)
def _dwt_max_level(data_len, filter_len):
    """Compute max decomposition level (matches pywt.dwt_max_level)."""
    if filter_len <= 1 or data_len < filter_len:
        return 0
    # floor(log2(data_len / (filter_len - 1)))
    ratio = data_len / (filter_len - 1)
    if ratio <= 1:
        return 0
    level = 0
    while ratio >= 2:
        ratio /= 2
        level += 1
    return level


@njit(cache=True, fastmath=True)
def _process_single_wavelet(
    signal,
    lo_filter,
    hi_filter,
    max_level,
    approx_buf,
    details_buf,
    detail_lengths,
    work_buf,
    result,
    offset
):
    """Process one wavelet: DWT + statistics."""
    n = len(signal)
    flen = len(lo_filter)

    # Compute actual max level for this wavelet (matches pywt behavior)
    actual_max_level = min(max_level, _dwt_max_level(n, flen))
    if actual_max_level < 1:
        actual_max_level = 1

    # Reset
    detail_lengths[:] = 0

    # Copy signal to work buffer
    current_len = n
    for i in range(n):
        work_buf[i] = signal[i]

    # Multilevel decomposition - store temporarily
    temp_details = np.empty((max_level, n + 20), dtype=np.float64)
    temp_lengths = np.zeros(max_level, dtype=np.int64)
    actual_levels = 0

    for level in range(actual_max_level):
        out_len = (current_len + flen - 1) // 2
        if out_len < 1:
            break

        # DWT level
        for i in range(out_len):
            lo_sum = 0.0
            hi_sum = 0.0

            for j in range(flen):
                sig_idx = 2 * i + 1 - (flen - 1) + j
                sig_idx = _symmetric_reflect(sig_idx, current_len)
                lo_sum += work_buf[sig_idx] * lo_filter[flen - 1 - j]
                hi_sum += work_buf[sig_idx] * hi_filter[flen - 1 - j]

            approx_buf[i] = lo_sum
            temp_details[level, i] = hi_sum

        temp_lengths[level] = out_len
        actual_levels = level + 1

        # approx becomes next input
        for i in range(out_len):
            work_buf[i] = approx_buf[i]
        current_len = out_len

    # Reverse order to match pywt convention (coarsest first)
    # pywt stores: [cA, cD_n, cD_{n-1}, ..., cD_1]
    # So details_buf[0] should be coarsest (last computed), details_buf[n-1] = finest (first computed)
    for lvl in range(actual_levels):
        src_lvl = actual_levels - 1 - lvl  # Reverse mapping
        detail_lengths[lvl] = temp_lengths[src_lvl]
        dlen = temp_lengths[src_lvl]
        for i in range(dlen):
            details_buf[lvl, i] = temp_details[src_lvl, i]

    # Compute stats
    _compute_wavelet_stats_v2(approx_buf, current_len, details_buf, detail_lengths, max_level, result, offset)


@njit(cache=True, fastmath=True)
def _compute_all_wavelets_numba(signal, max_level):
    """
    Compute DWT + statistics for all 5 wavelets in one call.

    Returns flat array of shape (5 * (6 + 8*max_level),)
    """
    n_per_wavelet = N_GLOBAL + N_PER_LEVEL * max_level
    n_total = N_WAVELETS * n_per_wavelet

    result = np.zeros(n_total, dtype=np.float64)

    # Allocate work buffers once
    n = len(signal)
    max_buf_size = n + 20  # Some padding for filter overlap
    approx_buf = np.empty(max_buf_size, dtype=np.float64)
    details_buf = np.empty((max_level, max_buf_size), dtype=np.float64)
    detail_lengths = np.zeros(max_level, dtype=np.int64)
    work_buf = np.empty(max_buf_size, dtype=np.float64)

    # Process each wavelet
    _process_single_wavelet(signal, HAAR_LO, HAAR_HI, max_level,
                            approx_buf, details_buf, detail_lengths, work_buf,
                            result, 0 * n_per_wavelet)

    _process_single_wavelet(signal, DB4_LO, DB4_HI, max_level,
                            approx_buf, details_buf, detail_lengths, work_buf,
                            result, 1 * n_per_wavelet)

    _process_single_wavelet(signal, DB6_LO, DB6_HI, max_level,
                            approx_buf, details_buf, detail_lengths, work_buf,
                            result, 2 * n_per_wavelet)

    _process_single_wavelet(signal, COIF3_LO, COIF3_HI, max_level,
                            approx_buf, details_buf, detail_lengths, work_buf,
                            result, 3 * n_per_wavelet)

    _process_single_wavelet(signal, SYM4_LO, SYM4_HI, max_level,
                            approx_buf, details_buf, detail_lengths, work_buf,
                            result, 4 * n_per_wavelet)

    return result


@njit(cache=True, fastmath=True)
def _linear_interp_numba(x_new, x_old, y_old):
    """
    Linear interpolation (like np.interp).
    x_new, x_old must be sorted ascending.
    """
    n_new = len(x_new)
    n_old = len(x_old)
    y_new = np.empty(n_new, dtype=np.float64)

    j = 0
    for i in range(n_new):
        xi = x_new[i]

        # Find bracketing interval
        while j < n_old - 1 and x_old[j + 1] < xi:
            j += 1

        if xi <= x_old[0]:
            y_new[i] = y_old[0]
        elif xi >= x_old[n_old - 1]:
            y_new[i] = y_old[n_old - 1]
        else:
            # Linear interpolation
            x0, x1 = x_old[j], x_old[j + 1]
            y0, y1 = y_old[j], y_old[j + 1]
            t = (xi - x0) / (x1 - x0)
            y_new[i] = y0 + t * (y1 - y0)

    return y_new


@njit(cache=True, fastmath=True)
def compute_wavelet_features_full_numba(
    norm_series,
    mjd,
    max_level,
    do_interpolate,
    n_interp_points
):
    """
    Full pipeline: optional interpolation + all wavelets + statistics.

    Returns flat array of shape (5 * (6 + 8*max_level),)
    """
    MIN_LEN = 8

    # Handle short series
    n_per_wavelet = N_GLOBAL + N_PER_LEVEL * max_level
    n_total = N_WAVELETS * n_per_wavelet

    if len(norm_series) < MIN_LEN:
        return np.zeros(n_total, dtype=np.float64)

    # Clean NaN/inf
    signal = np.empty(len(norm_series), dtype=np.float64)
    for i in range(len(norm_series)):
        v = norm_series[i]
        if np.isnan(v) or np.isinf(v):
            signal[i] = 0.0
        else:
            signal[i] = v

    # Interpolation
    if do_interpolate and len(mjd) >= 2:
        mjd_clean = np.empty(len(mjd), dtype=np.float64)
        for i in range(len(mjd)):
            v = mjd[i]
            if np.isnan(v) or np.isinf(v):
                mjd_clean[i] = 0.0
            else:
                mjd_clean[i] = v

        t_min = mjd_clean[0]
        t_max = mjd_clean[0]
        for i in range(len(mjd_clean)):
            if mjd_clean[i] < t_min:
                t_min = mjd_clean[i]
            if mjd_clean[i] > t_max:
                t_max = mjd_clean[i]

        if t_max > t_min:
            # Create regular grid
            t_regular = np.empty(n_interp_points, dtype=np.float64)
            step = (t_max - t_min) / (n_interp_points - 1)
            for i in range(n_interp_points):
                t_regular[i] = t_min + i * step

            signal = _linear_interp_numba(t_regular, mjd_clean, signal)

    return _compute_all_wavelets_numba(signal, max_level)


# =============================================================================
# Reference implementation using pywt (from astro_flares.py)
# =============================================================================

def compute_wavelet_features_pywt(norm_series, mjd, wavelets, max_level, interpolate, n_interp_points):
    """Reference implementation using pywt."""
    from astro_flares import _compute_wavelet_features_array
    return _compute_wavelet_features_array(norm_series, mjd, wavelets, max_level, interpolate, n_interp_points)


# =============================================================================
# Tests
# =============================================================================

def test_full_correctness():
    """Test full pipeline against existing pywt implementation."""
    print("\n" + "="*60)
    print("FULL PIPELINE CORRECTNESS TEST")
    print("="*60)

    np.random.seed(42)
    wavelets = ['haar', 'db4', 'db6', 'coif3', 'sym4']

    test_cases = [
        (32, False),
        (64, False),
        (100, False),
        (64, True),  # With interpolation
        (100, True),
    ]

    max_level = 4
    n_interp = 128

    # Warmup numba
    print("Warming up numba JIT...")
    _ = compute_wavelet_features_full_numba(np.random.randn(64), np.arange(64, dtype=np.float64), 4, False, 128)

    for sig_len, do_interp in test_cases:
        signal = np.random.randn(sig_len)
        mjd = np.sort(np.random.rand(sig_len) * 100)

        # pywt reference
        result_pywt = compute_wavelet_features_pywt(signal, mjd, wavelets, max_level, do_interp, n_interp)

        # numba
        result_numba = compute_wavelet_features_full_numba(signal, mjd, max_level, do_interp, n_interp)

        max_diff = np.max(np.abs(result_pywt - result_numba))
        rel_diff = max_diff / (np.max(np.abs(result_pywt)) + 1e-10)

        status = "OK" if max_diff < 1e-10 else f"DIFF (max={max_diff:.2e}, rel={rel_diff:.2e})"
        interp_str = "interp" if do_interp else "no_interp"
        print(f"  len={sig_len:3d}, {interp_str:9s}: {status}")


def benchmark_full():
    """Benchmark full pipeline."""
    print("\n" + "="*60)
    print("FULL PIPELINE BENCHMARK")
    print("="*60)

    np.random.seed(42)
    wavelets = ['haar', 'db4', 'db6', 'coif3', 'sym4']

    signal_lengths = [32, 64, 100]
    max_level = 4
    n_interp = 128
    n_iterations = 5000

    # Warmup
    print("Warming up...")
    signal = np.random.randn(64)
    mjd = np.arange(64, dtype=np.float64)
    _ = compute_wavelet_features_full_numba(signal, mjd, max_level, False, n_interp)
    _ = compute_wavelet_features_pywt(signal, mjd, wavelets, max_level, False, n_interp)

    for sig_len in signal_lengths:
        signal = np.random.randn(sig_len)
        mjd = np.sort(np.random.rand(sig_len) * 100).astype(np.float64)

        print(f"\n--- Signal length: {sig_len} ---")

        # pywt timing
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = compute_wavelet_features_pywt(signal, mjd, wavelets, max_level, False, n_interp)
        pywt_time = time.perf_counter() - start

        # numba timing
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = compute_wavelet_features_full_numba(signal, mjd, max_level, False, n_interp)
        numba_time = time.perf_counter() - start

        speedup = pywt_time / numba_time
        pywt_per_call = pywt_time / n_iterations * 1e6
        numba_per_call = numba_time / n_iterations * 1e6

        print(f"  pywt:   {pywt_time:.3f}s total, {pywt_per_call:.1f} µs/call")
        print(f"  numba:  {numba_time:.3f}s total, {numba_per_call:.1f} µs/call")
        print(f"  Speedup: {speedup:.2f}x")


if __name__ == '__main__':
    test_full_correctness()
    benchmark_full()

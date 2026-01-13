"""
Test numba DWT implementation vs pywt.
Compare correctness and performance.
"""

import numpy as np
import pywt
from numba import njit
import time

# =============================================================================
# Wavelet filter coefficients (from pywt)
# =============================================================================

# Get coefficients from pywt
def get_wavelet_filters(name):
    w = pywt.Wavelet(name)
    return np.array(w.dec_lo), np.array(w.dec_hi)

HAAR_DEC_LO, HAAR_DEC_HI = get_wavelet_filters('haar')
DB4_DEC_LO, DB4_DEC_HI = get_wavelet_filters('db4')
DB6_DEC_LO, DB6_DEC_HI = get_wavelet_filters('db6')
COIF3_DEC_LO, COIF3_DEC_HI = get_wavelet_filters('coif3')
SYM4_DEC_LO, SYM4_DEC_HI = get_wavelet_filters('sym4')

print("Filter lengths:")
print(f"  haar:  {len(HAAR_DEC_LO)}")
print(f"  db4:   {len(DB4_DEC_LO)}")
print(f"  db6:   {len(DB6_DEC_LO)}")
print(f"  coif3: {len(COIF3_DEC_LO)}")
print(f"  sym4:  {len(SYM4_DEC_LO)}")

# =============================================================================
# Numba DWT implementation
# =============================================================================

@njit(cache=True)
def _symmetric_reflect(idx, n):
    """
    Symmetric boundary reflection (pywt 'symmetric' mode).
    Reflects at edge values: [..., x[1], x[0], x[0], x[1], ...]
    """
    if idx < 0:
        idx = -idx - 1
    if idx >= n:
        idx = 2 * n - 1 - idx
    # Handle multiple reflections for very small arrays
    while idx < 0 or idx >= n:
        if idx < 0:
            idx = -idx - 1
        if idx >= n:
            idx = 2 * n - 1 - idx
    return idx


@njit(cache=True)
def _dwt_single_level_numba(signal, lo_filter, hi_filter):
    """
    Single level DWT: convolution + decimation.
    Implements pywt mode='symmetric' (half-point symmetric extension).

    Returns (approx, detail) coefficient arrays.
    """
    n = len(signal)
    flen = len(lo_filter)

    # pywt 'symmetric' mode output length
    out_len = (n + flen - 1) // 2

    approx = np.empty(out_len, dtype=np.float64)
    detail = np.empty(out_len, dtype=np.float64)

    # pywt uses reversed filter and different offset calculation
    # Convolution with downsampling by 2
    for i in range(out_len):
        lo_sum = 0.0
        hi_sum = 0.0

        for j in range(flen):
            # Index into signal: downsample by 2, account for filter offset
            # pywt convention: filter is applied centered, then downsampled
            sig_idx = 2 * i + 1 - (flen - 1) + j

            # Apply symmetric boundary extension
            sig_idx = _symmetric_reflect(sig_idx, n)

            lo_sum += signal[sig_idx] * lo_filter[flen - 1 - j]
            hi_sum += signal[sig_idx] * hi_filter[flen - 1 - j]

        approx[i] = lo_sum
        detail[i] = hi_sum

    return approx, detail


@njit(cache=True)
def _wavedec_numba(signal, lo_filter, hi_filter, max_level):
    """
    Multilevel DWT decomposition (like pywt.wavedec).

    Returns:
    - approx: final approximation coefficients
    - details: list of detail coefficients (from level 1 to max_level)
    """
    current = signal.copy()
    details = []

    for level in range(max_level):
        n = len(current)
        flen = len(lo_filter)
        out_len = (n + flen - 1) // 2

        if out_len < 1:
            break

        approx, detail = _dwt_single_level_numba(current, lo_filter, hi_filter)
        details.append(detail)
        current = approx

    return current, details


def wavedec_numba_wrapper(signal, lo_filter, hi_filter, max_level):
    """Python wrapper for testing."""
    approx, details = _wavedec_numba(signal.astype(np.float64), lo_filter, hi_filter, max_level)
    # Return in pywt format: [approx, detail_n, detail_n-1, ..., detail_1]
    result = [approx]
    for d in reversed(details):
        result.append(d)
    return result


# =============================================================================
# Tests
# =============================================================================

def test_dwt_correctness():
    """Test that numba DWT matches pywt."""
    print("\n" + "="*60)
    print("CORRECTNESS TESTS")
    print("="*60)

    np.random.seed(42)

    wavelets = [
        ('haar', HAAR_DEC_LO, HAAR_DEC_HI),
        ('db4', DB4_DEC_LO, DB4_DEC_HI),
        ('db6', DB6_DEC_LO, DB6_DEC_HI),
        ('coif3', COIF3_DEC_LO, COIF3_DEC_HI),
        ('sym4', SYM4_DEC_LO, SYM4_DEC_HI),
    ]

    signal_lengths = [16, 32, 64, 100]
    max_level = 4

    for sig_len in signal_lengths:
        print(f"\nSignal length: {sig_len}")
        signal = np.random.randn(sig_len)

        for wav_name, lo_f, hi_f in wavelets:
            # pywt reference
            actual_level = min(max_level, pywt.dwt_max_level(sig_len, wav_name))
            coeffs_pywt = pywt.wavedec(signal, wav_name, level=actual_level, mode='symmetric')

            # numba implementation
            coeffs_numba = wavedec_numba_wrapper(signal, lo_f, hi_f, actual_level)

            # Compare
            all_match = True
            max_diff = 0.0

            for i, (c_pywt, c_numba) in enumerate(zip(coeffs_pywt, coeffs_numba)):
                if len(c_pywt) != len(c_numba):
                    print(f"  {wav_name}: LENGTH MISMATCH at level {i}: pywt={len(c_pywt)}, numba={len(c_numba)}")
                    all_match = False
                    continue

                diff = np.max(np.abs(c_pywt - c_numba))
                max_diff = max(max_diff, diff)
                if diff > 1e-10:
                    all_match = False

            status = "OK" if all_match else "DIFF"
            print(f"  {wav_name:6s}: {status} (max diff: {max_diff:.2e})")


def benchmark():
    """Benchmark numba vs pywt performance."""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)

    np.random.seed(42)

    wavelets = [
        ('haar', HAAR_DEC_LO, HAAR_DEC_HI),
        ('db4', DB4_DEC_LO, DB4_DEC_HI),
        ('db6', DB6_DEC_LO, DB6_DEC_HI),
        ('coif3', COIF3_DEC_LO, COIF3_DEC_HI),
        ('sym4', SYM4_DEC_LO, SYM4_DEC_HI),
    ]

    signal_lengths = [32, 64, 100]
    max_level = 4
    n_iterations = 10000

    # Warmup numba
    print("\nWarming up numba JIT...")
    signal_warmup = np.random.randn(64)
    for _, lo_f, hi_f in wavelets:
        _ = wavedec_numba_wrapper(signal_warmup, lo_f, hi_f, 3)

    for sig_len in signal_lengths:
        print(f"\n--- Signal length: {sig_len} ---")
        signal = np.random.randn(sig_len)

        # pywt timing (all wavelets)
        start = time.perf_counter()
        for _ in range(n_iterations):
            for wav_name, _, _ in wavelets:
                actual_level = min(max_level, pywt.dwt_max_level(sig_len, wav_name))
                _ = pywt.wavedec(signal, wav_name, level=actual_level, mode='symmetric')
        pywt_time = time.perf_counter() - start

        # numba timing (all wavelets)
        start = time.perf_counter()
        for _ in range(n_iterations):
            for wav_name, lo_f, hi_f in wavelets:
                actual_level = min(max_level, pywt.dwt_max_level(sig_len, wav_name))
                _ = wavedec_numba_wrapper(signal, lo_f, hi_f, actual_level)
        numba_time = time.perf_counter() - start

        speedup = pywt_time / numba_time
        print(f"  pywt:  {pywt_time:.3f}s ({n_iterations} iterations)")
        print(f"  numba: {numba_time:.3f}s ({n_iterations} iterations)")
        print(f"  Speedup: {speedup:.2f}x")


if __name__ == '__main__':
    test_dwt_correctness()
    benchmark()

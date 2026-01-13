"""Debug multilevel wavedec."""

import numpy as np
import pywt
from numba import njit

# DB4 filters
DB4_LO = np.array([
    -0.010597401784997278, 0.032883011666982945,
    0.030841381835986965, -0.18703481171888114,
    -0.02798376941698385, 0.6308807679295904,
    0.7148465705525415, 0.23037781330885523
], dtype=np.float64)
DB4_HI = np.array([
    -0.23037781330885523, 0.7148465705525415,
    -0.6308807679295904, -0.02798376941698385,
    0.18703481171888114, 0.030841381835986965,
    -0.032883011666982945, -0.010597401784997278
], dtype=np.float64)


@njit(cache=True)
def _symmetric_reflect(idx, n):
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


@njit(cache=True)
def dwt_single_level_numba(signal, lo_filter, hi_filter):
    n = len(signal)
    flen = len(lo_filter)
    out_len = (n + flen - 1) // 2

    approx = np.empty(out_len, dtype=np.float64)
    detail = np.empty(out_len, dtype=np.float64)

    for i in range(out_len):
        lo_sum = 0.0
        hi_sum = 0.0

        for j in range(flen):
            sig_idx = 2 * i + 1 - (flen - 1) + j
            sig_idx = _symmetric_reflect(sig_idx, n)
            lo_sum += signal[sig_idx] * lo_filter[flen - 1 - j]
            hi_sum += signal[sig_idx] * hi_filter[flen - 1 - j]

        approx[i] = lo_sum
        detail[i] = hi_sum

    return approx, detail


def wavedec_numba(signal, lo_filter, hi_filter, max_level):
    """Python wrapper for multilevel DWT."""
    current = signal.copy()
    details = []

    for level in range(max_level):
        if len(current) < len(lo_filter):
            break
        approx, detail = dwt_single_level_numba(current, lo_filter, hi_filter)
        details.append(detail)
        current = approx

    # Return in pywt format: [cA, cD_n, cD_{n-1}, ..., cD_1]
    result = [current]
    for d in reversed(details):
        result.append(d)
    return result


# Test
np.random.seed(42)
signal = np.random.randn(64).astype(np.float64)
max_level = 4

print("="*60)
print("MULTILEVEL WAVEDEC COMPARISON")
print("="*60)
print(f"Signal length: {len(signal)}, max_level: {max_level}")
print()

# pywt
actual_level = min(max_level, pywt.dwt_max_level(len(signal), 'db4'))
coeffs_pywt = pywt.wavedec(signal, 'db4', level=actual_level, mode='symmetric')
print(f"pywt actual_level: {actual_level}")
print(f"pywt coeffs count: {len(coeffs_pywt)}")
for i, c in enumerate(coeffs_pywt):
    print(f"  coeffs[{i}] length: {len(c)}")

print()

# numba
coeffs_numba = wavedec_numba(signal, DB4_LO, DB4_HI, actual_level)
print(f"numba coeffs count: {len(coeffs_numba)}")
for i, c in enumerate(coeffs_numba):
    print(f"  coeffs[{i}] length: {len(c)}")

print()

# Compare
print("="*60)
print("COEFFICIENT COMPARISON")
print("="*60)
for i in range(len(coeffs_pywt)):
    c_pywt = coeffs_pywt[i]
    c_numba = coeffs_numba[i] if i < len(coeffs_numba) else None

    if c_numba is None:
        print(f"coeffs[{i}]: numba missing!")
        continue

    if len(c_pywt) != len(c_numba):
        print(f"coeffs[{i}]: LENGTH MISMATCH pywt={len(c_pywt)}, numba={len(c_numba)}")
        continue

    max_diff = np.max(np.abs(c_pywt - c_numba))
    print(f"coeffs[{i}]: max_diff = {max_diff:.2e}")

    if max_diff > 1e-10:
        print(f"  First 5 pywt:  {c_pywt[:5]}")
        print(f"  First 5 numba: {c_numba[:5]}")

# Check energies
print()
print("="*60)
print("ENERGY COMPARISON")
print("="*60)
total_pywt = sum(np.sum(c**2) for c in coeffs_pywt)
total_numba = sum(np.sum(c**2) for c in coeffs_numba)
print(f"Total energy pywt:  {total_pywt:.6f}")
print(f"Total energy numba: {total_numba:.6f}")
print(f"Diff: {abs(total_pywt - total_numba):.2e}")

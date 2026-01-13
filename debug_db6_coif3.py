"""Debug db6 and coif3 DWT coefficients."""

import numpy as np
import pywt
from test_numba_wavelet_v2 import (
    DB6_LO, DB6_HI, COIF3_LO, COIF3_HI,
    _dwt_max_level
)
from numba import njit

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
def dwt_single_level(signal, lo_filter, hi_filter):
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


np.random.seed(42)
signal = np.random.randn(64).astype(np.float64)

print("="*60)
print("DB6 Single Level DWT")
print("="*60)

# pywt
cA_pywt, cD_pywt = pywt.dwt(signal, 'db6', mode='symmetric')
print(f"pywt: cA len={len(cA_pywt)}, cD len={len(cD_pywt)}")

# numba
cA_numba, cD_numba = dwt_single_level(signal, DB6_LO, DB6_HI)
print(f"numba: cA len={len(cA_numba)}, cD len={len(cD_numba)}")

if len(cA_pywt) == len(cA_numba):
    max_diff_cA = np.max(np.abs(cA_pywt - cA_numba))
    max_diff_cD = np.max(np.abs(cD_pywt - cD_numba))
    print(f"Max diff cA: {max_diff_cA:.2e}")
    print(f"Max diff cD: {max_diff_cD:.2e}")
else:
    print("LENGTH MISMATCH!")

# Energy comparison
energy_pywt = np.sum(cA_pywt**2) + np.sum(cD_pywt**2)
energy_numba = np.sum(cA_numba**2) + np.sum(cD_numba**2)
print(f"Energy pywt: {energy_pywt:.6f}")
print(f"Energy numba: {energy_numba:.6f}")

print()
print("="*60)
print("COIF3 Single Level DWT")
print("="*60)

# pywt
cA_pywt, cD_pywt = pywt.dwt(signal, 'coif3', mode='symmetric')
print(f"pywt: cA len={len(cA_pywt)}, cD len={len(cD_pywt)}")

# numba
cA_numba, cD_numba = dwt_single_level(signal, COIF3_LO, COIF3_HI)
print(f"numba: cA len={len(cA_numba)}, cD len={len(cD_numba)}")

if len(cA_pywt) == len(cA_numba):
    max_diff_cA = np.max(np.abs(cA_pywt - cA_numba))
    max_diff_cD = np.max(np.abs(cD_pywt - cD_numba))
    print(f"Max diff cA: {max_diff_cA:.2e}")
    print(f"Max diff cD: {max_diff_cD:.2e}")
else:
    print("LENGTH MISMATCH!")

# Energy comparison
energy_pywt = np.sum(cA_pywt**2) + np.sum(cD_pywt**2)
energy_numba = np.sum(cA_numba**2) + np.sum(cD_numba**2)
print(f"Energy pywt: {energy_pywt:.6f}")
print(f"Energy numba: {energy_numba:.6f}")

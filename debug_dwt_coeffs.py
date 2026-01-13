"""Debug raw DWT coefficients."""

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

# Get pywt filters for comparison
w = pywt.Wavelet('db4')
print("Filter comparison:")
print(f"  pywt dec_lo: {w.dec_lo}")
print(f"  my   DB4_LO: {list(DB4_LO)}")
print(f"  Match: {np.allclose(w.dec_lo, DB4_LO)}")
print()


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


# Test with simple signal
np.random.seed(42)
signal = np.random.randn(16)

print("Signal length:", len(signal))
print()

# pywt single level DWT
cA_pywt, cD_pywt = pywt.dwt(signal, 'db4', mode='symmetric')
print("pywt dwt:")
print(f"  cA length: {len(cA_pywt)}")
print(f"  cD length: {len(cD_pywt)}")

# numba single level DWT
cA_numba, cD_numba = dwt_single_level_numba(signal, DB4_LO, DB4_HI)
print(f"\nnumba dwt:")
print(f"  cA length: {len(cA_numba)}")
print(f"  cD length: {len(cD_numba)}")

print(f"\nOutput length comparison:")
print(f"  pywt: {len(cA_pywt)}, numba: {len(cA_numba)}")

# Compare first few coefficients
print("\n--- Approximation coefficients ---")
min_len = min(len(cA_pywt), len(cA_numba))
for i in range(min_len):
    diff = abs(cA_pywt[i] - cA_numba[i]) if i < len(cA_numba) else float('nan')
    print(f"  [{i}] pywt={cA_pywt[i]:12.8f}, numba={cA_numba[i] if i < len(cA_numba) else 'N/A':12.8f}, diff={diff:.2e}")

print("\n--- Detail coefficients ---")
for i in range(min_len):
    diff = abs(cD_pywt[i] - cD_numba[i]) if i < len(cD_numba) else float('nan')
    print(f"  [{i}] pywt={cD_pywt[i]:12.8f}, numba={cD_numba[i] if i < len(cD_numba) else 'N/A':12.8f}, diff={diff:.2e}")

# Check max differences
if len(cA_pywt) == len(cA_numba):
    print(f"\nMax approx diff: {np.max(np.abs(cA_pywt - cA_numba)):.2e}")
    print(f"Max detail diff: {np.max(np.abs(cD_pywt - cD_numba)):.2e}")

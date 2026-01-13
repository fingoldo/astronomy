"""Debug wavelet differences."""

import numpy as np
import pywt
import sys
sys.path.insert(0, r'C:\Users\TheLocalCommander\Machine Learning\Astronomy')

from test_numba_wavelet_v2 import (
    _compute_all_wavelets_numba,
    HAAR_LO, HAAR_HI, DB4_LO, DB4_HI,
    N_GLOBAL, N_PER_LEVEL, N_WAVELETS
)
from astro_flares import _compute_wavelet_features_array

np.random.seed(42)
signal = np.random.randn(64)
mjd = np.arange(64, dtype=np.float64)
wavelets = ['haar', 'db4', 'db6', 'coif3', 'sym4']
max_level = 4

# pywt reference
result_pywt = _compute_wavelet_features_array(signal, mjd, wavelets, max_level, False, 128)

# numba
result_numba = _compute_all_wavelets_numba(signal, max_level)

print("="*60)
print("COMPARING WAVELETS")
print("="*60)

n_per_wavelet = N_GLOBAL + N_PER_LEVEL * max_level

for i, wav in enumerate(wavelets):
    offset = i * n_per_wavelet
    pywt_slice = result_pywt[offset:offset + n_per_wavelet]
    numba_slice = result_numba[offset:offset + n_per_wavelet]

    diff = np.max(np.abs(pywt_slice - numba_slice))
    print(f"\n{wav}: max_diff = {diff:.2e}")

    # Global features
    print(f"  Global features:")
    global_names = ["total_energy", "detail_ratio", "max_detail", "entropy", "detail_approx_ratio", "dominant_level"]
    for j, name in enumerate(global_names):
        p = pywt_slice[j]
        n = numba_slice[j]
        d = abs(p - n)
        flag = "*" if d > 1e-10 else ""
        print(f"    {name:20s}: pywt={p:12.6f}, numba={n:12.6f}, diff={d:.2e} {flag}")

    # Per-level features (just level 1)
    print(f"  Level 1 features:")
    level_names = ["energy", "rel_energy", "mean", "std", "skewness", "kurtosis", "mad", "frac_above_2std"]
    base = N_GLOBAL
    for j, name in enumerate(level_names):
        p = pywt_slice[base + j]
        n = numba_slice[base + j]
        d = abs(p - n)
        flag = "*" if d > 1e-10 else ""
        print(f"    {name:20s}: pywt={p:12.6f}, numba={n:12.6f}, diff={d:.2e} {flag}")

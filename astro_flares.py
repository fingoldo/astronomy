"""
Flare analysis and visualization utilities for ZTF M-dwarf light curves.

This module provides tools for visualizing and comparing normalized magnitude
series from the ZTF M-dwarf Flares dataset.

Dataset: https://huggingface.co/datasets/snad-space/ztf-m-dwarf-flares-2025
Paper: https://arxiv.org/abs/2510.24655
"""

import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import polars as pl
from datasets import load_dataset, Dataset
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from numba import njit
from numba.typed import List as NumbaList
from pyutilz.system import clean_ram
from scipy import stats
from tqdm import tqdm

# MJD epoch: November 17, 1858, 00:00:00 UTC
MJD_EPOCH = datetime(1858, 11, 17, tzinfo=timezone.utc)

# Conversion factor from matplotlib inches to plotly pixels
INCHES_TO_PIXELS = 100

# Default configuration
DEFAULT_ENGINE = "streaming"
DEFAULT_CACHE_DIR = "data"
DEFAULT_BATCH_SIZE = 5_000_000
MIN_ROWS_FOR_CACHING = 10_000

# Histogram bins for distribution plots
HISTOGRAM_BINS = 30

# Wavelet processing
MIN_WAVELET_SEQUENCE_LENGTH = 8
WAVELET_CHUNK_SIZE = 1_000_000

# Epsilon for numerical stability (prevent division by zero)
EPSILON = 1e-10

# Wavelet feature structure (kept in function for numba compatibility, documented here for reference)
# WAVELET_N_GLOBAL_FEATURES = 6  # total_energy, detail_ratio, max_detail, entropy, detail_approx_ratio, dominant_level
# WAVELET_N_PER_LEVEL_FEATURES = 8  # energy, rel_energy, mean, std, skewness, kurtosis, mad, frac_above_2std

# Default wavelets for feature extraction
# - haar: simple, captures sharp transitions
# - db4, db6: asymmetric, good for fast rise/slow decay
# - coif3: vanishing moments in scaling function, helps with transients
# - sym4: symmetric reference for rise/decay comparison
DEFAULT_WAVELETS = ["haar", "db4", "db6", "coif3", "sym4"]

# =============================================================================
# Wavelet filter coefficients (from pywt, hardcoded for numba)
# =============================================================================

# Haar
_HAAR_LO = np.array([0.7071067811865476, 0.7071067811865476], dtype=np.float64)
_HAAR_HI = np.array([-0.7071067811865476, 0.7071067811865476], dtype=np.float64)

# Daubechies 4
_DB4_LO = np.array(
    [
        -0.010597401785069032,
        0.0328830116668852,
        0.030841381835560764,
        -0.18703481171909309,
        -0.027983769416859854,
        0.6308807679298589,
        0.7148465705529157,
        0.2303778133088965,
    ],
    dtype=np.float64,
)
_DB4_HI = np.array(
    [
        -0.2303778133088965,
        0.7148465705529157,
        -0.6308807679298589,
        -0.027983769416859854,
        0.18703481171909309,
        0.030841381835560764,
        -0.0328830116668852,
        -0.010597401785069032,
    ],
    dtype=np.float64,
)

# Daubechies 6
_DB6_LO = np.array(
    [
        -0.0010773010853084796,
        0.004777257510945511,
        0.0005538422011614961,
        -0.03158203931748603,
        0.027522865530305727,
        0.09750160558732304,
        -0.12976686756726194,
        -0.22626469396543983,
        0.31525035170919763,
        0.7511339080210954,
        0.49462389039845306,
        0.11154074335010947,
    ],
    dtype=np.float64,
)
_DB6_HI = np.array(
    [
        -0.11154074335010947,
        0.49462389039845306,
        -0.7511339080210954,
        0.31525035170919763,
        0.22626469396543983,
        -0.12976686756726194,
        -0.09750160558732304,
        0.027522865530305727,
        0.03158203931748603,
        0.0005538422011614961,
        -0.004777257510945511,
        -0.0010773010853084796,
    ],
    dtype=np.float64,
)

# Coiflet 3
_COIF3_LO = np.array(
    [
        -3.459977319727278e-05,
        -7.0983302506379e-05,
        0.0004662169598204029,
        0.0011175187708306303,
        -0.0025745176881367972,
        -0.009007976136730624,
        0.015880544863669452,
        0.03455502757329774,
        -0.08230192710629983,
        -0.07179982161915484,
        0.42848347637737,
        0.7937772226260872,
        0.40517690240911824,
        -0.06112339000297255,
        -0.06577191128146936,
        0.023452696142077168,
        0.007782596425672746,
        -0.003793512864380802,
    ],
    dtype=np.float64,
)
_COIF3_HI = np.array(
    [
        0.003793512864380802,
        0.007782596425672746,
        -0.023452696142077168,
        -0.06577191128146936,
        0.06112339000297255,
        0.40517690240911824,
        -0.7937772226260872,
        0.42848347637737,
        0.07179982161915484,
        -0.08230192710629983,
        -0.03455502757329774,
        0.015880544863669452,
        0.009007976136730624,
        -0.0025745176881367972,
        -0.0011175187708306303,
        0.0004662169598204029,
        7.0983302506379e-05,
        -3.459977319727278e-05,
    ],
    dtype=np.float64,
)

# Symlet 4
_SYM4_LO = np.array(
    [
        -0.07576571478927333,
        -0.02963552764599851,
        0.49761866763201545,
        0.8037387518059161,
        0.29785779560527736,
        -0.09921954357684722,
        -0.012603967262037833,
        0.0322231006040427,
    ],
    dtype=np.float64,
)
_SYM4_HI = np.array(
    [
        -0.0322231006040427,
        -0.012603967262037833,
        0.09921954357684722,
        0.29785779560527736,
        -0.8037387518059161,
        0.49761866763201545,
        0.02963552764599851,
        -0.07576571478927333,
    ],
    dtype=np.float64,
)

# Wavelet feature structure constants
_WAVELET_N_GLOBAL = 6  # total_energy, detail_ratio, max_detail, entropy, detail_approx_ratio, dominant_level
_WAVELET_N_PER_LEVEL = 8  # energy, rel_energy, mean, std, skewness, kurtosis, mad, frac_above_2std
_WAVELET_N_WAVELETS = 5

DataFrameType = Union[pl.DataFrame, pd.DataFrame]

logger = logging.getLogger(__name__)


# =============================================================================
# Numba-accelerated statistics
# =============================================================================


@njit(cache=True)
def _compute_array_stats_numba(arr: np.ndarray) -> tuple[float, float, float, float, float]:
    """Compute mean, std, skewness, kurtosis, median in one pass (numba-accelerated).

    ~400x faster than scipy.stats.skew/kurtosis for small arrays.

    Uses scipy-compatible formulas:
    - std: population std (ddof=0)
    - skewness: biased estimator (scipy default bias=True)
    - kurtosis: excess kurtosis, biased (scipy default bias=True, fisher=True)

    Parameters
    ----------
    arr : np.ndarray
        1D array of float64 values.

    Returns
    -------
    tuple[float, float, float, float, float]
        (mean, std, skewness, kurtosis, median)
    """
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    if n == 1:
        return arr[0], 0.0, 0.0, 0.0, arr[0]

    # Mean (single pass)
    total = 0.0
    for i in range(n):
        total += arr[i]
    mean = total / n

    # Moments (single pass)
    m2 = 0.0  # sum of (x - mean)^2
    m3 = 0.0  # sum of (x - mean)^3
    m4 = 0.0  # sum of (x - mean)^4
    for i in range(n):
        diff = arr[i] - mean
        diff2 = diff * diff
        m2 += diff2
        m3 += diff2 * diff
        m4 += diff2 * diff2

    # Population std
    variance = m2 / n
    std = np.sqrt(variance)

    # Skewness (biased, scipy default)
    if std > 1e-10:
        skewness = (m3 / n) / (std**3)
    else:
        skewness = 0.0

    # Kurtosis (excess, biased, scipy default)
    if variance > 1e-10:
        kurtosis_val = (m4 / n) / (variance**2) - 3.0
    else:
        kurtosis_val = 0.0

    # Median (requires sort)
    sorted_arr = np.sort(arr)  # np.sort returns a copy by default
    if n % 2 == 0:
        median = (sorted_arr[n // 2 - 1] + sorted_arr[n // 2]) / 2.0
    else:
        median = sorted_arr[n // 2]

    return mean, std, skewness, kurtosis_val, median


@njit(cache=True)
def _compute_wavelet_stats_numba(
    coeffs_approx: np.ndarray,
    coeffs_details: list,
    max_level: int,
) -> np.ndarray:
    """Compute all wavelet statistics in one numba call (~1.5x faster than Python loop).

    Parameters
    ----------
    coeffs_approx : np.ndarray
        Approximation coefficients from DWT.
    coeffs_details : list[np.ndarray]
        List of detail coefficients per level.
    max_level : int
        Maximum decomposition level (for padding).

    Returns
    -------
    np.ndarray
        Flat array of features (6 global + 8*max_level per-level).
        Order: total_energy, detail_ratio, max_detail, entropy, detail_approx_ratio, dominant_level,
               then for each level: energy, rel_energy, mean, std, skewness, kurtosis, mad, frac_above_2std
    """
    EPSILON = 1e-10
    n_global = 6
    n_per_level = 8
    n_features = n_global + n_per_level * max_level
    features = np.zeros(n_features, dtype=np.float64)

    # Compute energies
    approx_energy = np.sum(coeffs_approx**2)

    n_details = len(coeffs_details)
    detail_energies = np.zeros(n_details, dtype=np.float64)
    for i in range(n_details):
        detail_energies[i] = np.sum(coeffs_details[i] ** 2)

    total_detail_energy = np.sum(detail_energies)
    total_energy = approx_energy + total_detail_energy

    # Global features
    features[0] = total_energy
    features[1] = total_detail_energy / (total_energy + EPSILON)  # detail_ratio

    # max_detail
    max_detail = 0.0
    for i in range(n_details):
        max_abs = np.max(np.abs(coeffs_details[i]))
        if max_abs > max_detail:
            max_detail = max_abs
    features[2] = max_detail

    # entropy: -sum(p * log(p))
    entropy_val = 0.0
    for i in range(n_details):
        p = detail_energies[i] / (total_energy + EPSILON)
        if p > EPSILON:
            entropy_val -= p * np.log(p + EPSILON)
    features[3] = entropy_val

    # detail_approx_ratio
    features[4] = total_detail_energy / (approx_energy + EPSILON)

    # dominant_level (1-indexed)
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

            features[base_idx + 0] = d_energy  # energy
            features[base_idx + 1] = d_energy / (total_energy + EPSILON)  # rel_energy

            if len(d) > 1:
                d_mean, d_std, d_skew, d_kurt, d_median = _compute_array_stats_numba(d)
                features[base_idx + 2] = d_mean
                features[base_idx + 3] = d_std
                features[base_idx + 4] = d_skew
                features[base_idx + 5] = d_kurt

                # MAD (vectorized)
                features[base_idx + 6] = np.mean(np.abs(d - d_median))

                # frac_above_2std (vectorized)
                if d_std > EPSILON:
                    threshold = 2 * d_std
                    features[base_idx + 7] = np.mean(np.abs(d) > threshold)
            elif len(d) == 1:
                features[base_idx + 2] = d[0]  # mean = single value
        # else: all zeros (padding)

    return features


# =============================================================================
# Pure Numba DWT implementation (26-69x faster than pywt for short series)
# =============================================================================


@njit(cache=True, fastmath=True)
def _symmetric_reflect(idx: int, n: int) -> int:
    """Symmetric boundary reflection (pywt 'symmetric' mode)."""
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
def _dwt_max_level(data_len: int, filter_len: int) -> int:
    """Compute max decomposition level (matches pywt.dwt_max_level)."""
    if filter_len <= 1 or data_len < filter_len:
        return 0
    ratio = data_len / (filter_len - 1)
    if ratio <= 1:
        return 0
    level = 0
    while ratio >= 2:
        ratio /= 2
        level += 1
    return level


@njit(cache=True, fastmath=True)
def _compute_wavelet_stats_v2(
    approx: np.ndarray,
    approx_len: int,
    details: np.ndarray,
    detail_lengths: np.ndarray,
    max_level: int,
    result: np.ndarray,
    offset: int,
) -> None:
    """Compute wavelet statistics directly into result array."""
    EPSILON = 1e-10

    # Compute approx energy
    approx_energy = 0.0
    for i in range(approx_len):
        approx_energy += approx[i] * approx[i]

    # Find actual number of detail levels
    n_details = 0
    for lvl in range(max_level):
        if detail_lengths[lvl] > 0:
            n_details = lvl + 1

    # Detail energies
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
        base_idx = offset + _WAVELET_N_GLOBAL + lvl * _WAVELET_N_PER_LEVEL

        if lvl < n_details and detail_lengths[lvl] > 0:
            dlen = detail_lengths[lvl]
            d_energy = detail_energies[lvl]
            d = details[lvl, :dlen].copy()

            result[base_idx + 0] = d_energy
            result[base_idx + 1] = d_energy / (total_energy + EPSILON)

            if dlen > 1:
                d_mean, d_std, d_skew, d_kurt, d_median = _compute_array_stats_numba(d)
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
def _process_single_wavelet_numba(
    signal: np.ndarray,
    lo_filter: np.ndarray,
    hi_filter: np.ndarray,
    max_level: int,
    approx_buf: np.ndarray,
    details_buf: np.ndarray,
    detail_lengths: np.ndarray,
    work_buf: np.ndarray,
    result: np.ndarray,
    offset: int,
) -> None:
    """Process one wavelet: DWT + statistics."""
    n = len(signal)
    flen = len(lo_filter)

    # Compute actual max level for this wavelet
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

        # DWT level: convolution + downsampling
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

        # Approx becomes next input
        for i in range(out_len):
            work_buf[i] = approx_buf[i]
        current_len = out_len

    # Reverse order to match pywt convention (coarsest first)
    for lvl in range(actual_levels):
        src_lvl = actual_levels - 1 - lvl
        detail_lengths[lvl] = temp_lengths[src_lvl]
        dlen = temp_lengths[src_lvl]
        for i in range(dlen):
            details_buf[lvl, i] = temp_details[src_lvl, i]

    # Compute stats
    _compute_wavelet_stats_v2(approx_buf, current_len, details_buf, detail_lengths, max_level, result, offset)


@njit(cache=True, fastmath=True)
def _linear_interp_numba(x_new: np.ndarray, x_old: np.ndarray, y_old: np.ndarray) -> np.ndarray:
    """Linear interpolation (like np.interp). Arrays must be sorted."""
    n_new = len(x_new)
    n_old = len(x_old)
    y_new = np.empty(n_new, dtype=np.float64)

    j = 0
    for i in range(n_new):
        xi = x_new[i]
        while j < n_old - 1 and x_old[j + 1] < xi:
            j += 1

        if xi <= x_old[0]:
            y_new[i] = y_old[0]
        elif xi >= x_old[n_old - 1]:
            y_new[i] = y_old[n_old - 1]
        else:
            x0, x1 = x_old[j], x_old[j + 1]
            y0, y1 = y_old[j], y_old[j + 1]
            t = (xi - x0) / (x1 - x0)
            y_new[i] = y0 + t * (y1 - y0)

    return y_new


@njit(cache=True, fastmath=True)
def _compute_all_wavelets_numba(signal: np.ndarray, max_level: int) -> np.ndarray:
    """Compute DWT + statistics for all 5 wavelets in one call (26-69x faster than pywt)."""
    n_per_wavelet = _WAVELET_N_GLOBAL + _WAVELET_N_PER_LEVEL * max_level
    n_total = _WAVELET_N_WAVELETS * n_per_wavelet

    result = np.zeros(n_total, dtype=np.float64)

    # Allocate work buffers once
    n = len(signal)
    max_buf_size = n + 20
    approx_buf = np.empty(max_buf_size, dtype=np.float64)
    details_buf = np.empty((max_level, max_buf_size), dtype=np.float64)
    detail_lengths = np.zeros(max_level, dtype=np.int64)
    work_buf = np.empty(max_buf_size, dtype=np.float64)

    # Process each wavelet
    _process_single_wavelet_numba(signal, _HAAR_LO, _HAAR_HI, max_level, approx_buf, details_buf, detail_lengths, work_buf, result, 0 * n_per_wavelet)

    _process_single_wavelet_numba(signal, _DB4_LO, _DB4_HI, max_level, approx_buf, details_buf, detail_lengths, work_buf, result, 1 * n_per_wavelet)

    _process_single_wavelet_numba(signal, _DB6_LO, _DB6_HI, max_level, approx_buf, details_buf, detail_lengths, work_buf, result, 2 * n_per_wavelet)

    _process_single_wavelet_numba(signal, _COIF3_LO, _COIF3_HI, max_level, approx_buf, details_buf, detail_lengths, work_buf, result, 3 * n_per_wavelet)

    _process_single_wavelet_numba(signal, _SYM4_LO, _SYM4_HI, max_level, approx_buf, details_buf, detail_lengths, work_buf, result, 4 * n_per_wavelet)

    return result


@njit(cache=True, fastmath=True)
def _compute_wavelet_features_numba(
    norm_series: np.ndarray,
    mjd: np.ndarray,
    max_level: int,
    do_interpolate: bool,
    n_interp_points: int,
) -> np.ndarray:
    """Full wavelet pipeline: optional interpolation + all wavelets + statistics."""
    n_per_wavelet = _WAVELET_N_GLOBAL + _WAVELET_N_PER_LEVEL * max_level
    n_total = _WAVELET_N_WAVELETS * n_per_wavelet

    if len(norm_series) < MIN_WAVELET_SEQUENCE_LENGTH:
        return np.zeros(n_total, dtype=np.float64)

    # Clean NaN/inf
    signal = np.empty(len(norm_series), dtype=np.float64)
    for i in range(len(norm_series)):
        v = norm_series[i]
        if np.isnan(v) or np.isinf(v):
            signal[i] = 0.0
        else:
            signal[i] = v

    # Interpolation to regular grid
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
            t_regular = np.empty(n_interp_points, dtype=np.float64)
            step = (t_max - t_min) / (n_interp_points - 1)
            for i in range(n_interp_points):
                t_regular[i] = t_min + i * step
            signal = _linear_interp_numba(t_regular, mjd_clean, signal)

    return _compute_all_wavelets_numba(signal, max_level)


# Pre-computed feature name templates for wavelet features
_WAVELET_GLOBAL_NAMES = ["total_energy", "detail_ratio", "max_detail", "entropy", "detail_approx_ratio", "dominant_level"]
_WAVELET_LEVEL_NAMES = ["energy", "rel_energy", "mean", "std", "skewness", "kurtosis", "mad", "frac_above_2std"]


def _get_wavelet_feature_names(wavelets: list[str], max_level: int, prefix: str = "") -> list[str]:
    """Get ordered list of wavelet feature names for batch processing.

    Parameters
    ----------
    wavelets : list[str]
        Wavelet types (e.g., ["haar", "db4"])
    max_level : int
        Maximum decomposition level
    prefix : str
        Prefix for feature names (e.g., "norm" -> "norm_wv_haar_...")

    Returns
    -------
    list[str]
        Ordered list of feature names matching _compute_wavelet_features_array output
    """
    pfx = f"{prefix}_" if prefix else ""
    names = []
    for wav in wavelets:
        for name in _WAVELET_GLOBAL_NAMES:
            names.append(f"{pfx}wv_{wav}_{name}")
        for lvl in range(1, max_level + 1):
            for name in _WAVELET_LEVEL_NAMES:
                names.append(f"{pfx}wv_{wav}_d{lvl}_{name}")
    return names


def _compute_wavelet_features_array(
    norm_series: np.ndarray,
    mjd: np.ndarray | None,
    wavelets: list[str],
    max_level: int,
    interpolate: bool,
    n_interp_points: int,
) -> np.ndarray:
    """Compute wavelet features returning numpy array (batch-optimized version).

    Uses pure numba implementation (26-69x faster than pywt) when wavelets match
    DEFAULT_WAVELETS, otherwise falls back to pywt.

    Returns
    -------
    np.ndarray
        1D array of shape (n_wavelets * (6 + 8*max_level),) with all features
    """
    # Fast path: use pure numba implementation for default wavelets
    if wavelets == DEFAULT_WAVELETS:
        mjd_arr = mjd if mjd is not None else np.empty(0, dtype=np.float64)
        return _compute_wavelet_features_numba(
            norm_series.astype(np.float64),
            mjd_arr.astype(np.float64),
            max_level,
            interpolate and mjd is not None,
            n_interp_points,
        )

    # Fallback to pywt for custom wavelets
    import pywt

    n_global = 6
    n_per_level = 8
    n_features_per_wavelet = n_global + n_per_level * max_level
    n_total_features = len(wavelets) * n_features_per_wavelet

    if len(norm_series) < MIN_WAVELET_SEQUENCE_LENGTH:
        return np.zeros(n_total_features, dtype=np.float64)

    norm_series = np.nan_to_num(norm_series, nan=0.0, posinf=0.0, neginf=0.0)

    if mjd is not None and interpolate and len(mjd) >= 2:
        mjd_clean = np.nan_to_num(mjd, nan=0.0, posinf=0.0, neginf=0.0)
        t_min, t_max = mjd_clean.min(), mjd_clean.max()
        if t_max > t_min:
            t_regular = np.linspace(t_min, t_max, n_interp_points)
            norm_series = np.interp(t_regular, mjd_clean, norm_series)

    result = np.zeros(n_total_features, dtype=np.float64)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*boundary effects.*", module="pywt")

        for wav_idx, wav in enumerate(wavelets):
            offset = wav_idx * n_features_per_wavelet
            try:
                actual_max_level = min(max_level, pywt.dwt_max_level(len(norm_series), wav))
                if actual_max_level < 1:
                    actual_max_level = 1

                coeffs = pywt.wavedec(norm_series, wav, level=actual_max_level)

                coeffs_approx = coeffs[0].astype(np.float64)
                coeffs_details = NumbaList()
                for c in coeffs[1:]:
                    coeffs_details.append(c.astype(np.float64))

                feat_array = _compute_wavelet_stats_numba(coeffs_approx, coeffs_details, max_level)
                result[offset : offset + n_features_per_wavelet] = feat_array

            except (ValueError, RuntimeError):
                pass

    return result


# =============================================================================
# Helper Functions
# =============================================================================


def _get_row(df: DataFrameType | dict, index: int) -> dict:
    """Extract row as dict from any supported DataFrame type.

    Parameters
    ----------
    df : DataFrameType or dict
        DataFrame or dict containing the data.
    index : int
        Row index to extract (ignored if df is dict).

    Returns
    -------
    dict
        Row data as dictionary.

    Raises
    ------
    TypeError
        If df is not a supported type.
    """
    if isinstance(df, pl.DataFrame):
        return df.row(index, named=True)
    elif isinstance(df, dict):
        return df
    elif isinstance(df, pd.DataFrame):
        return df.iloc[index].to_dict()
    raise TypeError(f"Unsupported type: {type(df)}")


def _figsize_to_pixels(figsize: tuple[int, int]) -> tuple[int, int]:
    """Convert matplotlib figsize (inches) to plotly pixels.

    Parameters
    ----------
    figsize : tuple[int, int]
        Figure size as (width, height) in inches.

    Returns
    -------
    tuple[int, int]
        Figure size as (width, height) in pixels.
    """
    return (figsize[0] * INCHES_TO_PIXELS, figsize[1] * INCHES_TO_PIXELS)


def normalize_magnitude(mag: np.ndarray, magerr: np.ndarray) -> np.ndarray:
    """Normalize magnitude using median-based scaling.

    Formula: (mag - median(mag)) / median(magerr)

    Parameters
    ----------
    mag : np.ndarray
        Magnitude values.
    magerr : np.ndarray
        Magnitude error values.

    Returns
    -------
    np.ndarray
        Normalized magnitude values.

    Raises
    ------
    ValueError
        If median(magerr) is 0 (division by zero).
    """
    med_err = np.median(magerr)
    if med_err == 0 or np.isclose(med_err, 0):
        raise ValueError("Cannot normalize: median(magerr) is 0")
    return (mag - np.median(mag)) / med_err


def _norm_expr(float32: bool = True) -> pl.Expr:
    """Create Polars expression for normalized magnitude.

    Computes: (mag - median(mag)) / median(magerr) for each row's list.

    Parameters
    ----------
    float32 : bool, default True
        If True, cast result to Float32.

    Returns
    -------
    pl.Expr
        Polars expression that produces a "norm" column.
    """
    mag_centered = pl.col("mag").list.eval(pl.element() - pl.element().median())
    magerr_median = pl.col("magerr").list.eval(pl.element().median()).list.first()
    norm_expr = mag_centered / magerr_median
    if float32:
        norm_expr = norm_expr.list.eval(pl.element().cast(pl.Float32))
    return norm_expr.alias("norm")


def _load_or_migrate_cache(
    cache_path: Path | None,
    prefix: str,
    dataset_len: int,
) -> tuple[Path | None, pl.DataFrame | None]:
    """Check cache and handle migration from old naming scheme.

    Parameters
    ----------
    cache_path : Path or None
        Directory containing cache files. If None, caching is disabled.
    prefix : str
        Cache file prefix (e.g., "main", "additional", "wavelet").
    dataset_len : int
        Number of rows in dataset (used in filename).

    Returns
    -------
    tuple[Path | None, pl.DataFrame | None]
        (cache_file, cached_df) where cached_df is the loaded DataFrame
        if found in cache, otherwise None. cache_file is the path to use
        for saving new results (None if caching disabled).
    """
    if cache_path is None or dataset_len < MIN_ROWS_FOR_CACHING:
        return None, None

    cache_file = cache_path / f"features_{prefix}_{dataset_len}.parquet"

    # Check if cache exists
    if cache_file.exists():
        logger.info(f"[{prefix}] Loading from cache...")
        return cache_file, pl.read_parquet(cache_file, parallel="columns")

    # Migrate old cache file naming scheme if applicable
    old_cache_file = cache_path / f"features_{prefix}.parquet"
    if old_cache_file.exists():
        old_rows = pl.scan_parquet(old_cache_file).select(pl.len()).collect().item()
        if old_rows == dataset_len:
            logger.info(f"[{prefix}] Migrating {old_cache_file.name} -> {cache_file.name}")
            old_cache_file.rename(cache_file)
            return cache_file, pl.read_parquet(cache_file, parallel="columns")

    return cache_file, None


def view_series(
    df: DataFrameType | dict,
    index: int,
    figsize: tuple[int, int] = (8, 4),
    advanced: bool = False,
    singlepoint_min_outlying_factor: float | None = None,
    verbose: int = 0,
    backend: str = "plotly",
    plot_file: str | None = None,
    title: str | None = None,
    show: bool = True,
):
    """
    Plot light curve with error bars.

    Displays magnitude vs UTC date/time with error bars.

    Parameters
    ----------
    df : polars.DataFrame, pandas.DataFrame, or dict
        DataFrame or dict containing 'mag', 'magerr', 'mjd', and 'class' columns.
    index : int
        Index of the record to plot.
    figsize : tuple[int, int], default (8, 4)
        Figure size as (width, height) in inches.
    advanced : bool, default False
        If True, also displays distribution plots for mjd_diff (with kurtosis)
        and normalized magnitude (with mean and skewness).
    singlepoint_min_outlying_factor : float or None, default None
        If specified, excludes single-point outliers from the plot.
        An outlier is defined as a point where mag < Q1 - factor*IQR or
        mag > Q3 + factor*IQR. Only excludes if exactly 1 such point exists.
    verbose : int, default 0
        If > 0, log info messages about outlier exclusion.
    backend : str, default "plotly"
        Plotting backend: "plotly" or "matplotlib".
    plot_file : str or None, default None
        If specified, save the figure to this file path.
    title : str or None, default None
        Custom title for the plot. If None, generates default title with
        record index, class, and npoints.
    show : bool, default True
        If True, display the plot interactively. Set to False to only save
        to file without displaying.

    Returns
    -------
    Figure or tuple of Figures
        If advanced=False, returns the main light curve figure.
        If advanced=True, returns (main_fig, mjd_diff_fig, norm_fig).
        Figure type depends on backend (go.Figure for plotly, plt.Figure for matplotlib).
    """
    row = _get_row(df, index)

    mjd = np.array(row["mjd"])
    mag = np.array(row["mag"])
    magerr = np.array(row["magerr"])
    cls = row["class"]
    row_id = row.get("id", None)

    # Exclude single-point outliers if requested
    if singlepoint_min_outlying_factor is not None:
        q1, q3 = np.percentile(mag, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - singlepoint_min_outlying_factor * iqr
        upper_bound = q3 + singlepoint_min_outlying_factor * iqr
        outlier_mask = (mag < lower_bound) | (mag > upper_bound)
        n_outliers = np.sum(outlier_mask)

        if n_outliers == 1:
            outlier_idx = np.where(outlier_mask)[0][0]
            outlier_mag = mag[outlier_idx]
            outlier_mjd = mjd[outlier_idx]
            if verbose > 0:
                id_str = f", id={row_id}" if row_id is not None else ""
                logger.info(
                    f"[view_series] Record #{index}{id_str}: Excluding single-point outlier at idx={outlier_idx}, "
                    f"mag={outlier_mag:.4f}, mjd={outlier_mjd:.4f} "
                    f"(bounds: [{lower_bound:.4f}, {upper_bound:.4f}], factor={singlepoint_min_outlying_factor})"
                )
            # Exclude the outlier
            keep_mask = ~outlier_mask
            mjd = mjd[keep_mask]
            mag = mag[keep_mask]
            magerr = magerr[keep_mask]

    # Convert MJD to UTC datetime
    utc_dates = [MJD_EPOCH + pd.Timedelta(days=float(m)) for m in mjd]

    # Use provided title or generate default
    if title is None:
        title = f"Record #{index} — class: {cls}, npoints: {len(mag)}"

    if backend == "matplotlib":
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        ax.errorbar(utc_dates, mag, yerr=magerr, fmt="o", markersize=4, capsize=2)
        ax.set_title(title)
        ax.set_xlabel("Date (UTC)")
        ax.set_ylabel("mag")
        ax.invert_yaxis()  # Brighter = lower magnitude
        fig.autofmt_xdate()

        if plot_file:
            fig.savefig(plot_file, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

        if not advanced:
            return fig

        # Distribution plot for mjd_diff
        mjd_diff = np.diff(mjd)
        mjd_diff_kurtosis = stats.kurtosis(mjd_diff)

        fig_mjd_diff, ax2 = plt.subplots(figsize=figsize)
        ax2.hist(mjd_diff, bins=HISTOGRAM_BINS)
        ax2.set_title(f"mjd_diff distribution — kurtosis: {mjd_diff_kurtosis:.3f}")
        ax2.set_xlabel("mjd_diff (days)")
        ax2.set_ylabel("count")
        if show:
            plt.show()
        else:
            plt.close(fig_mjd_diff)

        # Distribution plot for norm
        norm = (mag - np.median(mag)) / np.median(magerr)
        norm_mean = np.mean(norm)
        norm_skewness = stats.skew(norm)

        fig_norm, ax3 = plt.subplots(figsize=figsize)
        ax3.hist(norm, bins=HISTOGRAM_BINS)
        ax3.set_title(f"norm distribution — mean: {norm_mean:.3f}, skewness: {norm_skewness:.3f}")
        ax3.set_xlabel("norm")
        ax3.set_ylabel("count")
        if show:
            plt.show()
        else:
            plt.close(fig_norm)

        return fig, fig_mjd_diff, fig_norm

    # Default: plotly backend
    width, height = _figsize_to_pixels(figsize)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=utc_dates,
            y=mag,
            mode="markers",
            name="mag",
            error_y={"type": "data", "array": magerr, "visible": True},
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date (UTC)",
        yaxis_title="mag",
        width=width,
        height=height,
    )

    # Invert y-axis (brighter = lower magnitude in astronomy)
    fig.update_yaxes(autorange="reversed")

    if plot_file:
        fig.write_image(plot_file)

    if show:
        fig.show()

    if not advanced:
        return fig

    # Distribution plot for mjd_diff
    mjd_diff = np.diff(mjd)
    mjd_diff_kurtosis = stats.kurtosis(mjd_diff)

    fig_mjd_diff = go.Figure()
    fig_mjd_diff.add_trace(go.Histogram(x=mjd_diff, name="mjd_diff", nbinsx=30))
    fig_mjd_diff.update_layout(
        title=f"mjd_diff distribution — kurtosis: {mjd_diff_kurtosis:.3f}",
        xaxis_title="mjd_diff (days)",
        yaxis_title="count",
        width=width,
        height=height,
    )
    if show:
        fig_mjd_diff.show()

    # Distribution plot for norm
    norm = (mag - np.median(mag)) / np.median(magerr)
    norm_mean = np.mean(norm)
    norm_skewness = stats.skew(norm)

    fig_norm = go.Figure()
    fig_norm.add_trace(go.Histogram(x=norm, name="norm", nbinsx=30))
    fig_norm.update_layout(
        title=f"norm distribution — mean: {norm_mean:.3f}, skewness: {norm_skewness:.3f}",
        xaxis_title="norm",
        yaxis_title="count",
        width=width,
        height=height,
    )
    if show:
        fig_norm.show()

    return fig, fig_mjd_diff, fig_norm


def norm_series(
    df: DataFrameType | dict,
    index: int,
    figsize: tuple[int, int] = (8, 4),
) -> None:
    """
    Plot normalized magnitude series for a given record.

    Displays a dual-axis plot showing raw magnitude values and their
    median-normalized counterparts scaled by median error.

    Parameters
    ----------
    df : polars.DataFrame, pandas.DataFrame, or dict
        DataFrame or dict containing 'mag', 'magerr', and 'class' columns.
        Each row represents a light curve observation.
    index : int
        Index of the record to plot.
    figsize : tuple[int, int], default (8, 4)
        Figure size as (width, height) in inches.

    Notes
    -----
    Normalization formula: (mag - median(mag)) / median(magerr)

    The correlation coefficient between raw and normalized magnitude
    is displayed in the plot title.
    """
    row = _get_row(df, index)

    mag = np.array(row["mag"])
    magerr = np.array(row["magerr"])
    cls = row["class"]

    norm = normalize_magnitude(mag, magerr)
    correlation = np.corrcoef(mag, norm)[0, 1]

    width, height = _figsize_to_pixels(figsize)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(y=mag, mode="lines", name="mag"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=norm, mode="lines", name="norm"),
        secondary_y=True,
    )

    fig.update_layout(
        title=f"Record #{index} — class: {cls}, corr={correlation:.3f}",
        width=width,
        height=height,
        legend={"x": 0.01, "y": 0.99},
    )
    fig.update_yaxes(title_text="mag", secondary_y=False)
    fig.update_yaxes(title_text="norm", secondary_y=True)

    fig.show()


def compare_classes(
    df: DataFrameType,
    figsize: tuple[int, int] = (8, 4),
) -> None:
    """
    Compare random samples from each class by plotting their normalized series.

    Randomly selects one sample from class 0 (non-flare) and one from class 1
    (flare), then displays their normalized magnitude series side by side.

    Parameters
    ----------
    df : polars.DataFrame or pandas.DataFrame
        DataFrame containing 'mag', 'magerr', and 'class' columns.
    figsize : tuple[int, int], default (8, 4)
        Figure size as (width, height) in inches.

    Raises
    ------
    ValueError
        If no samples exist for class 0 or class 1.

    See Also
    --------
    norm_series : Plot a single normalized magnitude series.
    """
    if isinstance(df, pl.DataFrame):
        class_0 = df.filter(pl.col("class") == 0)
        class_1 = df.filter(pl.col("class") == 1)
        if class_0.height == 0:
            raise ValueError("No samples found with class=0 (non-flare)")
        if class_1.height == 0:
            raise ValueError("No samples found with class=1 (flare)")
        non_flare_idx = class_0.with_row_index().sample(1)["index"][0]
        flare_idx = class_1.with_row_index().sample(1)["index"][0]
    else:
        class_0 = df[df["class"] == 0]
        class_1 = df[df["class"] == 1]
        if len(class_0) == 0:
            raise ValueError("No samples found with class=0 (non-flare)")
        if len(class_1) == 0:
            raise ValueError("No samples found with class=1 (flare)")
        non_flare_idx = class_0.sample(1).index[0]
        flare_idx = class_1.sample(1).index[0]

    norm_series(df, non_flare_idx, figsize)
    norm_series(df, flare_idx, figsize)


def extract_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract statistical features from light curve data for ML.

    .. deprecated::
        Use :func:`extract_features_polars` instead for better performance.
        This function iterates row-by-row and is significantly slower.

    Computes statistical features for mag, magerr, and normalized magnitude
    arrays for each record in the dataset.

    Parameters
    ----------
    df : polars.DataFrame
        DataFrame with columns: id, class, mjd, mag, magerr.
        mag and magerr must be List(Float64) columns.

    Returns
    -------
    polars.DataFrame
        DataFrame with columns: id, class, npoints, and statistical features
        prefixed by array name (mag_, magerr_, norm_).

    Features per array
    ------------------
    - std: Standard deviation
    - skewness: Fisher-Pearson skewness
    - kurtosis: Fisher kurtosis (excess)
    - frac_below_3sigma: Fraction of points below -3*std
    - amplitude_sigma: Peak-to-peak amplitude
    - mean: Arithmetic mean
    - median: Median value
    """
    warnings.warn(
        "extract_features() is deprecated, use extract_features_polars() instead",
        DeprecationWarning,
        stacklevel=2,
    )

    def compute_array_features(arr: np.ndarray, prefix: str) -> dict[str, float]:
        std = np.std(arr)
        return {
            f"{prefix}_std": std,
            f"{prefix}_skewness": stats.skew(arr),
            f"{prefix}_kurtosis": stats.kurtosis(arr),
            f"{prefix}_frac_below_3sigma": np.mean(arr < -3 * std),
            f"{prefix}_amplitude_sigma": np.ptp(arr),
            f"{prefix}_mean": np.mean(arr),
            f"{prefix}_median": np.median(arr),
        }

    records: list[dict] = []
    for row in df.iter_rows(named=True):
        mag = np.array(row["mag"])
        magerr = np.array(row["magerr"])
        norm = normalize_magnitude(mag, magerr)

        record = {
            "id": row["id"],
            "class": row["class"],
            "npoints": len(mag),
        }
        record.update(compute_array_features(mag, "mag"))
        record.update(compute_array_features(magerr, "magerr"))
        record.update(compute_array_features(norm, "norm"))

        records.append(record)

    return pl.DataFrame(records)


def extract_features_polars(
    df: pl.DataFrame,
    normalize: str | None = None,
    float32: bool = True,
    engine: str = DEFAULT_ENGINE,
) -> pl.DataFrame:
    """
    Extract statistical features using native Polars operations (parallelized).

    Vectorized version of extract_features() that uses Polars' list operations
    for parallel computation across all rows.

    Parameters
    ----------
    df : polars.DataFrame
        DataFrame with optional columns: 'id', 'class', 'mag', 'magerr', 'mjd'.
        List columns must be List(Float64).
    float32 : bool, default True
        If True, cast float columns to Float32 to save memory.
    normalize : str or None, default None
        Normalization method to apply before computing statistics:
        - None: No normalization (raw values)
        - "minmax": (x - min) / (max - min), scales to [0, 1]
        - "zscore": (x - mean) / std, standardizes to mean=0, std=1
    engine : str, default "streaming"
        Polars execution engine: "streaming" for memory-efficient processing,
        "eager" for standard in-memory execution.

    Returns
    -------
    polars.DataFrame
        DataFrame with 'id', optional 'class', 'npoints', and statistical
        features for available columns (mag_, magerr_, norm_, mjd_diff_).
        - norm_ features require both 'mag' and 'magerr'
        - mjd_diff_ features require 'mjd'

    Features per array
    ------------------
    - mean, std, min, max, median, q25, q75: Basic statistics
    - skewness, kurtosis, entropy (except norm): Distribution shape
    - first, last, arg_min, arg_max: Positional
    - n_unique, trend_changes: Uniqueness & structure

    Raises
    ------
    ValueError
        If engine is not in possible_engines, or if normalize method is unknown.
    """
    possible_engines = ("streaming", "cpu", "gpu")
    if engine not in possible_engines:
        raise ValueError(f"engine must be in {possible_engines}, got '{engine}'")

    def normalize_expr(c: pl.Expr) -> pl.Expr:
        if normalize is None:
            return c
        elif normalize == "minmax":
            # Handle division by zero when max == min
            range_val = c.list.max() - c.list.min()
            return pl.when(range_val == 0).then(0.0).otherwise((c - c.list.min()) / range_val)
        elif normalize == "zscore":
            return (c - c.list.mean()) / c.list.std()
        else:
            raise ValueError(f"Unknown normalize method: {normalize}")

    def stats_exprs(col: str) -> list[pl.Expr]:
        c = normalize_expr(pl.col(col))
        exprs = [
            # Basic statistics
            c.list.mean().alias(f"{col}_mean"),
            c.list.std().alias(f"{col}_std"),
            c.list.min().alias(f"{col}_min"),
            c.list.max().alias(f"{col}_max"),
            c.list.eval(pl.element().median()).list.first().alias(f"{col}_median"),
            c.list.eval(pl.element().quantile(0.25)).list.first().alias(f"{col}_q25"),
            c.list.eval(pl.element().quantile(0.75)).list.first().alias(f"{col}_q75"),
            # Distribution shape
            c.list.eval(pl.element().skew()).list.first().alias(f"{col}_skewness"),
            c.list.eval(pl.element().kurtosis()).list.first().alias(f"{col}_kurtosis"),
        ]
        # Entropy only for non-negative columns (not norm which can be negative)
        if col != "norm":
            exprs.append(c.list.eval(pl.element().entropy()).list.first().alias(f"{col}_entropy"))
        exprs.extend(
            [
                # Positional
                c.list.first().alias(f"{col}_first"),
                c.list.last().alias(f"{col}_last"),
                c.list.arg_min().alias(f"{col}_arg_min"),
                c.list.arg_max().alias(f"{col}_arg_max"),
                # Uniqueness & structure
                c.list.n_unique().alias(f"{col}_n_unique"),
                c.list.eval(pl.element().diff().sign().diff().ne(0).sum()).list.first().alias(f"{col}_trend_changes"),
            ]
        )
        return exprs

    cols = set(df.columns)
    has_mag = "mag" in cols
    has_magerr = "magerr" in cols
    has_mjd = "mjd" in cols
    has_class = "class" in cols
    has_norm = "norm" in cols  # norm can be passed directly or derived
    has_mjd_diff = "mjd_diff" in cols  # mjd_diff can be passed directly or derived

    # Build derived columns (only if not already present)
    derived_exprs = []
    if has_mag and has_magerr and not has_norm:
        derived_exprs.append(_norm_expr(float32=False))  # Cast happens at end
        has_norm = True
    if has_mjd and not has_mjd_diff:
        mjd_diff_expr = pl.col("mjd").list.eval(pl.element().diff().drop_nulls())
        if float32:
            mjd_diff_expr = mjd_diff_expr.list.eval(pl.element().cast(pl.Float32))
        derived_exprs.append(mjd_diff_expr.alias("mjd_diff"))
        has_mjd_diff = True

    df_enriched = df.with_columns(derived_exprs) if derived_exprs else df

    # Build select expressions
    select_exprs: list[pl.Expr | str] = []
    if "id" in cols:
        select_exprs.append("id")
    if has_class:
        select_exprs.append("class")

    # npoints from first available list column
    for len_col in ("mag", "magerr", "mjd", "norm", "mjd_diff"):
        if len_col in cols or (len_col == "norm" and has_norm) or (len_col == "mjd_diff" and has_mjd_diff):
            select_exprs.append(pl.col(len_col).list.len().alias("npoints"))
            break

    # Add stats for each available column
    if has_mag:
        select_exprs.extend(stats_exprs("mag"))
    if has_magerr:
        select_exprs.extend(stats_exprs("magerr"))
    if has_norm:
        select_exprs.extend(stats_exprs("norm"))
    if has_mjd_diff:
        select_exprs.extend(stats_exprs("mjd_diff"))

    # Execute with specified engine (always use lazy API)
    result = df_enriched.lazy().select(select_exprs).collect(engine=engine)

    if float32:
        result = result.cast({c: pl.Float32 for c in result.columns if result[c].dtype == pl.Float64})
    return result


def extract_features_sparingly(
    dataset: Dataset,
    normalize: str | None = None,
    float32: bool = True,
    engine: str = DEFAULT_ENGINE,
    cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
) -> pl.DataFrame:
    """
    Extract features from HuggingFace Dataset with minimal RAM usage.

    Processes data in batches to avoid loading the entire dataset into memory.
    Caches final results to disk as parquet files.

    Parameters
    ----------
    dataset : datasets.Dataset
        HuggingFace Dataset with columns: id, class, mag, magerr, mjd.
    normalize : str or None, default None
        Normalization method passed to extract_features_polars.
    float32 : bool, default True
        If True, cast float columns to Float32 to save memory.
    engine : str, default "streaming"
        Polars execution engine: "streaming" for memory-efficient processing,
        "eager" for standard in-memory execution.
    cache_dir : str, Path, or None, default "data"
        Directory for caching intermediate parquet files. If None, caching is disabled.

    Returns
    -------
    polars.DataFrame
        DataFrame with id, class, npoints, ts (UTC timestamp from max mjd),
        and statistical features for mag, magerr, norm, mjd_diff.
    """
    # Setup cache directory
    cache_path = Path(cache_dir) if cache_dir else None
    if cache_path:
        cache_path.mkdir(parents=True, exist_ok=True)

    dataset_len = len(dataset)

    # Check cache
    cache_file, cached_df = _load_or_migrate_cache(cache_path, "main", dataset_len)
    if cached_df is not None:
        return cached_df

    # Determine which columns to process
    has_id = "id" in dataset.column_names
    has_class = "class" in dataset.column_names
    has_mag = "mag" in dataset.column_names
    has_magerr = "magerr" in dataset.column_names
    has_mjd = "mjd" in dataset.column_names
    has_norm = has_mag and has_magerr

    # Build list of columns to load per batch
    cols_to_load: list[str] = []
    if has_id:
        cols_to_load.append("id")
    if has_class:
        cols_to_load.append("class")
    if has_mag:
        cols_to_load.append("mag")
    if has_magerr:
        cols_to_load.append("magerr")
    if has_mjd:
        cols_to_load.append("mjd")

    # =========================================================================
    # Batch processing
    # =========================================================================
    results: list[pl.DataFrame] = []

    for start in tqdm(range(0, dataset_len, DEFAULT_BATCH_SIZE), desc="main features", unit="batch"):
        end = min(start + DEFAULT_BATCH_SIZE, dataset_len)

        # Load batch
        batch = dataset.select(range(start, end))
        df = batch.select_columns(cols_to_load).to_polars()

        # Compute norm if we have mag and magerr
        if has_norm:
            df = df.with_columns(_norm_expr(float32))

        # Extract features for each column type
        batch_features: list[pl.DataFrame] = []

        # Meta columns (id, class)
        meta_cols = []
        if has_id:
            meta_cols.append("id")
        if has_class:
            meta_cols.append("class")
        if meta_cols:
            batch_features.append(df.select(meta_cols))

        # Compute features using extract_features_polars for each column group
        if has_mag:
            mag_features = extract_features_polars(df.select("mag"), normalize=normalize, float32=float32, engine=engine)
            batch_features.append(mag_features)
            del mag_features

        if has_magerr:
            magerr_features = extract_features_polars(df.select("magerr"), normalize=normalize, float32=float32, engine=engine)
            # Drop npoints if already present
            if "npoints" in magerr_features.columns and any("npoints" in f.columns for f in batch_features):
                magerr_features = magerr_features.drop("npoints")
            batch_features.append(magerr_features)
            del magerr_features

        if has_norm:
            norm_features = extract_features_polars(df.select("norm"), normalize=normalize, float32=float32, engine=engine)
            if "npoints" in norm_features.columns and any("npoints" in f.columns for f in batch_features):
                norm_features = norm_features.drop("npoints")
            batch_features.append(norm_features)
            del norm_features

        if has_mjd:
            # Cast mjd to float32 if requested
            df_mjd = df.select("mjd")
            if float32:
                df_mjd = df_mjd.with_columns(pl.col("mjd").list.eval(pl.element().cast(pl.Float32)))
            mjd_features = extract_features_polars(df_mjd, normalize=normalize, float32=float32, engine=engine)
            if "npoints" in mjd_features.columns and any("npoints" in f.columns for f in batch_features):
                mjd_features = mjd_features.drop("npoints")
            batch_features.append(mjd_features)
            del df_mjd, mjd_features

            # Compute ts = UTC timestamp from max(mjd)
            ts_col = (
                df.select(pl.col("mjd").list.max().alias("mjd_max"))
                .with_columns((pl.lit(MJD_EPOCH) + pl.duration(days=pl.col("mjd_max"))).alias("ts"))
                .select("ts")
            )
            batch_features.append(ts_col)
            del ts_col

        # Combine batch features
        batch_result = pl.concat(batch_features, how="horizontal")
        results.append(batch_result)

        del df, batch, batch_features, batch_result
        clean_ram()

    # =========================================================================
    # Combine batches
    # =========================================================================
    result = pl.concat(results)
    del results
    clean_ram()

    # Cache result
    if cache_file:
        result.write_parquet(cache_file, compression="zstd")
        logger.info(f"[main] Saved to {cache_file}")

    return result


def _get_additional_feature_exprs(
    col_name: str = "norm",
    prefix: str | None = None,
    include_mjd_features: bool = True,
) -> list[pl.Expr]:
    """Return list of Polars expressions for additional flare-specific features.

    Reusable by both extract_additional_features_sparingly and _process_all_chunk.
    Can generate features for any column (norm, velocity, etc.).

    Parameters
    ----------
    col_name : str, default "norm"
        Column name to compute features from (e.g., "norm", "velocity").
    prefix : str or None, default None
        Prefix for feature names. If None, uses col_name.
        Example: prefix="vel" for velocity column.
    include_mjd_features : bool, default True
        If True, include mjd-dependent features (rise_decay_time_ratio, mjd_span).
        Set to False for columns like velocity where mjd features don't apply.

    Returns
    -------
    list[pl.Expr]
        List of Polars expressions for all additional features.
    """
    col = pl.col(col_name)
    if prefix is None:
        prefix = col_name

    # Feature expressions
    n_below_3sigma = col.list.eval((pl.element() < -3).cast(pl.Int32).sum()).list.first().alias(f"{prefix}_n_below_3sigma")

    max_consecutive = (
        col.list.eval(
            (
                (pl.element() < -2).cast(pl.Int32).cum_sum()
                - pl.when(pl.element() >= -2).then((pl.element() < -2).cast(pl.Int32).cum_sum()).otherwise(None).forward_fill().fill_null(0)
            ).max()
        )
        .list.first()
        .fill_null(0)
        .alias(f"{prefix}_max_consecutive_below_2sigma")
    )

    npoints = col.list.len()
    peak_idx = col.list.arg_min()
    rise_decay_idx_ratio = ((peak_idx.cast(pl.Float32) + 1.0) / (npoints - peak_idx).cast(pl.Float32)).alias(f"{prefix}_rise_decay_idx_ratio")

    n_local_minima = (
        col.list.eval(((pl.element().diff() < 0).cast(pl.Int32) * (pl.element().diff().shift(-1) > 0).cast(pl.Int32)).sum())
        .list.first()
        .fill_null(0)
        .alias(f"{prefix}_n_local_minima")
    )

    n_zero_crossings = (
        col.list.eval(((pl.element().sign() * pl.element().shift(1).sign()) < 0).cast(pl.Int32).sum())
        .list.first()
        .fill_null(0)
        .alias(f"{prefix}_n_zero_crossings")
    )

    # MJD-dependent features (only for norm, not velocity)
    if include_mjd_features:
        mjd_first = pl.col("mjd").list.first()
        mjd_last = pl.col("mjd").list.last()
        mjd_at_peak = pl.col("mjd").list.get(col.list.arg_min())
        rise_time = mjd_at_peak - mjd_first
        decay_time = mjd_last - mjd_at_peak
        rise_decay_time_ratio = (rise_time / (decay_time + EPSILON)).alias(f"{prefix}_rise_decay_time_ratio")
        mjd_span = (mjd_last - mjd_first).alias("mjd_span")

    # =========================================================================
    # Additional shape features for flare detection
    # =========================================================================

    # 1. Peak position ratio: 0.1 = near start (fast rise), 0.9 = near end
    peak_position_ratio = (peak_idx.cast(pl.Float32) / (npoints - 1).cast(pl.Float32).clip(1, None)).alias(f"{prefix}_peak_position_ratio")

    # 2. Max consecutive ABOVE threshold (artifact detection)
    max_consecutive_above_1sigma = (
        col.list.eval(
            (
                (pl.element() > 1).cast(pl.Int32).cum_sum()
                - pl.when(pl.element() <= 1).then((pl.element() > 1).cast(pl.Int32).cum_sum()).otherwise(None).forward_fill().fill_null(0)
            ).max()
        )
        .list.first()
        .fill_null(0)
        .alias(f"{prefix}_max_consecutive_above_1sigma")
    )

    # 3. Longest monotonic runs (slow decay signature)
    _longest_inc_expr = (
        col.list.eval(
            (
                (pl.element().diff() > 0).cast(pl.Int32).cum_sum()
                - pl.when(pl.element().diff() <= 0).then((pl.element().diff() > 0).cast(pl.Int32).cum_sum()).otherwise(None).forward_fill().fill_null(0)
            ).max()
        )
        .list.first()
        .fill_null(0)
    )
    _longest_dec_expr = (
        col.list.eval(
            (
                (pl.element().diff() < 0).cast(pl.Int32).cum_sum()
                - pl.when(pl.element().diff() >= 0).then((pl.element().diff() < 0).cast(pl.Int32).cum_sum()).otherwise(None).forward_fill().fill_null(0)
            ).max()
        )
        .list.first()
        .fill_null(0)
    )
    longest_monotonic_increase = _longest_inc_expr.alias(f"{prefix}_longest_monotonic_increase")
    longest_monotonic_decrease = _longest_dec_expr.alias(f"{prefix}_longest_monotonic_decrease")

    # 4. Isolated outliers: single-point dips (artifacts) vs clustered dips (flares)
    n_isolated_below_2sigma = (
        col.list.eval(
            (
                (pl.element() < -2).cast(pl.Int32)
                * (pl.element().shift(1).fill_null(0) >= -2).cast(pl.Int32)
                * (pl.element().shift(-1).fill_null(0) >= -2).cast(pl.Int32)
            ).sum()
        )
        .list.first()
        .fill_null(0)
        .alias(f"{prefix}_n_isolated_below_2sigma")
    )

    # 5. Fraction of points beyond thresholds (density metrics)
    frac_below_2sigma = col.list.eval((pl.element() < -2).cast(pl.Float32).mean()).list.first().fill_null(0.0).alias(f"{prefix}_frac_below_2sigma")

    frac_below_1sigma = col.list.eval((pl.element() < -1).cast(pl.Float32).mean()).list.first().fill_null(0.0).alias(f"{prefix}_frac_below_1sigma")

    # 6. Local maxima count (noise indicator)
    n_local_maxima = (
        col.list.eval(((pl.element().diff() > 0).cast(pl.Int32) * (pl.element().diff().shift(-1) < 0).cast(pl.Int32)).sum())
        .list.first()
        .fill_null(0)
        .alias(f"{prefix}_n_local_maxima")
    )

    # 7. Peak depth: how much lower is min compared to mean
    peak_depth = (col.list.mean() - col.list.min()).alias(f"{prefix}_peak_depth")

    # 8. First vs second half comparison (pre-flare vs flare+decay)
    half_len = npoints // 2
    first_half_mean = col.list.head(half_len).list.mean()
    second_half_mean = col.list.tail(half_len).list.mean()
    half_diff = (first_half_mean - second_half_mean).fill_null(0.0).alias(f"{prefix}_first_second_half_diff")

    # 9. Ratio metrics combining existing computations
    minima_maxima_ratio = (
        (
            (
                col.list.eval(((pl.element().diff() < 0).cast(pl.Int32) * (pl.element().diff().shift(-1) > 0).cast(pl.Int32)).sum())
                .list.first()
                .cast(pl.Float32)
                + 1.0
            )
            / (
                col.list.eval(((pl.element().diff() > 0).cast(pl.Int32) * (pl.element().diff().shift(-1) < 0).cast(pl.Int32)).sum())
                .list.first()
                .cast(pl.Float32)
                + 1.0
            )
        )
        .fill_null(1.0)
        .alias(f"{prefix}_minima_maxima_ratio")
    )

    # 10. Monotonic ratio: increase/decrease balance
    monotonic_ratio = (
        ((_longest_inc_expr.cast(pl.Float32) + 1.0) / (_longest_dec_expr.cast(pl.Float32) + 1.0)).fill_null(1.0).alias(f"{prefix}_monotonic_ratio")
    )

    # =========================================================================
    # 11. Run statistics: n_runs, mean_run_length, total_in_runs
    # =========================================================================

    _n_runs_below_2sigma_expr = (
        col.list.eval(((pl.element() < -2).cast(pl.Int32) > (pl.element().shift(1).fill_null(0) < -2).cast(pl.Int32)).sum()).list.first().fill_null(0)
    )
    n_runs_below_2sigma = _n_runs_below_2sigma_expr.alias(f"{prefix}_n_runs_below_2sigma")

    _total_below_2sigma_expr = col.list.eval((pl.element() < -2).cast(pl.Int32).sum()).list.first().fill_null(0)

    mean_run_below_2sigma = (
        (_total_below_2sigma_expr.cast(pl.Float32) / (_n_runs_below_2sigma_expr.cast(pl.Float32).clip(1, None)))
        .fill_null(0.0)
        .alias(f"{prefix}_mean_run_below_2sigma")
    )

    _n_runs_above_1sigma_expr = (
        col.list.eval(((pl.element() > 1).cast(pl.Int32) > (pl.element().shift(1).fill_null(0) > 1).cast(pl.Int32)).sum()).list.first().fill_null(0)
    )
    n_runs_above_1sigma = _n_runs_above_1sigma_expr.alias(f"{prefix}_n_runs_above_1sigma")

    _total_above_1sigma_expr = col.list.eval((pl.element() > 1).cast(pl.Int32).sum()).list.first().fill_null(0)

    mean_run_above_1sigma = (
        (_total_above_1sigma_expr.cast(pl.Float32) / (_n_runs_above_1sigma_expr.cast(pl.Float32).clip(1, None)))
        .fill_null(0.0)
        .alias(f"{prefix}_mean_run_above_1sigma")
    )

    run_ratio = (
        (
            ((_total_below_2sigma_expr.cast(pl.Float32) + 1.0) / (_n_runs_below_2sigma_expr.cast(pl.Float32).clip(1, None)))
            / ((_total_above_1sigma_expr.cast(pl.Float32) + 1.0) / (_n_runs_above_1sigma_expr.cast(pl.Float32).clip(1, None)))
        )
        .fill_null(1.0)
        .alias(f"{prefix}_run_length_ratio")
    )

    # Build result list
    result = [
        # Original features
        n_below_3sigma,
        max_consecutive,
        rise_decay_idx_ratio,
        n_local_minima,
        n_zero_crossings,
    ]

    # MJD-dependent features only when requested
    if include_mjd_features:
        result.extend([rise_decay_time_ratio, mjd_span])

    result.extend(
        [
            # New features: shape and position
            peak_position_ratio,
            max_consecutive_above_1sigma,
            longest_monotonic_increase,
            longest_monotonic_decrease,
            n_isolated_below_2sigma,
            frac_below_2sigma,
            frac_below_1sigma,
            n_local_maxima,
            peak_depth,
            half_diff,
            minima_maxima_ratio,
            monotonic_ratio,
            # New features: run statistics
            n_runs_below_2sigma,
            mean_run_below_2sigma,
            n_runs_above_1sigma,
            mean_run_above_1sigma,
            run_ratio,
        ]
    )

    return result


def _get_argextremum_stats_exprs(
    index_col: str = "mag",
    stats_cols: list[str] | None = None,
    compute_additional: bool = False,
    compute_argmin_stats: bool = True,
    compute_argmax_stats: bool = False,
) -> list[pl.Expr]:
    """Generate expressions for statistics on sub-series split by argmax/argmin.

    For each stats_col, computes statistics on sub-series split at the
    argmax and/or argmin position of the index column.

    Parameters
    ----------
    index_col : str, default "mag"
        Column to find argmax/argmin in (determines split point).
    stats_cols : list[str] or None
        Columns to compute statistics on. If None, uses [index_col].
    compute_additional : bool, default False
        If True, compute additional statistics (skewness, kurtosis, quantiles,
        sigma counts) in addition to basic stats.
    compute_argmin_stats : bool, default True
        If True, compute stats for sub-series split at argmin:
        - series[:argmin(index_col)] - "to_argmin" prefix
        - series[argmin(index_col):] - "from_argmin" prefix
    compute_argmax_stats : bool, default False
        If True, compute stats for sub-series split at argmax:
        - series[:argmax(index_col)] - "to_argmax" prefix
        - series[argmax(index_col):] - "from_argmax" prefix

    Returns
    -------
    list[pl.Expr]
        List of Polars expressions for sub-series statistics.

    Notes
    -----
    For stellar flare detection, argmin (brightest point in magnitude system)
    is typically more relevant than argmax, hence the default values.
    """
    if not compute_argmin_stats and not compute_argmax_stats:
        return []

    if stats_cols is None:
        stats_cols = [index_col]

    idx_col = pl.col(index_col)
    list_len = idx_col.list.len()

    result: list[pl.Expr] = []

    for col_name in stats_cols:
        col = pl.col(col_name)

        # Build list of slices to compute based on flags
        slices: list[tuple] = []

        if compute_argmax_stats:
            argmax_idx = idx_col.list.arg_max()
            # to_argmax: elements [0, argmax)
            to_argmax = col.list.head(argmax_idx)
            # from_argmax: elements [argmax, end]
            from_argmax = col.list.tail(list_len - argmax_idx)
            slices.extend(
                [
                    (to_argmax, f"{col_name}_to_argmax"),
                    (from_argmax, f"{col_name}_from_argmax"),
                ]
            )

        if compute_argmin_stats:
            argmin_idx = idx_col.list.arg_min()
            # to_argmin: elements [0, argmin)
            to_argmin = col.list.head(argmin_idx)
            # from_argmin: elements [argmin, end]
            from_argmin = col.list.tail(list_len - argmin_idx)
            slices.extend(
                [
                    (to_argmin, f"{col_name}_to_argmin"),
                    (from_argmin, f"{col_name}_from_argmin"),
                ]
            )

        for subseries, prefix in slices:
            # Basic statistics (always computed)
            result.extend(
                [
                    subseries.list.len().alias(f"{prefix}_len"),
                    subseries.list.mean().alias(f"{prefix}_mean"),
                    subseries.list.std().alias(f"{prefix}_std"),
                    subseries.list.min().alias(f"{prefix}_min"),
                    subseries.list.max().alias(f"{prefix}_max"),
                    (subseries.list.max() - subseries.list.min()).alias(f"{prefix}_range"),
                    # Slope proxy: (last - first) / len
                    ((subseries.list.last() - subseries.list.first()) / subseries.list.len().cast(pl.Float32).clip(1, None)).alias(f"{prefix}_slope"),
                ]
            )

            # Additional statistics (optional)
            if compute_additional:
                # Quantiles and IQR
                q25 = subseries.list.eval(pl.element().quantile(0.25, interpolation="linear")).list.first()
                q75 = subseries.list.eval(pl.element().quantile(0.75, interpolation="linear")).list.first()
                median = subseries.list.eval(pl.element().median()).list.first()

                result.extend(
                    [
                        median.alias(f"{prefix}_median"),
                        q25.alias(f"{prefix}_q25"),
                        q75.alias(f"{prefix}_q75"),
                        (q75 - q25).alias(f"{prefix}_iqr"),
                    ]
                )

                # Skewness and kurtosis
                std_expr = subseries.list.std()
                skewness = subseries.list.eval(((pl.element() - pl.element().mean()) ** 3).mean()).list.first() / (std_expr**3 + EPSILON)
                kurtosis = subseries.list.eval(((pl.element() - pl.element().mean()) ** 4).mean()).list.first() / (std_expr**4 + EPSILON) - 3.0

                result.extend(
                    [
                        skewness.alias(f"{prefix}_skewness"),
                        kurtosis.alias(f"{prefix}_kurtosis"),
                    ]
                )

                # Sigma counts (relative to sub-series mean/std)
                n_above_2sigma = subseries.list.eval((pl.element() > (pl.element().mean() + 2 * pl.element().std())).sum()).list.first()
                n_below_2sigma = subseries.list.eval((pl.element() < (pl.element().mean() - 2 * pl.element().std())).sum()).list.first()
                n_above_3sigma = subseries.list.eval((pl.element() > (pl.element().mean() + 3 * pl.element().std())).sum()).list.first()
                n_below_3sigma = subseries.list.eval((pl.element() < (pl.element().mean() - 3 * pl.element().std())).sum()).list.first()

                result.extend(
                    [
                        n_above_2sigma.alias(f"{prefix}_n_above_2sigma"),
                        n_below_2sigma.alias(f"{prefix}_n_below_2sigma"),
                        n_above_3sigma.alias(f"{prefix}_n_above_3sigma"),
                        n_below_3sigma.alias(f"{prefix}_n_below_3sigma"),
                    ]
                )

                # Energy (sum of squares)
                energy = subseries.list.eval((pl.element() ** 2).sum()).list.first()
                result.append(energy.alias(f"{prefix}_energy"))

    return result


def _clean_single_outlier_chunked(
    df: pl.DataFrame,
    od_col: str = "mag",
    od_iqr: float = 40.0,
    chunk_size: int = 100_000,
) -> pl.DataFrame:
    """Clean single IQR outliers using memory-efficient chunked processing.

    For each row, if exactly one element in the list column lies outside
    [Q1 - od_iqr*IQR, Q3 + od_iqr*IQR], replace it with the average of its
    neighbors. If zero or more than one outlier exists, leave unchanged.

    This implementation processes data in chunks to reduce peak memory usage,
    making it suitable for large datasets that don't fit in memory when exploded.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with a list column specified by od_col.
    od_col : str, default "mag"
        Name of the list column to clean.
    od_iqr : float, default 40.0
        IQR multiplier for outlier detection threshold.
    chunk_size : int, default 100_000
        Number of rows to process at once. Lower = less memory, slower.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with two new columns:
        - {od_col} : cleaned list (replaces original)
        - had_od : bool, True if single outlier was detected and replaced

    Notes
    -----
    Memory usage is approximately: chunk_size * avg_list_length * 8 bytes
    For 100k rows with 50 elements each: ~40MB peak vs ~2GB for full explode.
    """
    n_rows = len(df)
    if n_rows == 0:
        return df.with_columns(pl.lit(False).alias("had_od"))

    # Process in chunks
    results = []
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        chunk = df.slice(start, end - start)

        # Process chunk using the native method (explode/implode)
        # This limits memory to chunk_size * expansion_factor
        chunk_result = _clean_single_outlier_native_impl(chunk, od_col, od_iqr)
        results.append(chunk_result)

    return pl.concat(results)


def _clean_single_outlier_numba(
    df: pl.DataFrame,
    od_col: str = "mag",
    od_iqr: float = 40.0,
    from_row: int | None = None,
    to_row: int | None = None,
) -> pl.DataFrame:
    """Clean single IQR outliers using numba-accelerated row processing.

    Most memory-efficient implementation - processes one row at a time.
    Uses numba JIT compilation for the inner loop to maintain good performance.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with a list column specified by od_col.
    od_col : str, default "mag"
        Name of the list column to clean.
    od_iqr : float, default 40.0
        IQR multiplier for outlier detection threshold.
    from_row : int or None, default None
        If both from_row and to_row are specified, process only rows [from_row:to_row).
    to_row : int or None, default None
        If both from_row and to_row are specified, process only rows [from_row:to_row).

    Returns
    -------
    pl.DataFrame
        Original DataFrame with:
        - {od_col} : cleaned list (replaces original)
        - had_od : bool, True if single outlier was detected and replaced

    Notes
    -----
    Memory usage is O(max_list_length) regardless of dataset size.
    Requires numba for good performance, falls back to pure Python if unavailable.
    """
    try:
        from numba import njit

        HAS_NUMBA = True
    except ImportError:
        HAS_NUMBA = False

    if HAS_NUMBA:

        @njit(cache=True)
        def _process_row_numba(arr: np.ndarray, od_iqr: float) -> tuple:
            """Process single row with numba acceleration."""
            n = len(arr)
            if n < 3:
                return arr.copy(), False

            # Compute quartiles using sorted array
            sorted_arr = np.sort(arr)
            q1_idx = int(0.25 * (n - 1))
            q3_idx = int(0.75 * (n - 1))

            # Linear interpolation for quartiles
            q1_frac = 0.25 * (n - 1) - q1_idx
            q3_frac = 0.75 * (n - 1) - q3_idx

            q1 = sorted_arr[q1_idx] * (1 - q1_frac) + sorted_arr[min(q1_idx + 1, n - 1)] * q1_frac
            q3 = sorted_arr[q3_idx] * (1 - q3_frac) + sorted_arr[min(q3_idx + 1, n - 1)] * q3_frac

            iqr = q3 - q1
            lower = q1 - od_iqr * iqr
            upper = q3 + od_iqr * iqr

            # Find outliers
            outlier_idx = -1
            n_outliers = 0
            for i in range(n):
                if arr[i] < lower or arr[i] > upper:
                    n_outliers += 1
                    outlier_idx = i
                    if n_outliers > 1:
                        break

            # Replace if exactly one outlier
            if n_outliers != 1:
                return arr.copy(), False

            result = arr.copy()
            if outlier_idx == 0:
                result[0] = arr[1]
            elif outlier_idx == n - 1:
                result[n - 1] = arr[n - 2]
            else:
                result[outlier_idx] = (arr[outlier_idx - 1] + arr[outlier_idx + 1]) / 2

            return result, True

        process_row = _process_row_numba
    else:
        # Pure Python fallback
        def process_row(arr: np.ndarray, od_iqr: float) -> tuple:
            n = len(arr)
            if n < 3:
                return arr.copy(), False

            q1 = np.percentile(arr, 25, interpolation="linear")
            q3 = np.percentile(arr, 75, interpolation="linear")
            iqr = q3 - q1
            lower, upper = q1 - od_iqr * iqr, q3 + od_iqr * iqr

            outlier_mask = (arr < lower) | (arr > upper)
            n_outliers = outlier_mask.sum()

            if n_outliers != 1:
                return arr.copy(), False

            result = arr.copy()
            idx = np.argmax(outlier_mask)
            if idx == 0:
                result[0] = arr[1]
            elif idx == n - 1:
                result[-1] = arr[-2]
            else:
                result[idx] = (arr[idx - 1] + arr[idx + 1]) / 2

            return result, True

    # Process each row
    col_data = df[od_col].to_list()
    cleaned = []
    had_od = []

    for row in col_data:
        arr = np.array(row, dtype=np.float64)
        result, was_cleaned = process_row(arr, od_iqr)
        cleaned.append(result.tolist())
        had_od.append(was_cleaned)

    # Build result DataFrame
    other_cols = [c for c in df.columns if c != od_col]
    result_df = df.select(other_cols).with_columns(
        [
            pl.Series(od_col, cleaned),
            pl.Series("had_od", had_od),
        ]
    )

    return result_df


def _clean_single_outlier_native(
    df: pl.DataFrame,
    od_col: str = "mag",
    od_iqr: float = 40.0,
) -> pl.DataFrame:
    """Clean single IQR outliers - memory-efficient chunked implementation.

    This is the recommended entry point. Automatically uses chunked processing
    to limit memory usage while maintaining vectorization benefits.

    For each row, if exactly one element in the list column lies outside
    [Q1 - od_iqr*IQR, Q3 + od_iqr*IQR], replace it with the average of its
    neighbors. If zero or more than one outlier exists, leave unchanged.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with a list column specified by od_col.
    od_col : str, default "mag"
        Name of the list column to clean.
    od_iqr : float, default 40.0
        IQR multiplier for outlier detection threshold.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with two new columns:
        - {od_col} : cleaned list (replaces original)
        - had_od : bool, True if single outlier was detected and replaced

    See Also
    --------
    _clean_single_outlier_chunked : Explicit chunk size control
    _clean_single_outlier_numba : Lowest memory usage (row-by-row)
    _clean_single_outlier_native_legacy : Original explode/implode implementation
    """
    return _clean_single_outlier_chunked(df, od_col, od_iqr, chunk_size=100_000)


def _clean_single_outlier_native_legacy(
    df: pl.DataFrame,
    od_col: str = "mag",
    od_iqr: float = 40.0,
) -> pl.DataFrame:
    """Clean single IQR outliers using native Polars operations (vectorized).

    .. deprecated::
        This function uses the full explode/implode pattern which can consume
        excessive memory on large datasets. Use `_clean_single_outlier_native`
        (chunked) or `_clean_single_outlier_numba` (row-by-row) instead.

    For each row, if exactly one element in the list column lies outside
    [Q1 - od_iqr*IQR, Q3 + od_iqr*IQR], replace it with the average of its
    neighbors. If zero or more than one outlier exists, leave unchanged.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with a list column specified by od_col.
    od_col : str, default "mag"
        Name of the list column to clean.
    od_iqr : float, default 40.0
        IQR multiplier for outlier detection threshold.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with two new columns:
        - {od_col} : cleaned list (replaces original)
        - had_od : bool, True if single outlier was detected and replaced

    Notes
    -----
    Uses explode/implode pattern for native Polars vectorization.
    WARNING: Memory usage is O(n_rows * avg_list_length) which can be
    prohibitive for large datasets. Consider using the chunked version.
    """
    warnings.warn(
        "_clean_single_outlier_native_legacy uses excessive memory. " "Use _clean_single_outlier_native (chunked) or _clean_single_outlier_numba instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _clean_single_outlier_native_impl(df, od_col, od_iqr)


def _clean_single_outlier_native_impl(
    df: pl.DataFrame,
    od_col: str = "mag",
    od_iqr: float = 40.0,
) -> pl.DataFrame:
    """Internal implementation of explode/implode outlier cleaning.

    This is the core vectorized implementation used by both the legacy
    function and the chunked version. Do not call directly on large datasets.
    """
    col = od_col

    # Step 1: Add row index and compute bounds
    # Use interpolation='linear' to match numpy.percentile behavior
    df_work = df.with_row_index("_row_idx")

    df_work = df_work.with_columns(
        [
            pl.col(col).list.eval(pl.element().quantile(0.25, interpolation="linear")).list.first().alias("_q1"),
            pl.col(col).list.eval(pl.element().quantile(0.75, interpolation="linear")).list.first().alias("_q3"),
        ]
    )

    df_work = df_work.with_columns(
        (pl.col("_q3") - pl.col("_q1")).alias("_iqr"),
    )

    df_work = df_work.with_columns(
        [
            (pl.col("_q1") - od_iqr * pl.col("_iqr")).alias("_lower"),
            (pl.col("_q3") + od_iqr * pl.col("_iqr")).alias("_upper"),
        ]
    )

    # Step 2: Add element indices to list
    df_work = df_work.with_columns(pl.int_ranges(0, pl.col(col).list.len()).alias("_elem_idx"))

    # Step 3: Explode both column and indices
    df_exploded = df_work.explode([col, "_elem_idx"])

    # Step 4: Mark outliers
    df_exploded = df_exploded.with_columns(((pl.col(col) < pl.col("_lower")) | (pl.col(col) > pl.col("_upper"))).alias("_is_outlier"))

    # Step 5: Count outliers per row AND get previous/next values in one pass
    df_exploded = df_exploded.sort(["_row_idx", "_elem_idx"])
    df_exploded = df_exploded.with_columns(
        [
            pl.col("_is_outlier").sum().over("_row_idx").alias("_n_outliers"),
            pl.col(col).shift(1).over("_row_idx").alias("_prev_val"),
            pl.col(col).shift(-1).over("_row_idx").alias("_next_val"),
        ]
    )

    # Step 6: Compute replacement value (average of neighbors)
    df_exploded = df_exploded.with_columns(
        pl.when(pl.col("_prev_val").is_null())
        .then(pl.col("_next_val"))  # First element: use next
        .when(pl.col("_next_val").is_null())
        .then(pl.col("_prev_val"))  # Last element: use previous
        .otherwise((pl.col("_prev_val") + pl.col("_next_val")) / 2)
        .alias("_replacement")
    )

    # Step 7: Apply replacement only if n_outliers == 1 and this is the outlier
    df_exploded = df_exploded.with_columns(
        pl.when((pl.col("_n_outliers") == 1) & pl.col("_is_outlier")).then(pl.col("_replacement")).otherwise(pl.col(col)).alias("_cleaned_elem")
    )

    # Step 8: Implode back with had_od flag
    df_cleaned = df_exploded.group_by("_row_idx", maintain_order=True).agg(
        [
            pl.col("_cleaned_elem").alias(col),  # Replace original column
            # had_od = True if exactly 1 outlier was found
            (pl.col("_n_outliers").first() == 1).alias("had_od"),
        ]
    )

    # Step 9: Join back to get all original columns except the cleaned one
    other_cols = [c for c in df.columns if c != col]
    if other_cols:
        result = df.with_row_index("_row_idx").select(["_row_idx"] + other_cols).join(df_cleaned, on="_row_idx", how="left").drop("_row_idx")
    else:
        result = df_cleaned.drop("_row_idx")

    return result


def extract_additional_features_sparingly(
    dataset: Dataset,
    float32: bool = True,
    engine: str = DEFAULT_ENGINE,
    cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
) -> pl.DataFrame:
    """
    Extract additional flare-specific features for artifact rejection.

    Temporal dynamics reveal the fundamental distinction between genuine astronomical
    events and random instrumental noise. Genuine flares exhibit a coherent pattern
    of brightness evolution, with interconnected points showing a clear progression
    of intensity over time. This structural integrity distinguishes them from
    isolated, random spikes that lack any meaningful temporal progression.

    Key difference from real flares:

    +-------------------------------+---------------------------+
    | Real Flare                    | Single-point Artifact     |
    +-------------------------------+---------------------------+
    | Multiple consecutive bright   | 1 isolated bright point   |
    | points                        |                           |
    +-------------------------------+---------------------------+
    | Smooth rise + decay           | Spike with no neighbors   |
    +-------------------------------+---------------------------+
    | ~10-30+ points in the event   | 1 point                   |
    +-------------------------------+---------------------------+

    Parameters
    ----------
    dataset : datasets.Dataset
        HuggingFace Dataset with columns: mag, magerr, mjd.
    float32 : bool, default True
        If True, cast float columns to Float32 to save memory.
    engine : str, default "streaming"
        Polars execution engine.
    cache_dir : str, Path, or None, default "data"
        Directory for caching intermediate parquet files.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:
        - norm_n_below_3sigma: count of points below -3 sigma (bright outliers)
        - norm_max_consecutive_below_2sigma: max consecutive points below -2 sigma
        - norm_rise_decay_idx_ratio: index-based rise/decay ratio (< 1 for flares)
        - norm_rise_decay_time_ratio: time-based rise/decay ratio (< 1 for flares)
        - norm_n_local_minima: count of local minima (brightness peaks)
        - norm_n_zero_crossings: count of zero crossings (periodicity indicator)
        - mjd_span: observation span (mjd_last - mjd_first)

    See Also
    --------
    compute_fraction_features : Divide count features by npoints.
    """
    cache_path = Path(cache_dir) if cache_dir else None
    if cache_path:
        cache_path.mkdir(parents=True, exist_ok=True)

    dataset_len = len(dataset)

    # Check cache
    cache_file, cached_df = _load_or_migrate_cache(cache_path, "additional", dataset_len)
    if cached_df is not None:
        return cached_df

    # Get feature expressions from helper
    all_features = _get_additional_feature_exprs()

    # =========================================================================
    # Batch processing
    # =========================================================================
    results: list[pl.DataFrame] = []

    for start in tqdm(range(0, dataset_len, DEFAULT_BATCH_SIZE), desc="additional features", unit="batch"):
        end = min(start + DEFAULT_BATCH_SIZE, dataset_len)

        # Load batch
        batch = dataset.select(range(start, end))
        df = batch.select_columns(["mag", "magerr", "mjd"]).to_polars()

        # Compute norm
        df = df.with_columns(_norm_expr(float32))

        # Drop mag/magerr, keep norm and mjd
        df = df.drop(["mag", "magerr"])

        # Compute all features
        batch_result = df.lazy().select(all_features).collect(engine=engine)

        if float32:
            batch_result = batch_result.cast({c: pl.Float32 for c in batch_result.columns if batch_result[c].dtype == pl.Float64})

        results.append(batch_result)

        del df, batch, batch_result
        clean_ram()

    # =========================================================================
    # Combine batches
    # =========================================================================
    result = pl.concat(results)
    del results
    clean_ram()

    # Cache result
    if cache_file:
        result.write_parquet(cache_file, compression="zstd")
        logger.info(f"    Saved to {cache_file}")

    return result


def _compute_wavelet_features_single(
    norm_series: np.ndarray,
    mjd: np.ndarray | None = None,
    wavelets: list[str] | None = None,
    max_level: int = 6,
    interpolate: bool = True,
    n_interp_points: int = 64,
    prefix: str = "",
) -> dict[str, float]:
    """
    Compute wavelet features for a single normalized magnitude series.

    Parameters
    ----------
    norm_series : np.ndarray
        Normalized magnitude series (mag - median) / magerr_median.
    mjd : np.ndarray, optional
        Time array. If provided and interpolate=True, resamples to regular grid.
    wavelets : list[str], optional
        Wavelet types to use. Default: DEFAULT_WAVELETS
    max_level : int
        Maximum decomposition level. Default: 6
    interpolate : bool
        If True and mjd is provided, interpolate to regular time grid. Default: True
    n_interp_points : int
        Number of points for interpolation grid. Default: 64
    prefix : str
        Prefix for feature names (e.g., "norm" -> "norm_wv_haar_..."). Default: ""

    Returns
    -------
    dict[str, float]
        Dictionary of wavelet features including:
        - Per wavelet: total_energy, detail_ratio, max_detail, entropy,
          detail_approx_ratio, dominant_level
        - Per level: d{N}_energy, d{N}_rel_energy, d{N}_mean, d{N}_std,
          d{N}_skewness, d{N}_kurtosis, d{N}_mad, d{N}_frac_above_2std
    """
    if wavelets is None:
        wavelets = DEFAULT_WAVELETS

    # Fast path: use pure numba implementation (26-69x faster) for default wavelets
    if wavelets == DEFAULT_WAVELETS:
        mjd_arr = mjd if mjd is not None else np.empty(0, dtype=np.float64)
        feat_array = _compute_wavelet_features_numba(
            norm_series.astype(np.float64),
            mjd_arr.astype(np.float64) if mjd is not None else mjd_arr,
            max_level,
            interpolate and mjd is not None,
            n_interp_points,
        )
        # Convert to dict
        feature_names = _get_wavelet_feature_names(wavelets, max_level, prefix)
        return {name: float(feat_array[i]) for i, name in enumerate(feature_names)}

    # Fallback to pywt for custom wavelets
    import pywt

    n_global = 6
    n_per_level = 8

    # Pre-compute feature names (once per wavelet)
    pfx = f"{prefix}_" if prefix else ""
    all_names = {}
    for wav in wavelets:
        wav_names = []
        for name in _WAVELET_GLOBAL_NAMES:
            wav_names.append(f"{pfx}wv_{wav}_{name}")
        for lvl in range(1, max_level + 1):
            for name in _WAVELET_LEVEL_NAMES:
                wav_names.append(f"{pfx}wv_{wav}_d{lvl}_{name}")
        all_names[wav] = wav_names

    features = {}

    # Handle edge cases
    if len(norm_series) < MIN_WAVELET_SEQUENCE_LENGTH:
        for wav in wavelets:
            for name in all_names[wav]:
                features[name] = 0.0
        return features

    # Remove NaN/inf
    norm_series = np.nan_to_num(norm_series, nan=0.0, posinf=0.0, neginf=0.0)

    # Interpolate to regular time grid if mjd is provided
    if mjd is not None and interpolate and len(mjd) >= 2:
        mjd_clean = np.nan_to_num(mjd, nan=0.0, posinf=0.0, neginf=0.0)
        t_min, t_max = mjd_clean.min(), mjd_clean.max()
        if t_max > t_min:  # Valid time range
            t_regular = np.linspace(t_min, t_max, n_interp_points)
            norm_series = np.interp(t_regular, mjd_clean, norm_series)

    # Process each wavelet using numba-accelerated feature computation
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*boundary effects.*", module="pywt")

        for wav in wavelets:
            try:
                # Determine actual max level based on signal length
                actual_max_level = min(max_level, pywt.dwt_max_level(len(norm_series), wav))
                if actual_max_level < 1:
                    actual_max_level = 1

                # Discrete Wavelet Transform decomposition
                coeffs = pywt.wavedec(norm_series, wav, level=actual_max_level)

                # Convert to typed list for numba
                coeffs_approx = coeffs[0].astype(np.float64)
                coeffs_details = NumbaList()
                for c in coeffs[1:]:
                    coeffs_details.append(c.astype(np.float64))

                # Compute all features in numba (~1.5x faster than Python loop)
                feat_array = _compute_wavelet_stats_numba(coeffs_approx, coeffs_details, max_level)

                # Map to dict
                names = all_names[wav]
                for i, name in enumerate(names):
                    features[name] = float(feat_array[i])

            except (ValueError, RuntimeError):
                # Fallback on wavelet computation errors
                for name in all_names[wav]:
                    features[name] = 0.0

    return features


def _safe_normalize(mag: np.ndarray, magerr: np.ndarray) -> np.ndarray:
    """Normalize magnitude series with fallback for zero magerr.

    Uses normalize_magnitude() but falls back to centering without
    scaling if median(magerr) is zero.
    """
    try:
        return normalize_magnitude(mag, magerr)
    except ValueError:
        return mag - np.median(mag)


def _process_wavelet_chunk(
    dataset_name: str,
    hf_cache_dir: str,
    split: str,
    start_idx: int,
    end_idx: int,
    wavelets: list[str],
    max_level: int,
    output_dir: str,
    chunk_id: int,
    interpolate: bool = True,
    n_interp_points: int = 64,
) -> tuple[str, int]:
    """Process a chunk of dataset indices for wavelet features.

    Each worker loads the dataset independently and writes results to a parquet file.
    Returns (path to output file, number of records).
    Skips processing if output file already exists.
    """
    # Check if chunk already exists - skip if so
    output_path = Path(output_dir) / f"wavelet_chunk_{chunk_id:05d}.parquet"
    if output_path.exists():
        try:
            existing_count = pl.scan_parquet(output_path).select(pl.len()).collect().item()
            return str(output_path), existing_count
        except Exception:
            pass  # File corrupted, recompute

    from datasets import load_dataset

    dataset = load_dataset(dataset_name, cache_dir=hf_cache_dir, split=split)

    # Batch processing: pre-allocate array
    chunk_size = end_idx - start_idx
    feature_names = _get_wavelet_feature_names(wavelets, max_level, prefix="norm")
    n_features = len(feature_names)
    all_features = np.zeros((chunk_size, n_features), dtype=np.float64)
    row_indices = np.arange(start_idx, end_idx, dtype=np.int64)

    for j, i in enumerate(range(start_idx, end_idx)):
        row = dataset[i]
        mjd_arr = np.array(row["mjd"], dtype=np.float64)
        mag_arr = np.array(row["mag"], dtype=np.float64)
        magerr_arr = np.array(row["magerr"], dtype=np.float64)
        norm = _safe_normalize(mag_arr, magerr_arr)
        all_features[j] = _compute_wavelet_features_array(norm, mjd_arr, wavelets, max_level, interpolate, n_interp_points)

    # Create DataFrame from array (much faster than list of dicts)
    df = pl.DataFrame(all_features, schema=feature_names)
    df = df.with_columns(pl.Series("row_index", row_indices))
    df = df.cast({c: pl.Float32 for c in df.columns if df[c].dtype == pl.Float64})
    # output_path already defined at top of function
    df.write_parquet(output_path)
    return str(output_path), chunk_size


def _process_all_chunk(
    dataset_name: str,
    hf_cache_dir: str,
    split: str,
    start_idx: int,
    end_idx: int,
    output_dir: str,
    chunk_id: int,
    normalize: str | None,
    float32: bool,
    engine: str,
    wavelets: list[str],
    max_level: int,
    interpolate: bool,
    n_interp_points: int,
    argextremum_stats_col: str | None = "mag",
    argextremum_compute_additional_stats: bool = True,
    argextremum_compute_argmin_stats: bool = True,
    argextremum_compute_argmax_stats: bool = False,
    od_col: str = "mag",
    od_iqr: float = 40.0,
) -> tuple[str, int]:
    """Process a chunk: main + additional + fraction + wavelet features.

    Each worker loads the dataset independently and writes all features
    to a single parquet file. Returns (path to output file, number of records).
    Skips processing if output file already exists.
    """
    # Check if chunk already exists - skip if so
    output_path = Path(output_dir) / f"all_features_chunk_{chunk_id:05d}.parquet"
    if output_path.exists():
        try:
            existing_count = pl.scan_parquet(output_path).select(pl.len()).collect().item()
            return str(output_path), existing_count
        except Exception:
            pass  # File corrupted, recompute

    from datasets import load_dataset

    dataset = load_dataset(dataset_name, cache_dir=hf_cache_dir, split=split)
    batch = dataset.select(range(start_idx, end_idx))

    # =========================================================================
    # 1. Main features (vectorized) - reuse logic from extract_features_sparingly
    # =========================================================================
    cols_to_load = ["id", "class", "mag", "magerr", "mjd"]
    df = batch.select_columns([c for c in cols_to_load if c in batch.column_names]).to_polars()

    has_mag = "mag" in df.columns
    has_magerr = "magerr" in df.columns
    has_mjd = "mjd" in df.columns
    has_norm = has_mag and has_magerr
    has_velocity = has_mag and has_mjd

    # =========================================================================
    # 0. Outlier detection/cleaning (before norm and velocity computation)
    # =========================================================================
    if od_iqr and od_iqr > 0 and od_col in df.columns:
        df = _clean_single_outlier_native(df, od_col=od_col, od_iqr=od_iqr)
        clean_ram()
        # had_od column is now present

    # Compute norm if we have mag and magerr
    if has_norm:
        df = df.with_columns(_norm_expr(float32))

    # Compute velocity = mag.diff() / mjd.diff() (rate of magnitude change)
    if has_velocity:
        velocity_expr = pl.col("mag").list.eval(pl.element().diff().drop_nulls()) / pl.col("mjd").list.eval(pl.element().diff().drop_nulls())
        if float32:
            velocity_expr = velocity_expr.list.eval(pl.element().cast(pl.Float32))
        df = df.with_columns(velocity_expr.alias("velocity"))

    main_parts: list[pl.DataFrame] = []

    # Meta columns (id, class, had_od)
    meta_cols = [c for c in ["id", "class", "had_od"] if c in df.columns]
    if meta_cols:
        main_parts.append(df.select(meta_cols))

    # Stats for each column using extract_features_polars
    if has_mag:
        mag_features = extract_features_polars(df.select("mag"), normalize=normalize, float32=float32, engine=engine)
        main_parts.append(mag_features)

    if has_magerr:
        magerr_features = extract_features_polars(df.select("magerr"), normalize=normalize, float32=float32, engine=engine)
        # Drop npoints if already present
        if "npoints" in magerr_features.columns and any("npoints" in p.columns for p in main_parts):
            magerr_features = magerr_features.drop("npoints")
        main_parts.append(magerr_features)

    if has_norm:
        norm_features = extract_features_polars(df.select("norm"), normalize=normalize, float32=float32, engine=engine)
        if "npoints" in norm_features.columns and any("npoints" in p.columns for p in main_parts):
            norm_features = norm_features.drop("npoints")
        main_parts.append(norm_features)

    if has_velocity:
        velocity_features = extract_features_polars(df.select("velocity"), normalize=normalize, float32=float32, engine=engine)
        if "npoints" in velocity_features.columns and any("npoints" in p.columns for p in main_parts):
            velocity_features = velocity_features.drop("npoints")
        main_parts.append(velocity_features)

    if has_mjd:
        # Cast mjd to float32 if requested
        df_mjd = df.select("mjd")
        if float32:
            df_mjd = df_mjd.with_columns(pl.col("mjd").list.eval(pl.element().cast(pl.Float32)))
        mjd_features = extract_features_polars(df_mjd, normalize=normalize, float32=float32, engine=engine)
        if "npoints" in mjd_features.columns and any("npoints" in p.columns for p in main_parts):
            mjd_features = mjd_features.drop("npoints")
        main_parts.append(mjd_features)

        # Compute ts = UTC timestamp from max(mjd)
        ts_col = (
            df.select(pl.col("mjd").list.max().alias("mjd_max"))
            .with_columns((pl.lit(MJD_EPOCH) + pl.duration(days=pl.col("mjd_max"))).alias("ts"))
            .select("ts")
        )
        main_parts.append(ts_col)

    main_features = pl.concat(main_parts, how="horizontal")
    clean_ram()

    # =========================================================================
    # 2. Additional features (vectorized) - uses helper function
    # =========================================================================
    additional_parts: list[pl.DataFrame] = []

    # Additional features for mag (without mjd-dependent features)
    if has_mag:
        mag_additional_exprs = _get_additional_feature_exprs("mag", include_mjd_features=False)
        df_for_mag = df.select(["mag"])
        mag_additional = df_for_mag.lazy().select(mag_additional_exprs).collect(engine=engine)
        additional_parts.append(mag_additional)

    # Additional features for norm (with mjd-dependent features)
    if has_norm and has_mjd:
        norm_additional_exprs = _get_additional_feature_exprs("norm", include_mjd_features=True)
        df_for_norm = df.select(["norm", "mjd"])
        norm_additional = df_for_norm.lazy().select(norm_additional_exprs).collect(engine=engine)
        additional_parts.append(norm_additional)

    # Additional features for velocity (without mjd-dependent features, prefix="vel")
    if has_velocity:
        vel_additional_exprs = _get_additional_feature_exprs("velocity", prefix="vel", include_mjd_features=False)
        df_for_vel = df.select(["velocity"])
        vel_additional = df_for_vel.lazy().select(vel_additional_exprs).collect(engine=engine)
        additional_parts.append(vel_additional)

    additional_features = pl.concat(additional_parts, how="horizontal") if additional_parts else pl.DataFrame()

    if float32 and len(additional_features) > 0:
        additional_features = additional_features.cast({c: pl.Float32 for c in additional_features.columns if additional_features[c].dtype == pl.Float64})

    # =========================================================================
    # 2b. Argextremum stats (sub-series split by argmax/argmin)
    # =========================================================================
    argextremum_features = pl.DataFrame()
    if argextremum_stats_col and argextremum_stats_col in df.columns:
        # Compute stats on mag, norm, velocity split by argmax/argmin of index_col
        stats_cols = [c for c in ["mag", "norm", "velocity"] if c in df.columns]
        argext_exprs = _get_argextremum_stats_exprs(
            index_col=argextremum_stats_col,
            stats_cols=stats_cols,
            compute_additional=argextremum_compute_additional_stats,
            compute_argmin_stats=argextremum_compute_argmin_stats,
            compute_argmax_stats=argextremum_compute_argmax_stats,
        )
        if argext_exprs:  # Only compute if there are expressions
            df_for_argext = df.select([argextremum_stats_col] + [c for c in stats_cols if c != argextremum_stats_col])
            argextremum_features = df_for_argext.lazy().select(argext_exprs).collect(engine=engine)

            if float32 and len(argextremum_features) > 0:
                argextremum_features = argextremum_features.cast(
                    {c: pl.Float32 for c in argextremum_features.columns if argextremum_features[c].dtype == pl.Float64}
                )

    # =========================================================================
    # 3. Combine main + additional + argextremum
    # =========================================================================
    parts_to_combine = [main_features, additional_features]
    if len(argextremum_features) > 0:
        parts_to_combine.append(argextremum_features)
    combined = pl.concat(parts_to_combine, how="horizontal")
    clean_ram()

    # =========================================================================
    # 4. Fraction features
    # =========================================================================
    fractions = compute_fraction_features(combined)
    combined = pl.concat([combined, fractions], how="horizontal")

    # =========================================================================
    # 5. Wavelet features (batch processing with pre-allocated array)
    # =========================================================================
    chunk_size = end_idx - start_idx
    feature_names = _get_wavelet_feature_names(wavelets, max_level, prefix="norm")
    n_features = len(feature_names)
    all_features = np.zeros((chunk_size, n_features), dtype=np.float64)

    for i in range(chunk_size):
        row = batch[i]
        mjd_arr = np.array(row["mjd"], dtype=np.float64)
        mag_arr = np.array(row["mag"], dtype=np.float64)
        magerr_arr = np.array(row["magerr"], dtype=np.float64)
        norm = _safe_normalize(mag_arr, magerr_arr)
        all_features[i] = _compute_wavelet_features_array(norm, mjd_arr, wavelets, max_level, interpolate, n_interp_points)

    wavelet_df = pl.DataFrame(all_features, schema=feature_names)
    if float32:
        wavelet_df = wavelet_df.cast({c: pl.Float32 for c in wavelet_df.columns if wavelet_df[c].dtype == pl.Float64})

    # =========================================================================
    # 6. Final horizontal concat and save
    # =========================================================================
    result = pl.concat([combined, wavelet_df], how="horizontal")
    result = result.with_columns(pl.lit(start_idx).alias("_start_idx"))  # for sorting

    # output_path already defined at top of function
    result.write_parquet(output_path)

    return str(output_path), len(result)


def extract_wavelet_features_sparingly(
    dataset_name: str,
    split: str,
    hf_cache_dir: str,
    wavelets: list[str] | None = None,
    max_level: int = 6,
    float32: bool = True,
    n_jobs: int = -1,
    cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
    interpolate: bool = True,
    n_interp_points: int = 64,
) -> pl.DataFrame:
    """
    Extract wavelet-based features for flare shape detection.

    Wavelets capture multi-scale temporal structure, making them ideal for
    detecting the characteristic fast-rise slow-decay shape of stellar flares.

    Features extracted per wavelet type:
    - total_energy: Total energy across all coefficients
    - detail_ratio: Fraction of energy in detail coefficients (transients)
    - max_detail: Maximum absolute detail coefficient (spike detection)
    - entropy: Wavelet entropy (lower = more coherent signal)
    - detail_approx_ratio: Detail to approximation energy ratio
    - dominant_level: Level with maximum energy
    - d{N}_energy: Energy at each decomposition level
    - d{N}_rel_energy: Relative energy (normalized to total)
    - d{N}_mean/std/skewness/kurtosis: Statistical moments of coefficients
    - d{N}_mad: Mean absolute deviation from median
    - d{N}_frac_above_2std: Fraction of outlier coefficients

    Default wavelet types (DEFAULT_WAVELETS):
    - haar: Sharp transitions detection (flare rise)
    - db4, db6: Asymmetric, captures rise/decay asymmetry
    - coif3: Vanishing moments for transient analysis
    - sym4: Symmetric reference for comparison

    Parameters
    ----------
    dataset_name : str
        HuggingFace dataset name (e.g., "snad-space/ztf-m-dwarf-flares-2025").
    split : str
        Dataset split to use (e.g., "target", "train", "test").
    hf_cache_dir : str
        HuggingFace cache directory for the dataset.
    wavelets : list[str], optional
        Wavelet types to compute. Default: DEFAULT_WAVELETS
    max_level : int, default 6
        Maximum decomposition level.
    float32 : bool, default True
        If True, cast to Float32 to save memory.
    n_jobs : int, default -1
        Number of parallel jobs. -1 = all physical cores.
    cache_dir : str, Path, or None, default "data"
        Directory for caching parquet files.
    interpolate : bool, default True
        If True, interpolate to regular time grid before DWT (uses mjd).
    n_interp_points : int, default 64
        Number of points for interpolation grid.

    Returns
    -------
    pl.DataFrame
        DataFrame with wavelet features (one row per light curve).

    Notes
    -----
    Computation is distributed across all physical cores using joblib.
    Each worker loads the dataset independently from cached files.
    Features are computed on normalized magnitude: (mag - median) / magerr_median.
    When interpolate=True, the signal is resampled to a regular time grid before DWT.
    """
    from joblib import Parallel, delayed
    from datasets import load_dataset
    import psutil

    if wavelets is None:
        wavelets = DEFAULT_WAVELETS

    # Determine number of physical cores (not logical/hyperthreaded)
    if n_jobs == -1:
        n_jobs = psutil.cpu_count(logical=False) or 4

    cache_path = Path(cache_dir) if cache_dir else None
    if cache_path:
        cache_path.mkdir(parents=True, exist_ok=True)

    # Load dataset to get length
    dataset = load_dataset(dataset_name, cache_dir=hf_cache_dir, split=split)
    dataset_len = len(dataset)

    wavelet_str = "_".join(wavelets)

    # Cache file
    use_cache = cache_path is not None and dataset_len >= MIN_ROWS_FOR_CACHING
    cache_file = cache_path / f"features_wavelet_{wavelet_str}_{dataset_len}.parquet" if use_cache else None

    # Check cache
    if cache_file and cache_file.exists():
        del dataset
        clean_ram()
        logger.info("[wavelet] Loading from cache...")
        return pl.read_parquet(cache_file, parallel="columns")

    logger.info(f"[wavelet] Extracting features using {n_jobs} cores, wavelets={wavelets}")

    # For small datasets, compute in-memory (dataset already loaded)
    if dataset_len < MIN_ROWS_FOR_CACHING:
        logger.info(f"[wavelet] Dataset small ({dataset_len} < {MIN_ROWS_FOR_CACHING}), computing in-memory...")

        # Batch processing: pre-allocate array, compute features, create DataFrame at end
        feature_names = _get_wavelet_feature_names(wavelets, max_level, prefix="norm")
        n_features = len(feature_names)
        all_features = np.zeros((dataset_len, n_features), dtype=np.float64)

        for i in tqdm(range(dataset_len), desc="wavelet features", unit="row"):
            row = dataset[i]
            mjd_arr = np.array(row["mjd"], dtype=np.float64)
            mag_arr = np.array(row["mag"], dtype=np.float64)
            magerr_arr = np.array(row["magerr"], dtype=np.float64)
            norm = _safe_normalize(mag_arr, magerr_arr)
            all_features[i] = _compute_wavelet_features_array(norm, mjd_arr, wavelets, max_level, interpolate, n_interp_points)

        # Create DataFrame from array (much faster than list of dicts)
        result = pl.DataFrame(all_features, schema=feature_names)
        result = result.with_columns(pl.Series("row_index", range(dataset_len)))
        if float32:
            result = result.cast({c: pl.Float32 for c in result.columns if result[c].dtype == pl.Float64})
        return result

    # =========================================================================
    # Large dataset: split across workers, each writes to separate parquet file
    # =========================================================================
    del dataset  # Free memory, workers will load their own
    clean_ram()

    chunk_ranges = [(i, min(i + WAVELET_CHUNK_SIZE, dataset_len)) for i in range(0, dataset_len, WAVELET_CHUNK_SIZE)]
    n_chunks = len(chunk_ranges)

    # Create output directory for chunk files
    chunks_dir = cache_path / f"wavelet_chunks_{wavelet_str}" if cache_path else Path(f"wavelet_chunks_{wavelet_str}")
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Check if chunks already exist with expected record count
    existing_files = list(chunks_dir.glob("wavelet_chunk_*.parquet"))
    if len(existing_files) == n_chunks:
        # Verify record count via lazy scan
        total_existing = pl.scan_parquet(chunks_dir / "wavelet_chunk_*.parquet").select(pl.len()).collect().item()
        if total_existing == dataset_len:
            logger.info(f"[wavelet] Found {n_chunks} existing chunks with {total_existing} records, reusing...")
            return pl.scan_parquet(chunks_dir / "wavelet_chunk_*.parquet").sort("row_index").collect()

    # Parallel processing - each worker writes results to its own parquet file
    logger.info(f"[wavelet] Computing {n_chunks} chunks of {WAVELET_CHUNK_SIZE} samples each...")
    jobs = [
        delayed(_process_wavelet_chunk)(
            dataset_name, hf_cache_dir, split, start, end, wavelets, max_level, str(chunks_dir), chunk_id, interpolate, n_interp_points
        )
        for chunk_id, (start, end) in enumerate(chunk_ranges)
    ]
    chunk_results = []
    with tqdm(total=len(jobs), desc="wavelet features", unit="chunk") as pbar:
        for result in Parallel(n_jobs=n_jobs, backend="loky", return_as="generator")(jobs):
            chunk_results.append(result)
            pbar.update(1)

    total_records = sum(n for _, n in chunk_results)
    logger.info(f"[wavelet] Computed {total_records} records in {len(chunk_results)} chunks")

    # Use lazy scan with wildcard, sort by row_index, collect
    result = pl.scan_parquet(chunks_dir / "wavelet_chunk_*.parquet").sort("row_index").collect()

    clean_ram()
    return result


def compute_fraction_features(
    df: pl.DataFrame,
    npoints_col: str = "npoints",
) -> pl.DataFrame:
    """
    Compute fraction features by dividing count features by npoints.

    This function normalizes count-based features from extract_additional_features_sparingly
    by dividing them by the total number of points in each light curve.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame containing both count features and npoints column.
        Expected count columns:
        - norm_n_below_3sigma
        - norm_max_consecutive_below_2sigma
        - norm_n_local_minima
        - norm_n_zero_crossings
    npoints_col : str, default "npoints"
        Name of the column containing point counts.

    Returns
    -------
    pl.DataFrame
        DataFrame with new fraction columns:
        - norm_frac_below_3sigma: norm_n_below_3sigma / npoints
        - norm_max_consecutive_frac_below_2sigma: norm_max_consecutive_below_2sigma / npoints
        - norm_frac_local_minima: norm_n_local_minima / npoints
        - norm_frac_zero_crossings: norm_n_zero_crossings / npoints

    Examples
    --------
    >>> features = extract_features_sparingly(dataset)
    >>> additional = extract_additional_features_sparingly(dataset)
    >>> combined = pl.concat([features, additional], how="horizontal")
    >>> fractions = compute_fraction_features(combined)
    >>> final = pl.concat([combined, fractions], how="horizontal")
    """
    npoints = pl.col(npoints_col).cast(pl.Float32)

    fraction_exprs = []

    # Fraction of points below -3 sigma
    if "norm_n_below_3sigma" in df.columns:
        fraction_exprs.append((pl.col("norm_n_below_3sigma") / npoints).alias("norm_frac_below_3sigma"))

    # Max consecutive as fraction of total
    if "norm_max_consecutive_below_2sigma" in df.columns:
        fraction_exprs.append((pl.col("norm_max_consecutive_below_2sigma") / npoints).alias("norm_max_consecutive_frac_below_2sigma"))

    # Local minima as fraction of total
    if "norm_n_local_minima" in df.columns:
        fraction_exprs.append((pl.col("norm_n_local_minima") / npoints).alias("norm_frac_local_minima"))

    # Zero crossings as fraction of total
    if "norm_n_zero_crossings" in df.columns:
        fraction_exprs.append((pl.col("norm_n_zero_crossings") / npoints).alias("norm_frac_zero_crossings"))

    if not fraction_exprs:
        raise ValueError("No count columns found to convert to fractions")

    return df.select(fraction_exprs)


def rank_discriminative_features(
    df_population: pl.DataFrame,
    df_rare: pl.DataFrame,
    exclude_cols: list[str] | None = None,
    epsilon: float = 1e-10,
) -> pl.DataFrame:
    """
    Rank features by their ability to discriminate rare events from population.

    Computes various statistics for each numeric feature in both datasets
    and calculates ratios/differences to identify the most discriminative features.

    Parameters
    ----------
    df_population : pl.DataFrame
        Features computed on the large/general population dataset.
    df_rare : pl.DataFrame
        Features computed on the rare event dataset (e.g., known flares).
    exclude_cols : list[str] or None, default None
        Columns to exclude from analysis (e.g., 'id', 'class', 'ts').
        If None, defaults to ['id', 'class', 'ts', 'npoints'].
    epsilon : float, default 1e-10
        Small value to prevent division by zero.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:
        - feature: feature name
        - For each stat (mean, median, std, q25, q75, q90, q95, min, max):
          - {stat}_pop: statistic on population
          - {stat}_rare: statistic on rare events
          - {stat}_ratio: rare/pop ratio
          - {stat}_log_ratio: log2(rare/pop) for symmetric interpretation
        - cohens_d_mean: standardized effect size for mean
        - cohens_d_median: standardized effect size for median
        - max_abs_log_ratio: maximum absolute log-ratio across all stats
        Sorted by max_abs_log_ratio descending (most discriminative first).

    Examples
    --------
    >>> features_big = extract_features_polars(big_dataset)
    >>> features_flares = extract_features_polars(flares_dataset)
    >>> ranking = rank_discriminative_features(features_big, features_flares)
    >>> print(ranking.head(10))  # Top 10 most discriminative features
    """
    if exclude_cols is None:
        exclude_cols = ["id", "class", "ts", "npoints"]

    # Get numeric columns present in both datasets
    pop_cols = set(df_population.columns) - set(exclude_cols)
    rare_cols = set(df_rare.columns) - set(exclude_cols)
    common_cols = list(pop_cols & rare_cols)

    # Filter to numeric columns only
    numeric_cols = [c for c in common_cols if df_population[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)]

    if not numeric_cols:
        raise ValueError("No common numeric columns found between datasets")

    # Define statistics to compute
    stat_exprs = {
        "mean": lambda c: pl.col(c).mean(),
        "median": lambda c: pl.col(c).median(),
        "std": lambda c: pl.col(c).std(),
    }

    # Compute statistics for population
    pop_stats = df_population.select([expr(col).alias(f"{col}_{stat}") for col in numeric_cols for stat, expr in stat_exprs.items()])

    # Compute statistics for rare events
    rare_stats = df_rare.select([expr(col).alias(f"{col}_{stat}") for col in numeric_cols for stat, expr in stat_exprs.items()])

    # Build result rows
    results = []
    for col in numeric_cols:
        row = {"feature": col}

        log_ratios = []
        for stat in stat_exprs:
            pop_val = pop_stats[f"{col}_{stat}"][0]
            rare_val = rare_stats[f"{col}_{stat}"][0]

            # Handle None values
            if pop_val is None or rare_val is None:
                row[f"{stat}_pop"] = pop_val
                row[f"{stat}_rare"] = rare_val
                row[f"{stat}_ratio"] = None
                row[f"{stat}_log_ratio"] = None
                continue

            row[f"{stat}_pop"] = pop_val
            row[f"{stat}_rare"] = rare_val

            # Compute ratio (handle near-zero denominators)
            denom = abs(pop_val) + epsilon
            ratio = rare_val / denom if pop_val >= 0 else rare_val / -denom
            row[f"{stat}_ratio"] = ratio

            # Log ratio (handle negative values by using signed log)
            if ratio > 0:
                log_ratio = np.log2(ratio + epsilon)
            else:
                log_ratio = -np.log2(abs(ratio) + epsilon)
            row[f"{stat}_log_ratio"] = log_ratio
            log_ratios.append(abs(log_ratio))

        # Cohen's d effect size (for both mean and median)
        mean_pop = row.get("mean_pop")
        mean_rare = row.get("mean_rare")
        median_pop = row.get("median_pop")
        median_rare = row.get("median_rare")
        std_pop = row.get("std_pop")
        std_rare = row.get("std_rare")

        if all(v is not None for v in [std_pop, std_rare]):
            # Pooled standard deviation
            n_pop = len(df_population)
            n_rare = len(df_rare)
            pooled_std = np.sqrt(((n_pop - 1) * std_pop**2 + (n_rare - 1) * std_rare**2) / (n_pop + n_rare - 2))

            if mean_pop is not None and mean_rare is not None:
                row["cohens_d_mean"] = (mean_rare - mean_pop) / (pooled_std + epsilon)
            else:
                row["cohens_d_mean"] = None

            if median_pop is not None and median_rare is not None:
                row["cohens_d_median"] = (median_rare - median_pop) / (pooled_std + epsilon)
            else:
                row["cohens_d_median"] = None
        else:
            row["cohens_d_mean"] = None
            row["cohens_d_median"] = None

        # Max absolute log ratio for sorting
        row["max_abs_log_ratio"] = max(log_ratios) if log_ratios else 0.0

        results.append(row)

    # Create DataFrame and sort by discriminative power
    result_df = pl.DataFrame(results)
    result_df = result_df.sort("max_abs_log_ratio", descending=True)

    return result_df


def extract_all_features(
    dataset_name: str,
    split: str,
    hf_cache_dir: str,
    normalize: str | None = None,
    float32: bool = True,
    engine: str = DEFAULT_ENGINE,
    n_jobs: int = -1,
    cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
    chunk_size: int = WAVELET_CHUNK_SIZE,
    wavelets: list[str] | None = None,
    max_level: int = 5,
    interpolate: bool = True,
    n_interp_points: int = 64,
    argextremum_stats_col: str | None = "mag",
    argextremum_compute_additional_stats: bool = True,
    argextremum_compute_argmin_stats: bool = True,
    argextremum_compute_argmax_stats: bool = False,
    od_col: str = "mag",
    od_iqr: float = 40.0,
    from_row: int | None = None,
    to_row: int | None = None,
) -> pl.DataFrame:
    """
    Extract ALL features (main + additional + fraction + wavelet) in one pass.

    Processes HuggingFace dataset in parallel chunks, saves each chunk to disk,
    then combines results. This is a unified replacement for calling
    extract_features_sparingly, extract_additional_features_sparingly, and
    extract_wavelet_features_sparingly separately.

    Parameters
    ----------
    dataset_name : str
        HuggingFace dataset name (e.g., "snad-space/ztf-m-dwarf-flares-2025").
    split : str
        Dataset split to use (e.g., "target", "train", "test").
    hf_cache_dir : str
        HuggingFace cache directory for the dataset.
    normalize : str or None, default None
        Normalization method passed to extract_features_polars.
    float32 : bool, default True
        If True, cast float columns to Float32 to save memory.
    engine : str, default "streaming"
        Polars execution engine: "streaming" for memory-efficient processing.
    n_jobs : int, default -1
        Number of parallel jobs. -1 = all physical cores.
    cache_dir : str, Path, or None, default "data"
        Directory for caching parquet files.
    chunk_size : int, default WAVELET_CHUNK_SIZE (1_000_000)
        Number of rows per chunk for parallel processing.
    wavelets : list[str], optional
        Wavelet types to compute. Default: DEFAULT_WAVELETS
    max_level : int, default 5
        Maximum wavelet decomposition level.
    interpolate : bool, default True
        If True, interpolate to regular time grid before DWT.
    n_interp_points : int, default 64
        Number of points for interpolation grid.
    argextremum_stats_col : str or None, default "mag"
        Column to use for argextremum stats. If None, argextremum stats are disabled.
    argextremum_compute_additional_stats : bool, default True
        If True, compute additional argextremum stats (skewness, kurtosis, etc.).
    argextremum_compute_argmin_stats : bool, default True
        If True, compute stats for sub-series split at argmin (brightest point).
    argextremum_compute_argmax_stats : bool, default False
        If True, compute stats for sub-series split at argmax (faintest point).
    od_col : str, default "mag"
        Column for single-outlier detection and cleaning.
    od_iqr : float, default 40.0
        IQR multiplier for outlier detection threshold.
    from_row : int or None, default None
        If both from_row and to_row are specified, process only rows [from_row:to_row).
        Useful for testing or external chunking. Disables parallel chunk processing.
    to_row : int or None, default None
        If both from_row and to_row are specified, process only rows [from_row:to_row).

    Returns
    -------
    pl.DataFrame
        DataFrame with all features:
        - id, class, npoints, ts (metadata)
        - mag_*, magerr_*, norm_*, velocity_*, mjd_diff_* (main statistical features)
          velocity = (mag[i+1] - mag[i]) / (mjd[i+1] - mjd[i]) - rate of magnitude change
        - norm_n_below_3sigma, norm_max_consecutive_*, etc. (additional features)
        - norm_frac_* (fraction features)
        - wv_haar_*, wv_db4_*, etc. (wavelet features)

    Examples
    --------
    >>> # Before (multiple calls):
    >>> # big_features = extract_features_sparingly(dataset["train"])
    >>> # additional = extract_additional_features_sparingly(dataset["train"])
    >>> # big_features = enrich_features(big_features, additional)
    >>> # wave_features = extract_wavelet_features_sparingly(...)
    >>> # big_features = pl.concat([big_features, wave_features], how="horizontal")
    >>>
    >>> # After (single call):
    >>> all_features = extract_all_features(
    ...     dataset_name="snad-space/ztf-m-dwarf-flares-2025",
    ...     split="train",
    ...     hf_cache_dir="./hf_cache",
    ...     cache_dir="./output",
    ... )
    """
    from joblib import Parallel, delayed
    from datasets import load_dataset
    import psutil

    if wavelets is None:
        wavelets = DEFAULT_WAVELETS
    if n_jobs == -1:
        n_jobs = psutil.cpu_count(logical=False) or 4

    cache_path = Path(cache_dir) if cache_dir else None
    if cache_path:
        cache_path.mkdir(parents=True, exist_ok=True)

    # Load dataset to get length
    dataset = load_dataset(dataset_name, cache_dir=hf_cache_dir, split=split)
    full_dataset_len = len(dataset)

    # Slice dataset if from_row/to_row specified
    if from_row is not None and to_row is not None:
        dataset = dataset.select(range(from_row, min(to_row, full_dataset_len)))
        dataset_len = len(dataset)
        row_suffix = f"_rows{from_row}-{to_row}"
    else:
        dataset_len = full_dataset_len
        row_suffix = ""

    # Cache file check
    wavelet_str = "_".join(wavelets)
    use_cache = cache_path is not None and dataset_len >= MIN_ROWS_FOR_CACHING
    cache_file = cache_path / f"features_all_{wavelet_str}_{dataset_len}{row_suffix}.parquet" if use_cache else None

    if cache_file and cache_file.exists():
        del dataset
        clean_ram()
        logger.info("[all_features] Loading from cache...")
        return pl.read_parquet(cache_file, parallel="columns")

    logger.info(f"[all_features] Extracting features using {n_jobs} cores, wavelets={wavelets}")

    # =========================================================================
    # Small dataset or row-sliced: in-memory processing (dataset already loaded)
    # =========================================================================
    # Use in-memory path if dataset is small OR if row slicing is used
    # (chunked workers would load full dataset, ignoring from_row/to_row)
    use_inmemory = dataset_len < MIN_ROWS_FOR_CACHING or row_suffix != ""
    if use_inmemory:
        logger.info(f"[all_features] Computing in-memory (len={dataset_len}, sliced={bool(row_suffix)})...")

        # Main features
        cols_to_load = ["id", "class", "mag", "magerr", "mjd"]
        df = dataset.select_columns([c for c in cols_to_load if c in dataset.column_names]).to_polars()

        has_mag = "mag" in df.columns
        has_magerr = "magerr" in df.columns
        has_mjd = "mjd" in df.columns
        has_norm = has_mag and has_magerr
        has_velocity = has_mag and has_mjd

        # Outlier detection/cleaning (before norm and velocity computation)
        if od_iqr and od_iqr > 0 and od_col in df.columns:
            df = _clean_single_outlier_native(df, od_col=od_col, od_iqr=od_iqr)
            # had_od column is now present

        if has_norm:
            df = df.with_columns(_norm_expr(float32))

        # Compute velocity = mag.diff() / mjd.diff() (rate of magnitude change)
        if has_velocity:
            velocity_expr = pl.col("mag").list.eval(pl.element().diff().drop_nulls()) / pl.col("mjd").list.eval(pl.element().diff().drop_nulls())
            if float32:
                velocity_expr = velocity_expr.list.eval(pl.element().cast(pl.Float32))
            df = df.with_columns(velocity_expr.alias("velocity"))

        main_parts: list[pl.DataFrame] = []
        meta_cols = [c for c in ["id", "class", "had_od"] if c in df.columns]
        if meta_cols:
            main_parts.append(df.select(meta_cols))

        if has_mag:
            main_parts.append(extract_features_polars(df.select("mag"), normalize=normalize, float32=float32, engine=engine))
        if has_magerr:
            magerr_f = extract_features_polars(df.select("magerr"), normalize=normalize, float32=float32, engine=engine)
            if "npoints" in magerr_f.columns and any("npoints" in p.columns for p in main_parts):
                magerr_f = magerr_f.drop("npoints")
            main_parts.append(magerr_f)
        if has_norm:
            norm_f = extract_features_polars(df.select("norm"), normalize=normalize, float32=float32, engine=engine)
            if "npoints" in norm_f.columns and any("npoints" in p.columns for p in main_parts):
                norm_f = norm_f.drop("npoints")
            main_parts.append(norm_f)
        if has_velocity:
            velocity_f = extract_features_polars(df.select("velocity"), normalize=normalize, float32=float32, engine=engine)
            if "npoints" in velocity_f.columns and any("npoints" in p.columns for p in main_parts):
                velocity_f = velocity_f.drop("npoints")
            main_parts.append(velocity_f)
        if has_mjd:
            df_mjd = df.select("mjd")
            if float32:
                df_mjd = df_mjd.with_columns(pl.col("mjd").list.eval(pl.element().cast(pl.Float32)))
            mjd_f = extract_features_polars(df_mjd, normalize=normalize, float32=float32, engine=engine)
            if "npoints" in mjd_f.columns and any("npoints" in p.columns for p in main_parts):
                mjd_f = mjd_f.drop("npoints")
            main_parts.append(mjd_f)
            ts_col = (
                df.select(pl.col("mjd").list.max().alias("mjd_max"))
                .with_columns((pl.lit(MJD_EPOCH) + pl.duration(days=pl.col("mjd_max"))).alias("ts"))
                .select("ts")
            )
            main_parts.append(ts_col)

        main_features = pl.concat(main_parts, how="horizontal")

        # Additional features for mag, norm, velocity
        additional_parts: list[pl.DataFrame] = []

        if has_mag:
            mag_add_exprs = _get_additional_feature_exprs("mag", include_mjd_features=False)
            mag_add = df.select(["mag"]).lazy().select(mag_add_exprs).collect(engine=engine)
            additional_parts.append(mag_add)

        if has_norm and has_mjd:
            norm_add_exprs = _get_additional_feature_exprs("norm", include_mjd_features=True)
            norm_add = df.select(["norm", "mjd"]).lazy().select(norm_add_exprs).collect(engine=engine)
            additional_parts.append(norm_add)

        if has_velocity:
            vel_add_exprs = _get_additional_feature_exprs("velocity", prefix="vel", include_mjd_features=False)
            vel_add = df.select(["velocity"]).lazy().select(vel_add_exprs).collect(engine=engine)
            additional_parts.append(vel_add)

        additional_features = pl.concat(additional_parts, how="horizontal") if additional_parts else pl.DataFrame()
        if float32 and len(additional_features) > 0:
            additional_features = additional_features.cast({c: pl.Float32 for c in additional_features.columns if additional_features[c].dtype == pl.Float64})

        # Argextremum stats (sub-series split by argmax/argmin)
        argextremum_features = pl.DataFrame()
        if argextremum_stats_col and argextremum_stats_col in df.columns:
            stats_cols = [c for c in ["mag", "norm", "velocity"] if c in df.columns]
            argext_exprs = _get_argextremum_stats_exprs(
                index_col=argextremum_stats_col,
                stats_cols=stats_cols,
                compute_additional=argextremum_compute_additional_stats,
                compute_argmin_stats=argextremum_compute_argmin_stats,
                compute_argmax_stats=argextremum_compute_argmax_stats,
            )
            if argext_exprs:  # Only compute if there are expressions
                df_for_argext = df.select([argextremum_stats_col] + [c for c in stats_cols if c != argextremum_stats_col])
                argextremum_features = df_for_argext.lazy().select(argext_exprs).collect(engine=engine)
                if float32 and len(argextremum_features) > 0:
                    argextremum_features = argextremum_features.cast(
                        {c: pl.Float32 for c in argextremum_features.columns if argextremum_features[c].dtype == pl.Float64}
                    )

        parts_to_combine = [main_features, additional_features]
        if len(argextremum_features) > 0:
            parts_to_combine.append(argextremum_features)
        combined = pl.concat(parts_to_combine, how="horizontal")
        fractions = compute_fraction_features(combined)
        combined = pl.concat([combined, fractions], how="horizontal")

        # Wavelet features (batch processing with pre-allocated array)
        feature_names = _get_wavelet_feature_names(wavelets, max_level, prefix="norm")
        n_features = len(feature_names)
        all_wavelet_features = np.zeros((dataset_len, n_features), dtype=np.float64)

        for i in tqdm(range(dataset_len), desc="wavelet features", unit="row"):
            row = dataset[i]
            mjd_arr = np.array(row["mjd"], dtype=np.float64)
            mag_arr = np.array(row["mag"], dtype=np.float64)
            magerr_arr = np.array(row["magerr"], dtype=np.float64)
            norm = _safe_normalize(mag_arr, magerr_arr)
            all_wavelet_features[i] = _compute_wavelet_features_array(norm, mjd_arr, wavelets, max_level, interpolate, n_interp_points)

        wavelet_df = pl.DataFrame(all_wavelet_features, schema=feature_names)
        if float32:
            wavelet_df = wavelet_df.cast({c: pl.Float32 for c in wavelet_df.columns if wavelet_df[c].dtype == pl.Float64})

        result = pl.concat([combined, wavelet_df], how="horizontal")
        return result

    # =========================================================================
    # Large dataset: chunk processing with joblib
    # =========================================================================
    del dataset  # Free memory, workers will load their own
    clean_ram()

    chunk_ranges = [(i, min(i + chunk_size, dataset_len)) for i in range(0, dataset_len, chunk_size)]
    n_chunks = len(chunk_ranges)

    chunks_dir = cache_path / f"all_features_chunks_{wavelet_str}" if cache_path else Path(f"all_features_chunks_{wavelet_str}")
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Check if chunks already exist with expected record count
    existing_files = list(chunks_dir.glob("all_features_chunk_*.parquet"))
    if len(existing_files) == n_chunks:
        total_existing = pl.scan_parquet(chunks_dir / "all_features_chunk_*.parquet").select(pl.len()).collect().item()
        if total_existing == dataset_len:
            logger.info(f"[all_features] Found {n_chunks} existing chunks with {total_existing} records, reusing...")
            result = pl.scan_parquet(chunks_dir / "all_features_chunk_*.parquet").sort("_start_idx").drop("_start_idx").collect()
            if cache_file:
                result.write_parquet(cache_file, compression="zstd")
                logger.info(f"[all_features] Saved to {cache_file}")
            return result

    # Parallel processing - each worker writes results to its own parquet file
    logger.info(f"[all_features] Computing {n_chunks} chunks of {chunk_size} samples each...")
    jobs = [
        delayed(_process_all_chunk)(
            dataset_name,
            hf_cache_dir,
            split,
            start,
            end,
            str(chunks_dir),
            chunk_id,
            normalize,
            float32,
            engine,
            wavelets,
            max_level,
            interpolate,
            n_interp_points,
            argextremum_stats_col,
            argextremum_compute_additional_stats,
            argextremum_compute_argmin_stats,
            argextremum_compute_argmax_stats,
            od_col,
            od_iqr,
        )
        for chunk_id, (start, end) in enumerate(chunk_ranges)
    ]

    chunk_results = []
    with tqdm(total=len(jobs), desc="all features", unit="chunk") as pbar:
        for result in Parallel(n_jobs=n_jobs, backend="loky", return_as="generator")(jobs):
            chunk_results.append(result)
            pbar.update(1)

    total_records = sum(n for _, n in chunk_results)
    logger.info(f"[all_features] Computed {total_records} records in {len(chunk_results)} chunks")

    # Use lazy scan with wildcard, sort by _start_idx, drop helper column, collect
    result = pl.scan_parquet(chunks_dir / "all_features_chunk_*.parquet").sort("_start_idx").drop("_start_idx").collect()

    # Save final cache
    if cache_file:
        result.write_parquet(cache_file, compression="zstd")
        logger.info(f"[all_features] Saved to {cache_file}")

    clean_ram()
    return result

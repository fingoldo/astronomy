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
from typing import Callable, Union

import numpy as np
import psutil
import pandas as pd
import polars as pl
from datasets import load_dataset, Dataset
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyutilz.system import clean_ram
from scipy import stats

# MJD epoch: November 17, 1858, 00:00:00 UTC
MJD_EPOCH = datetime(1858, 11, 17, tzinfo=timezone.utc)

# Conversion factor from matplotlib inches to plotly pixels
INCHES_TO_PIXELS = 100

# Default configuration
DEFAULT_ENGINE = "streaming"
DEFAULT_CACHE_DIR = "data"
DEFAULT_RAM_THRESHOLD_GB = 512

DataFrameType = Union[pl.DataFrame, pd.DataFrame]

logger = logging.getLogger(__name__)


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
    if med_err == 0:
        raise ValueError("Cannot normalize: median(magerr) is 0")
    return (mag - np.median(mag)) / med_err


def _handle_npoints(df: pl.DataFrame, has_npoints: bool) -> tuple[pl.DataFrame, bool]:
    """Handle duplicate npoints column in feature DataFrames.

    If npoints column exists and we already have one, drop it.
    Otherwise, mark that we now have npoints.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame that may contain npoints column.
    has_npoints : bool
        Whether npoints has already been encountered.

    Returns
    -------
    tuple[pl.DataFrame, bool]
        Updated DataFrame and has_npoints flag.
    """
    if "npoints" in df.columns:
        if has_npoints:
            return df.drop("npoints"), has_npoints
        return df, True
    return df, has_npoints


def view_series(
    df: DataFrameType | dict,
    index: int,
    figsize: tuple[int, int] = (8, 4),
    advanced: bool = False,
) -> None:
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
    """
    row = _get_row(df, index)

    mjd = np.array(row["mjd"])
    mag = np.array(row["mag"])
    magerr = np.array(row["magerr"])
    cls = row["class"]

    # Convert MJD to UTC datetime
    utc_dates = [MJD_EPOCH + pd.Timedelta(days=float(m)) for m in mjd]

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
        title=f"Record #{index} — class: {cls}, npoints: {len(mag)}",
        xaxis_title="Date (UTC)",
        yaxis_title="mag",
        width=width,
        height=height,
    )

    # Invert y-axis (brighter = lower magnitude in astronomy)
    fig.update_yaxes(autorange="reversed")

    fig.show()

    if advanced:
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
        fig_norm.show()


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
        # Center mag within each list, then divide by magerr median
        mag_centered = pl.col("mag").list.eval(pl.element() - pl.element().median())
        magerr_median = pl.col("magerr").list.eval(pl.element().median()).list.first()
        derived_exprs.append((mag_centered / magerr_median).alias("norm"))
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
    ram_threshold_gb: float = DEFAULT_RAM_THRESHOLD_GB,
) -> pl.DataFrame:
    """
    Extract features from HuggingFace Dataset with minimal RAM usage.

    Processes each column separately to avoid loading the entire dataset
    into memory at once. Caches intermediate results to disk as parquet files.

    For systems with RAM below ram_threshold_gb, uses a two-pass approach:
    first computes and saves all features without keeping them in memory,
    then loads all cached files for final concatenation.

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
    ram_threshold_gb : float, default 512
        If system RAM is below this threshold (in GB), use two-pass mode
        to minimize memory usage during feature computation.

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

    # Check system RAM
    system_ram_gb = psutil.virtual_memory().total / (1024**3)
    low_ram_mode = system_ram_gb < ram_threshold_gb
    if low_ram_mode:
        logger.info(f"Low RAM mode: {system_ram_gb:.1f}GB < {ram_threshold_gb}GB threshold")
        if not cache_path:
            raise ValueError("cache_dir is required in low RAM mode")

    dataset_len = len(dataset)

    def is_cache_valid(name: str) -> bool:
        """Check if cache file exists and has correct row count."""
        if not cache_path:
            return False
        file_path = cache_path / f"features_{name}.parquet"
        if not file_path.exists():
            return False
        cached_rows = pl.scan_parquet(file_path).select(pl.len()).collect().item()
        return cached_rows == dataset_len

    def compute_and_save(name: str, compute_fn: Callable[[], pl.DataFrame], step: str) -> None:
        """Compute features and save to cache without returning."""
        file_path = cache_path / f"features_{name}.parquet"
        if is_cache_valid(name):
            logger.info(f"{step} {name} features already cached, skipping...")
            return

        logger.info(f"{step} Computing {name} features...")
        result = compute_fn()
        result.write_parquet(file_path, compression="zstd")
        logger.info(f"    Saved to {file_path}")
        del result
        clean_ram()

    def load_or_compute(name: str, compute_fn: Callable[[], pl.DataFrame], step: str) -> pl.DataFrame:
        """Load from cache or compute and save. Validates row count matches dataset."""
        if cache_path:
            file_path = cache_path / f"features_{name}.parquet"
            if is_cache_valid(name):
                logger.info(f"{step} Loading {name} features from cache...")
                return pl.read_parquet(file_path)
            elif file_path.exists():
                cached_rows = pl.scan_parquet(file_path).select(pl.len()).collect().item()
                logger.info(f"{step} Cache invalid ({cached_rows} rows vs {dataset_len}), recomputing...")

        logger.info(f"{step} Computing {name} features...")
        result = compute_fn()

        if cache_path:
            result.write_parquet(file_path, compression="zstd")
            logger.info(f"    Saved to {file_path}")

        return result

    # Define compute functions
    def compute_mag() -> pl.DataFrame:
        """Compute features for mag column."""
        logger.debug("    Loading mag column from dataset...")
        df_mag = dataset.select_columns(["mag"]).to_polars()
        logger.debug(f"    Loaded {df_mag.height} rows, columns: {df_mag.columns}")
        result = extract_features_polars(df_mag, normalize=normalize, float32=float32, engine=engine)
        logger.debug(f"    Result: {result.height} rows, {len(result.columns)} columns")
        del df_mag
        clean_ram()
        return result

    def compute_magerr() -> pl.DataFrame:
        """Compute features for magerr column."""
        logger.debug("    Loading magerr column from dataset...")
        df_magerr = dataset.select_columns(["magerr"]).to_polars()
        logger.debug(f"    Loaded {df_magerr.height} rows, columns: {df_magerr.columns}")
        result = extract_features_polars(df_magerr, normalize=normalize, float32=float32, engine=engine)
        logger.debug(f"    Result: {result.height} rows, {len(result.columns)} columns")
        del df_magerr
        clean_ram()
        return result

    def compute_norm() -> pl.DataFrame:
        """Compute normalized magnitude and extract features."""
        logger.debug("    Loading mag and magerr columns from dataset...")
        df = dataset.select_columns(["mag", "magerr"]).to_polars()
        logger.debug(f"    Loaded {df.height} rows, columns: {df.columns}")

        logger.debug("    Computing norm using list.eval for element-wise operations...")
        # Center mag within each list using list.eval (guaranteed element-wise operation)
        mag_centered = pl.col("mag").list.eval(pl.element() - pl.element().median())
        # Get magerr median as scalar per row
        magerr_median = pl.col("magerr").list.eval(pl.element().median()).list.first()
        # Divide centered values by magerr_median (broadcasts to list elements)
        norm_expr = mag_centered / magerr_median

        if float32:
            norm_expr = norm_expr.list.eval(pl.element().cast(pl.Float32))

        logger.debug("    Selecting norm column...")
        df_norm = df.select(norm_expr.alias("norm"))
        logger.debug(f"    df_norm: {df_norm.height} rows, columns: {df_norm.columns}, schema: {df_norm.schema}")

        # Debug: sample values from first row
        if df_norm.height > 0:
            first_norm = df_norm["norm"][0]
            if first_norm is not None and len(first_norm) > 0:
                logger.debug(f"    Sample norm[0][:5]: {list(first_norm[:5])}")
            else:
                logger.warning("    norm[0] is empty or None!")

        del df
        clean_ram()

        logger.debug("    Calling extract_features_polars on norm...")
        result = extract_features_polars(df_norm, normalize=normalize, float32=float32, engine=engine)
        logger.debug(f"    Result: {result.height} rows, {len(result.columns)} columns: {result.columns}")
        del df_norm
        clean_ram()
        return result

    def compute_mjd_diff() -> pl.DataFrame:
        """Compute features for mjd_diff (time intervals)."""
        logger.debug("    Loading mjd column from dataset...")
        df_mjd = dataset.select_columns(["mjd"]).to_polars()
        logger.debug(f"    Loaded {df_mjd.height} rows, columns: {df_mjd.columns}")

        # Cast mjd to float32 before computing diff if requested
        if float32:
            logger.debug("    Casting mjd to float32...")
            df_mjd = df_mjd.with_columns(pl.col("mjd").list.eval(pl.element().cast(pl.Float32)))

        result = extract_features_polars(df_mjd, normalize=normalize, float32=float32, engine=engine)
        logger.debug(f"    Result: {result.height} rows, {len(result.columns)} columns")
        del df_mjd
        clean_ram()
        return result

    # Determine which feature groups to process
    has_mag = "mag" in dataset.column_names
    has_magerr = "magerr" in dataset.column_names
    has_mjd = "mjd" in dataset.column_names
    has_norm = has_mag and has_magerr

    # LOW RAM MODE: First pass - compute and save all features without keeping in memory
    if low_ram_mode:
        logger.info("Pass 1: Computing and caching all features...")
        if has_mag:
            compute_and_save("mag", compute_mag, "[mag]")
        if has_magerr:
            compute_and_save("magerr", compute_magerr, "[magerr]")
        if has_norm:
            compute_and_save("norm", compute_norm, "[norm]")
        if has_mjd:
            compute_and_save("mjd_diff", compute_mjd_diff, "[mjd_diff]")
        logger.info("Pass 2: Loading cached features...")

    # Build feature DataFrames (load from cache in low RAM mode, or compute in normal mode)
    feature_dfs: list[pl.DataFrame] = []
    has_npoints = False

    if has_mag:
        features_mag = load_or_compute("mag", compute_mag, "[mag]")
        has_npoints = "npoints" in features_mag.columns
        feature_dfs.append(features_mag)
        del features_mag
        clean_ram()

    if has_magerr:
        features_magerr = load_or_compute("magerr", compute_magerr, "[magerr]")
        features_magerr, has_npoints = _handle_npoints(features_magerr, has_npoints)
        feature_dfs.append(features_magerr)
        del features_magerr
        clean_ram()

    if has_norm:
        features_norm = load_or_compute("norm", compute_norm, "[norm]")
        features_norm, has_npoints = _handle_npoints(features_norm, has_npoints)
        feature_dfs.append(features_norm)
        del features_norm
        clean_ram()

    ts_col = None
    if has_mjd:
        features_mjd = load_or_compute("mjd_diff", compute_mjd_diff, "[mjd_diff]")
        features_mjd, has_npoints = _handle_npoints(features_mjd, has_npoints)
        feature_dfs.append(features_mjd)
        del features_mjd

        # Compute ts = UTC timestamp from max(mjd) (always computed, not cached)
        df_mjd = dataset.select_columns(["mjd"]).to_polars()
        ts_col = (
            df_mjd.select(pl.col("mjd").list.max().alias("mjd_max"))
            .with_columns((pl.lit(MJD_EPOCH) + pl.duration(days=pl.col("mjd_max"))).alias("ts"))
            .select("ts")
        )
        del df_mjd
        clean_ram()

    # Add id and class columns
    meta_cols = []
    if "id" in dataset.column_names:
        meta_cols.append("id")
    if "class" in dataset.column_names:
        meta_cols.append("class")

    if meta_cols:
        logger.info("[meta] Adding metadata (id, class)...")
        df_meta = dataset.select_columns(meta_cols).to_polars()
        feature_dfs.insert(0, df_meta)
        del df_meta
        clean_ram()

    # Concatenate all feature DataFrames horizontally
    logger.info("[final] Concatenating results...")
    result = pl.concat(feature_dfs, how="horizontal")
    del feature_dfs
    clean_ram()

    # Add ts column if available
    if ts_col is not None:
        result = pl.concat([result, ts_col], how="horizontal")
        del ts_col
        clean_ram()

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
        HuggingFace Dataset with columns: mag, magerr.
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
        - norm_frac_below_3sigma: fraction of points below -3 sigma
        - norm_max_consecutive_below_2sigma: max consecutive points below -2 sigma
    """
    cache_path = Path(cache_dir) if cache_dir else None
    if cache_path:
        cache_path.mkdir(parents=True, exist_ok=True)

    cache_file = cache_path / "features_additional.parquet" if cache_path else None
    dataset_len = len(dataset)

    # Check cache validity
    if cache_file and cache_file.exists():
        cached_rows = pl.scan_parquet(cache_file).select(pl.len()).collect().item()
        if cached_rows == dataset_len:
            logger.info("[additional] Loading from cache...")
            return pl.read_parquet(cache_file)
        logger.info(f"[additional] Cache invalid ({cached_rows} vs {dataset_len}), recomputing...")

    logger.info("[additional] Computing flare-specific features...")

    # Load mag and magerr
    df = dataset.select_columns(["mag", "magerr"]).to_polars()

    # Compute norm: (mag - median(mag)) / median(magerr) per row
    mag_centered = pl.col("mag").list.eval(pl.element() - pl.element().median())
    magerr_median = pl.col("magerr").list.eval(pl.element().median()).list.first()
    norm_expr = mag_centered / magerr_median
    if float32:
        norm_expr = norm_expr.list.eval(pl.element().cast(pl.Float32))

    df = df.with_columns(norm_expr.alias("norm"))

    # Feature 1: Count of points below -3 sigma
    n_below_3sigma = (
        pl.col("norm")
        .list.eval((pl.element() < -3).cast(pl.Int32).sum())
        .list.first()
        .alias("norm_n_below_3sigma")
    )

    # Feature 2: Fraction of points below -3 sigma
    frac_below_3sigma = (
        pl.col("norm")
        .list.eval((pl.element() < -3).cast(pl.Float32).mean())
        .list.first()
        .alias("norm_frac_below_3sigma")
    )

    # Feature 3: Max consecutive points below -2 sigma (pure Polars using cumsum trick)
    # Algorithm:
    #   mask = element < -2 (True when bright)
    #   cumsum = running count of bright points
    #   base = cumsum at last non-bright point, forward-filled
    #   consecutive = cumsum - base (resets at each non-bright point)
    #   max(consecutive) = longest run of consecutive bright points
    max_consecutive = (
        pl.col("norm")
        .list.eval(
            (
                (pl.element() < -2).cast(pl.Int32).cum_sum()
                - pl.when(pl.element() >= -2)
                .then((pl.element() < -2).cast(pl.Int32).cum_sum())
                .otherwise(None)
                .forward_fill()
                .fill_null(0)
            ).max()
        )
        .list.first()
        .fill_null(0)
        .alias("norm_max_consecutive_below_2sigma")
    )

    # Execute
    result = df.lazy().select([n_below_3sigma, frac_below_3sigma, max_consecutive]).collect(engine=engine)

    if float32:
        result = result.cast({c: pl.Float32 for c in result.columns if result[c].dtype == pl.Float64})

    # Cache result
    if cache_file:
        result.write_parquet(cache_file, compression="zstd")
        logger.info(f"    Saved to {cache_file}")

    del df
    clean_ram()

    return result


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

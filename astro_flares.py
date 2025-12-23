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
MIN_ROWS_FOR_CACHING = 500_000

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
        ax2.hist(mjd_diff, bins=30)
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
        ax3.hist(norm, bins=30)
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

    fig.show()

    if not advanced:
        return

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

    # Only cache if dataset is large enough
    use_cache = cache_path is not None and dataset_len >= MIN_ROWS_FOR_CACHING
    cache_file = cache_path / f"features_main_{dataset_len}.parquet" if use_cache else None

    # Check cache validity
    if cache_file and cache_file.exists():
        logger.info("[main] Loading from cache...")
        return pl.read_parquet(cache_file, parallel="columns")

    # Migrate old cache file naming scheme if applicable
    if use_cache:
        old_cache_file = cache_path / "features_main.parquet"
        if old_cache_file.exists() and not cache_file.exists():
            old_rows = pl.scan_parquet(old_cache_file).select(pl.len()).collect().item()
            if old_rows == dataset_len:
                logger.info(f"[main] Migrating {old_cache_file.name} -> {cache_file.name}")
                old_cache_file.rename(cache_file)
                return pl.read_parquet(cache_file, parallel="columns")

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
            mag_centered = pl.col("mag").list.eval(pl.element() - pl.element().median())
            magerr_median = pl.col("magerr").list.eval(pl.element().median()).list.first()
            norm_expr = mag_centered / magerr_median
            if float32:
                norm_expr = norm_expr.list.eval(pl.element().cast(pl.Float32))
            df = df.with_columns(norm_expr.alias("norm"))

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

    # Only cache if dataset is large enough
    use_cache = cache_path is not None and dataset_len >= MIN_ROWS_FOR_CACHING
    cache_file = cache_path / f"features_additional_{dataset_len}.parquet" if use_cache else None

    # Check cache validity
    if cache_file and cache_file.exists():
        logger.info("[additional] Loading from cache...")
        return pl.read_parquet(cache_file, parallel="columns")

    # Migrate old cache file naming scheme if applicable
    if use_cache:
        old_cache_file = cache_path / "features_additional.parquet"
        if old_cache_file.exists() and not cache_file.exists():
            old_rows = pl.scan_parquet(old_cache_file).select(pl.len()).collect().item()
            if old_rows == dataset_len:
                logger.info(f"[additional] Migrating {old_cache_file.name} -> {cache_file.name}")
                old_cache_file.rename(cache_file)
                return pl.read_parquet(cache_file, parallel="columns")

    # =========================================================================
    # Feature expressions (defined once, used per batch)
    # =========================================================================

    # Norm computation
    def compute_norm_expr(float32: bool) -> pl.Expr:
        mag_centered = pl.col("mag").list.eval(pl.element() - pl.element().median())
        magerr_median = pl.col("magerr").list.eval(pl.element().median()).list.first()
        norm_expr = mag_centered / magerr_median
        if float32:
            norm_expr = norm_expr.list.eval(pl.element().cast(pl.Float32))
        return norm_expr.alias("norm")

    # Feature expressions
    n_below_3sigma = pl.col("norm").list.eval((pl.element() < -3).cast(pl.Int32).sum()).list.first().alias("norm_n_below_3sigma")

    max_consecutive = (
        pl.col("norm")
        .list.eval(
            (
                (pl.element() < -2).cast(pl.Int32).cum_sum()
                - pl.when(pl.element() >= -2).then((pl.element() < -2).cast(pl.Int32).cum_sum()).otherwise(None).forward_fill().fill_null(0)
            ).max()
        )
        .list.first()
        .fill_null(0)
        .alias("norm_max_consecutive_below_2sigma")
    )

    npoints = pl.col("norm").list.len()
    peak_idx = pl.col("norm").list.arg_min()
    rise_decay_idx_ratio = ((peak_idx.cast(pl.Float32) + 1.0) / (npoints - peak_idx).cast(pl.Float32)).alias("norm_rise_decay_idx_ratio")

    n_local_minima = (
        pl.col("norm")
        .list.eval(((pl.element().diff() < 0).cast(pl.Int32) * (pl.element().diff().shift(-1) > 0).cast(pl.Int32)).sum())
        .list.first()
        .fill_null(0)
        .alias("norm_n_local_minima")
    )

    n_zero_crossings = (
        pl.col("norm")
        .list.eval(((pl.element().sign() * pl.element().shift(1).sign()) < 0).cast(pl.Int32).sum())
        .list.first()
        .fill_null(0)
        .alias("norm_n_zero_crossings")
    )

    mjd_first = pl.col("mjd").list.first()
    mjd_last = pl.col("mjd").list.last()
    mjd_at_peak = pl.col("mjd").list.get(pl.col("norm").list.arg_min())
    rise_time = mjd_at_peak - mjd_first
    decay_time = mjd_last - mjd_at_peak
    rise_decay_time_ratio = (rise_time / (decay_time + 1e-10)).alias("norm_rise_decay_time_ratio")
    mjd_span = (mjd_last - mjd_first).alias("mjd_span")

    all_features = [
        n_below_3sigma,
        max_consecutive,
        rise_decay_idx_ratio,
        n_local_minima,
        n_zero_crossings,
        rise_decay_time_ratio,
        mjd_span,
    ]

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
        df = df.with_columns(compute_norm_expr(float32))

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

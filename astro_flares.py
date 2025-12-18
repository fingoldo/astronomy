"""
Flare analysis and visualization utilities for ZTF M-dwarf light curves.

This module provides tools for visualizing and comparing normalized magnitude
series from the ZTF M-dwarf Flares dataset.

Dataset: https://huggingface.co/datasets/snad-space/ztf-m-dwarf-flares-2025
Paper: https://arxiv.org/abs/2510.24655
"""

import logging
from datetime import datetime, timezone
from typing import Union

import numpy as np
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

DataFrameType = Union[pl.DataFrame, pd.DataFrame]

logger = logging.getLogger(__name__)


def view_series(
    df: DataFrameType,
    i: int,
    figsize: tuple[int, int] = (8, 4),
) -> None:
    """
    Plot light curve with error bars.

    Displays magnitude vs Modified Julian Date with error bars.

    Parameters
    ----------
    df : polars.DataFrame or pandas.DataFrame
        DataFrame containing 'mag', 'magerr', 'mjd', and 'class' columns.
    i : int
        Index of the record to plot.
    figsize : tuple[int, int], default (8, 4)
        Figure size as (width, height) in inches.
    """
    if isinstance(df, pl.DataFrame):
        row = df.row(i, named=True)
    elif isinstance(df, dict):
        row = df
    else:
        row = df.loc[i]

    mjd = np.array(row["mjd"])
    mag = np.array(row["mag"])
    magerr = np.array(row["magerr"])
    cls = row["class"]

    width = figsize[0] * INCHES_TO_PIXELS
    height = figsize[1] * INCHES_TO_PIXELS

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=mjd,
            y=mag,
            mode="markers",
            name="mag",
            error_y={"type": "data", "array": magerr, "visible": True},
        )
    )

    fig.update_layout(
        title=f"Record #{i} — class: {cls}",
        xaxis_title="MJD",
        yaxis_title="mag",
        width=width,
        height=height,
    )

    # Invert y-axis (brighter = lower magnitude in astronomy)
    fig.update_yaxes(autorange="reversed")

    fig.show()


def norm_series(
    df: DataFrameType,
    i: int,
    figsize: tuple[int, int] = (8, 4),
) -> None:
    """
    Plot normalized magnitude series for a given record.

    Displays a dual-axis plot showing raw magnitude values and their
    median-normalized counterparts scaled by median error.

    Parameters
    ----------
    df : polars.DataFrame or pandas.DataFrame
        DataFrame containing 'mag', 'magerr', and 'class' columns.
        Each row represents a light curve observation.
    i : int
        Index of the record to plot.
    figsize : tuple[int, int], default (8, 4)
        Figure size as (width, height) in inches.

    Notes
    -----
    Normalization formula: (mag - median(mag)) / median(magerr)

    The correlation coefficient between raw and normalized magnitude
    is displayed in the plot title.
    """
    if isinstance(df, pl.DataFrame):
        row = df.row(i, named=True)
    else:
        row = df.loc[i]

    mag = np.array(row["mag"])
    magerr = np.array(row["magerr"])
    cls = row["class"]

    norm = (mag - np.median(mag)) / np.median(magerr)
    correlation = np.corrcoef(mag, norm)[0, 1]

    width = figsize[0] * INCHES_TO_PIXELS
    height = figsize[1] * INCHES_TO_PIXELS

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
        title=f"Record #{i} — class: {cls}, corr={correlation:.3f}",
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

    See Also
    --------
    norm_series : Plot a single normalized magnitude series.
    """
    if isinstance(df, pl.DataFrame):
        i0 = df.with_row_index().filter(pl.col("class") == 0).sample(1)["index"][0]
        i1 = df.with_row_index().filter(pl.col("class") == 1).sample(1)["index"][0]
    else:
        i0 = df[df["class"] == 0].sample(1).index[0]
        i1 = df[df["class"] == 1].sample(1).index[0]

    norm_series(df, i0, figsize)
    norm_series(df, i1, figsize)


def extract_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract statistical features from light curve data for ML.

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

    def compute_array_features(arr: np.ndarray, prefix: str) -> dict:
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

    records = []
    for row in df.iter_rows(named=True):
        mag = np.array(row["mag"])
        magerr = np.array(row["magerr"])
        norm = (mag - np.median(mag)) / np.median(magerr)

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
    """

    def normalize_expr(c: pl.Expr) -> pl.Expr:
        if normalize is None:
            return c
        elif normalize == "minmax":
            return (c - c.list.min()) / (c.list.max() - c.list.min())
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
        exprs.extend([
            # Positional
            c.list.first().alias(f"{col}_first"),
            c.list.last().alias(f"{col}_last"),
            c.list.arg_min().alias(f"{col}_arg_min"),
            c.list.arg_max().alias(f"{col}_arg_max"),
            # Uniqueness & structure
            c.list.n_unique().alias(f"{col}_n_unique"),
            c.list.eval(pl.element().diff().sign().diff().ne(0).sum()).list.first().alias(f"{col}_trend_changes"),
        ])
        return exprs

    cols = set(df.columns)
    has_mag = "mag" in cols
    has_magerr = "magerr" in cols
    has_mjd = "mjd" in cols
    has_class = "class" in cols

    # Build derived columns
    derived_exprs = []
    if has_mag and has_magerr:
        mag_median = pl.col("mag").list.eval(pl.element().median()).list.first()
        magerr_median = pl.col("magerr").list.eval(pl.element().median()).list.first()
        derived_exprs.append(((pl.col("mag") - mag_median) / magerr_median).alias("norm"))
    if has_mjd:
        derived_exprs.append(
            pl.col("mjd").list.eval(pl.element().diff().drop_nulls()).alias("mjd_diff")
        )

    df_enriched = df.with_columns(derived_exprs) if derived_exprs else df

    # Build select expressions
    select_exprs: list[pl.Expr | str] = []
    if "id" in cols:
        select_exprs.append("id")
    if has_class:
        select_exprs.append("class")

    # npoints from first available list column
    for len_col in ("mag", "magerr", "mjd"):
        if len_col in cols:
            select_exprs.append(pl.col(len_col).list.len().alias("npoints"))
            break

    # Add stats for each available column
    if has_mag:
        select_exprs.extend(stats_exprs("mag"))
    if has_magerr:
        select_exprs.extend(stats_exprs("magerr"))
    if has_mag and has_magerr:
        select_exprs.extend(stats_exprs("norm"))
    if has_mjd:
        select_exprs.extend(stats_exprs("mjd_diff"))

    result = df_enriched.select(select_exprs)
    if float32:
        result = result.cast({c: pl.Float32 for c in result.columns if result[c].dtype == pl.Float64})
    return result


def extract_features_sparingly(
    dataset: Dataset,
    normalize: str | None = None,
    float32: bool = True,
) -> pl.DataFrame:
    """
    Extract features from HuggingFace Dataset with minimal RAM usage.

    Processes each column separately to avoid loading the entire dataset
    into memory at once. Calls clean_ram() after each intermediate step.

    Parameters
    ----------
    dataset : datasets.Dataset
        HuggingFace Dataset with columns: id, class, mag, magerr, mjd.
    normalize : str or None, default None
        Normalization method passed to extract_features_polars.
    float32 : bool, default True
        If True, cast float columns to Float32 to save memory.

    Returns
    -------
    polars.DataFrame
        DataFrame with id, class, npoints, ts (UTC timestamp from max mjd),
        and statistical features for mag, magerr, norm, mjd_diff.
    """
    feature_dfs: list[pl.DataFrame] = []
    has_npoints = False

    # Process mag features
    if "mag" in dataset.column_names:
        logger.info("[1/6] Computing mag features...")
        df_mag = dataset.select_columns(["mag"]).to_polars()
        features_mag = extract_features_polars(df_mag, normalize=normalize, float32=float32)
        has_npoints = "npoints" in features_mag.columns
        feature_dfs.append(features_mag)
        del df_mag, features_mag
        clean_ram()

    # Process magerr features
    if "magerr" in dataset.column_names:
        logger.info("[2/6] Computing magerr features...")
        df_magerr = dataset.select_columns(["magerr"]).to_polars()
        features_magerr = extract_features_polars(df_magerr, normalize=normalize, float32=float32)
        if has_npoints and "npoints" in features_magerr.columns:
            features_magerr = features_magerr.drop("npoints")
        elif "npoints" in features_magerr.columns:
            has_npoints = True
        feature_dfs.append(features_magerr)
        del df_magerr, features_magerr
        clean_ram()

    # Process norm features (requires both mag and magerr)
    if "mag" in dataset.column_names and "magerr" in dataset.column_names:
        logger.info("[3/6] Computing norm features...")
        df_norm = dataset.select_columns(["mag", "magerr"]).to_polars()
        features_norm = extract_features_polars(df_norm, normalize=normalize, float32=float32)
        # Keep only norm_* columns
        norm_cols = [c for c in features_norm.columns if c.startswith("norm_")]
        features_norm = features_norm.select(norm_cols)
        feature_dfs.append(features_norm)
        del df_norm, features_norm
        clean_ram()

    # Process mjd_diff features and compute ts
    ts_col = None
    if "mjd" in dataset.column_names:
        logger.info("[4/6] Computing mjd_diff features...")
        df_mjd = dataset.select_columns(["mjd"]).to_polars()
        features_mjd = extract_features_polars(df_mjd, normalize=normalize, float32=float32)
        if has_npoints and "npoints" in features_mjd.columns:
            features_mjd = features_mjd.drop("npoints")
        elif "npoints" in features_mjd.columns:
            has_npoints = True
        feature_dfs.append(features_mjd)

        # Compute ts = UTC timestamp from max(mjd)
        ts_col = df_mjd.select(
            pl.col("mjd").list.max().alias("mjd_max")
        ).with_columns(
            (pl.lit(MJD_EPOCH) + pl.duration(days=pl.col("mjd_max"))).alias("ts")
        ).select("ts")

        del df_mjd, features_mjd
        clean_ram()

    # Add id and class columns
    meta_cols = []
    if "id" in dataset.column_names:
        meta_cols.append("id")
    if "class" in dataset.column_names:
        meta_cols.append("class")

    if meta_cols:
        logger.info("[5/6] Adding metadata (id, class)...")
        df_meta = dataset.select_columns(meta_cols).to_polars()
        feature_dfs.insert(0, df_meta)
        del df_meta
        clean_ram()

    # Concatenate all feature DataFrames horizontally
    logger.info("[6/6] Concatenating results...")
    result = pl.concat(feature_dfs, how="horizontal")
    del feature_dfs
    clean_ram()

    # Add ts column if available
    if ts_col is not None:
        result = pl.concat([result, ts_col], how="horizontal")
        del ts_col
        clean_ram()

    return result

"""
Flare analysis and visualization utilities for ZTF M-dwarf light curves.

This module provides tools for visualizing and comparing normalized magnitude
series from the ZTF M-dwarf Flares dataset.

Dataset: https://huggingface.co/datasets/snad-space/ztf-m-dwarf-flares-2025
Paper: https://arxiv.org/abs/2510.24655
"""

from typing import Union

import numpy as np
import pandas as pd
import polars as pl
from datasets import load_dataset
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# Conversion factor from matplotlib inches to plotly pixels
INCHES_TO_PIXELS = 100

DataFrameType = Union[pl.DataFrame, pd.DataFrame]


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
) -> pl.DataFrame:
    """
    Extract statistical features using native Polars operations (parallelized).

    Vectorized version of extract_features() that uses Polars' list operations
    for parallel computation across all rows.

    Parameters
    ----------
    df : polars.DataFrame
        DataFrame with columns: id, class, mjd, mag, magerr.
        mag and magerr must be List(Float64) columns.
    normalize : str or None, default None
        Normalization method to apply before computing statistics:
        - None: No normalization (raw values)
        - "minmax": (x - min) / (max - min), scales to [0, 1]
        - "zscore": (x - mean) / std, standardizes to mean=0, std=1

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
    - amplitude_sigma: Peak-to-peak amplitude
    - mean: Arithmetic mean
    - median: Median value
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
        return [
            c.list.std().alias(f"{col}_std"),
            c.list.eval(pl.element().skew()).list.first().alias(f"{col}_skewness"),
            c.list.eval(pl.element().kurtosis()).list.first().alias(f"{col}_kurtosis"),
            (c.list.max() - c.list.min()).alias(f"{col}_amplitude_sigma"),
            c.list.mean().alias(f"{col}_mean"),
            c.list.eval(pl.element().median()).list.first().alias(f"{col}_median"),
        ]

    # Compute norm as element-wise: (mag - median(mag)) / median(magerr)
    mag_median = pl.col("mag").list.eval(pl.element().median()).list.first()
    magerr_median = pl.col("magerr").list.eval(pl.element().median()).list.first()

    df_with_norm = df.with_columns(((pl.col("mag") - mag_median) / magerr_median).alias("norm"))

    return df_with_norm.select(
        "id",
        "class",
        pl.col("mag").list.len().alias("npoints"),
        *stats_exprs("mag"),
        *stats_exprs("magerr"),
        *stats_exprs("norm"),
    )

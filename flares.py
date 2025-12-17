"""Flare analysis and visualization utilities."""

import numpy as np
import polars as pl
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def norm_series(df, i: int, figsize=(8, 4)):
    """
    Plot normalized magnitude series for a given record.

    Parameters
    ----------
    df : polars.DataFrame or pandas.DataFrame
        DataFrame containing 'mag', 'magerr', and 'class' columns.
    i : int
        Index of the record to plot.
    figsize : tuple, optional
        Figure size in (width, height) inches. Default is (8, 4).
    """
    if isinstance(df, pl.DataFrame):
        row = df.row(i, named=True)
    else:
        row = df.loc[i]
    mag = np.array(row["mag"])
    magerr = np.array(row["magerr"])
    cls = row["class"]
    norm = (mag - np.median(mag)) / magerr
    # matplotlib inches -> plotly pixels
    width = int(figsize[0] * 100)
    height = int(figsize[1] * 100)
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
        title=f"Record #{i} - class: {cls}, corr={np.corrcoef(mag, norm)[0][1]:.3f}",
        width=width,
        height=height,
        legend=dict(x=0.01, y=0.99),
    )
    fig.update_yaxes(title_text="mag", secondary_y=False)
    fig.update_yaxes(title_text="norm", secondary_y=True)
    fig.show()


def compare_classes(df, figsize=(8, 4)):
    """
    Compare random samples from each class by plotting their normalized series.

    Parameters
    ----------
    df : polars.DataFrame or pandas.DataFrame
        DataFrame containing 'mag', 'magerr', and 'class' columns.
    figsize : tuple, optional
        Figure size in (width, height) inches. Default is (8, 4).
    """
    if isinstance(df, pl.DataFrame):
        i0 = df.with_row_index().filter(pl.col("class") == 0).sample(1)["index"][0]
        i1 = df.with_row_index().filter(pl.col("class") == 1).sample(1)["index"][0]
    else:
        i0 = df[df["class"] == 0].sample(1).index[0]
        i1 = df[df["class"] == 1].sample(1).index[0]

    norm_series(df, i0, figsize)
    norm_series(df, i1, figsize)

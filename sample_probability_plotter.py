"""
Sample Probability Plotter

Samples and plots light curves from a probability parquet file produced by
the active learning pipeline (e.g., iter0200_all_probabilities.parquet).

Usage:
    python sample_probability_plotter.py \
        --parquet_path path/to/iter0200_all_probabilities.parquet \
        --dataset path/to/hf_dataset \
        --min_prob 0.3 --max_prob 0.7 \
        --num_samples 50 \
        --output_dir sampled_plots/
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import polars as pl

# Optional HuggingFace datasets
try:
    from datasets import Dataset, load_from_disk, load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    Dataset = None
    load_from_disk = None
    load_dataset = None

# Optional astro_flares for plotting
try:
    from astro_flares import view_series
    VIEW_SERIES_AVAILABLE = True
except ImportError:
    VIEW_SERIES_AVAILABLE = False
    view_series = None

# Jupyter detection
try:
    from pyutilz.pythonlib import is_jupyter_notebook
except ImportError:
    def is_jupyter_notebook():
        return False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def plot_sample_from_dataset(
    row_index: int,
    dataset,
    output_dir: Path,
    probability: float,
    sample_id: str | int | None = None,
    singlepoint_min_outlying_factor: float = 10.0,
    show: bool = False,
) -> bool:
    """
    Plot a single sample from the dataset.

    Parameters
    ----------
    row_index : int
        Index into the HuggingFace dataset.
    dataset : HuggingFace Dataset
        The light curve dataset.
    output_dir : Path
        Directory to save plots.
    probability : float
        P(flare) probability for this sample.
    sample_id : str | int | None
        Optional sample ID for filename.
    singlepoint_min_outlying_factor : float
        Outlier factor for view_series.
    show : bool
        Whether to display the plot.

    Returns
    -------
    bool
        True if plot was created successfully, False otherwise.
    """
    if not VIEW_SERIES_AVAILABLE:
        logger.error("view_series not available. Install astro_flares package.")
        return False

    try:
        # Get the light curve data from HuggingFace dataset
        lc_data = dataset[row_index]

        # Determine backend
        backend = "plotly" if is_jupyter_notebook() else "matplotlib"

        # Build filename
        id_str = sample_id if sample_id is not None else row_index
        prob_pct = probability * 100
        plot_file = output_dir / f"sample_{id_str}_row{row_index}_P{prob_pct:.2f}pct.png"

        # Plot with probability in title
        title = f"P(flare)={prob_pct:.2f}%"
        view_series(
            lc_data,
            index=row_index,
            backend=backend,
            singlepoint_min_outlying_factor=singlepoint_min_outlying_factor,
            plot_file=str(plot_file),
            title=title,
            show=show,
        )
        logger.info(f"Saved plot: {plot_file.name}")
        return True

    except (IndexError, KeyError, ValueError, OSError) as e:
        logger.warning(f"Failed to plot sample {row_index}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Sample and plot light curves from probability parquet files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--parquet_path",
        type=Path,
        required=True,
        help="Path to the probability parquet file (e.g., iter0200_all_probabilities.parquet)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HuggingFace dataset: local path or Hub name (e.g., 'snad-space/ztf-m-dwarf-flares-2025')",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (for Hub datasets)",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
        help="HuggingFace cache directory (e.g., 'R:/Caches/huggingface')",
    )
    parser.add_argument(
        "--min_prob",
        type=float,
        required=True,
        help="Minimum probability threshold (inclusive)",
    )
    parser.add_argument(
        "--max_prob",
        type=float,
        required=True,
        help="Maximum probability threshold (inclusive)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        required=True,
        help="Number of random samples to plot",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for plots",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--outlier_factor",
        type=float,
        default=10.0,
        help="Singlepoint min outlying factor for view_series",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively",
    )

    args = parser.parse_args()

    # Validate dependencies
    if not HF_AVAILABLE:
        logger.error("HuggingFace datasets not available. Install with: pip install datasets")
        return 1

    if not VIEW_SERIES_AVAILABLE:
        logger.error("astro_flares not available. Install the astro_flares package.")
        return 1

    # Validate inputs
    if not args.parquet_path.exists():
        logger.error(f"Parquet file not found: {args.parquet_path}")
        return 1

    if args.min_prob < 0 or args.min_prob > 1:
        logger.error(f"min_prob must be between 0 and 1, got {args.min_prob}")
        return 1

    if args.max_prob < 0 or args.max_prob > 1:
        logger.error(f"max_prob must be between 0 and 1, got {args.max_prob}")
        return 1

    if args.min_prob > args.max_prob:
        logger.error(f"min_prob ({args.min_prob}) cannot be greater than max_prob ({args.max_prob})")
        return 1

    if args.num_samples < 1:
        logger.error(f"num_samples must be at least 1, got {args.num_samples}")
        return 1

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    # Load parquet file
    logger.info(f"Loading parquet file: {args.parquet_path}")
    df = pl.read_parquet(args.parquet_path)
    logger.info(f"Loaded {len(df):,} samples with columns: {df.columns}")

    # Detect probability column name (support both formats)
    if "flare_prob" in df.columns:
        prob_col = "flare_prob"
    elif "probability" in df.columns:
        prob_col = "probability"
    else:
        logger.error(f"No probability column found. Expected 'flare_prob' or 'probability', got: {df.columns}")
        return 1
    logger.info(f"Using probability column: {prob_col}")

    # Filter by probability range
    filtered_df = df.filter(
        (pl.col(prob_col) >= args.min_prob) &
        (pl.col(prob_col) <= args.max_prob)
    )
    logger.info(
        f"Filtered to {len(filtered_df):,} samples with P(flare) in [{args.min_prob}, {args.max_prob}]"
    )

    if len(filtered_df) == 0:
        logger.error("No samples found in the specified probability range.")
        return 1

    # Sample randomly
    n_to_sample = min(args.num_samples, len(filtered_df))
    if n_to_sample < args.num_samples:
        logger.warning(
            f"Only {len(filtered_df)} samples available, sampling all of them instead of {args.num_samples}"
        )

    sampled_df = filtered_df.sample(n=n_to_sample, shuffle=True)
    logger.info(f"Randomly sampled {n_to_sample} samples")

    # Load dataset (local path or Hub name)
    dataset_path = Path(args.dataset)
    if dataset_path.exists():
        logger.info(f"Loading local dataset from: {args.dataset}")
        dataset = load_from_disk(str(dataset_path))
    else:
        cache_dir = str(args.cache_dir) if args.cache_dir else None
        logger.info(f"Loading dataset from HuggingFace Hub: {args.dataset} (split={args.split}, cache_dir={cache_dir})")
        dataset = load_dataset(args.dataset, split=args.split, cache_dir=cache_dir)
    logger.info(f"Dataset loaded with {len(dataset):,} samples")

    # Plot each sample
    success_count = 0
    fail_count = 0

    for i, row in enumerate(sampled_df.iter_rows(named=True)):
        row_index = row["row_index"]
        flare_prob = row[prob_col]
        sample_id = row.get("id", None)

        logger.info(f"Plotting sample {i+1}/{n_to_sample}: row_index={row_index}, P(flare)={flare_prob:.4f}")

        if row_index >= len(dataset):
            logger.warning(f"row_index {row_index} exceeds dataset size {len(dataset)}, skipping")
            fail_count += 1
            continue

        success = plot_sample_from_dataset(
            row_index=row_index,
            dataset=dataset,
            output_dir=args.output_dir,
            probability=flare_prob,
            sample_id=sample_id,
            singlepoint_min_outlying_factor=args.outlier_factor,
            show=args.show,
        )

        if success:
            success_count += 1
        else:
            fail_count += 1

    logger.info(f"Done! Successfully plotted {success_count} samples, {fail_count} failures.")
    return 0


if __name__ == "__main__":
    exit(main())

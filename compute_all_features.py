"""
Compute ALL features for the ZTF M-dwarf flares dataset.

This script extracts all features (main, additional, fraction, wavelet, argextremum)
in one unified pass and saves them to a parquet file for later use in flare detection models.

Usage:
    python compute_all_features.py --output-dir ./output --cache-dir ~/.cache/huggingface
    python compute_all_features.py --split train --argextremum-col mag --od-iqr 10

Environment variables:
    HF_HOME: HuggingFace cache directory (default: ~/.cache/huggingface)
    ASTRO_DATA_DIR: Output directory for data files (default: ./data)
"""

import argparse
import logging
import os
from pathlib import Path

from astro_flares import extract_all_features

logger = logging.getLogger(__name__)

# Configuration
DATASET_NAME = "snad-space/ztf-m-dwarf-flares-2025"
DEFAULT_SPLIT = "target"


def get_default_cache_dir() -> Path:
    """Get default HuggingFace cache directory."""
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def get_default_output_dir() -> Path:
    """Get default output directory."""
    return Path(os.environ.get("ASTRO_DATA_DIR", Path.cwd() / "data"))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute ALL features for ZTF M-dwarf flares dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=get_default_cache_dir(),
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=get_default_output_dir(),
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=DEFAULT_SPLIT,
        help="Dataset split to process (target, train, test)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="all_features.parquet",
        help="Output filename",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 = all physical cores)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Rows per chunk for parallel processing",
    )
    # Wavelet parameters
    parser.add_argument(
        "--wavelets",
        type=str,
        nargs="+",
        default=None,
        help="Wavelet types to compute (default: haar, db4, sym4, coif2)",
    )
    parser.add_argument(
        "--max-level",
        type=int,
        default=6,
        help="Maximum wavelet decomposition level",
    )
    parser.add_argument(
        "--no-interpolate",
        action="store_true",
        help="Disable interpolation to regular time grid before DWT",
    )
    parser.add_argument(
        "--n-interp-points",
        type=int,
        default=64,
        help="Number of points for interpolation grid",
    )
    # Argextremum parameters
    parser.add_argument(
        "--argextremum-col",
        type=str,
        default="mag",
        help="Column for argmax/argmin split (e.g., mag, norm). Use --no-argextremum to disable",
    )
    parser.add_argument(
        "--no-argextremum",
        action="store_true",
        help="Disable argextremum statistics",
    )
    parser.add_argument(
        "--no-argextremum-additional",
        action="store_true",
        help="Disable additional stats (skewness, kurtosis, etc.) for argextremum sub-series",
    )
    # Outlier detection parameters
    parser.add_argument(
        "--od-col",
        type=str,
        default="mag",
        help="Column for outlier detection",
    )
    parser.add_argument(
        "--od-iqr",
        type=float,
        default=40.0,
        help="IQR multiplier for outlier detection (0 = disabled)",
    )
    # Other parameters
    parser.add_argument(
        "--float64",
        action="store_true",
        help="Use float64 instead of float32",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    args = parse_args()
    output_file = args.output_dir / args.output_file

    logger.info(f"Computing ALL features for {DATASET_NAME} [{args.split}]")
    logger.info(f"Cache directory: {args.cache_dir}")
    logger.info(f"Output file: {output_file}")
    argext_col = None if args.no_argextremum else args.argextremum_col
    if argext_col:
        logger.info(f"Argextremum column: {argext_col}")
    if args.od_iqr > 0:
        logger.info(f"Outlier detection: {args.od_col} with IQR x{args.od_iqr}")

    features_df = extract_all_features(
        dataset_name=DATASET_NAME,
        split=args.split,
        hf_cache_dir=str(args.cache_dir),
        cache_dir=args.output_dir,
        n_jobs=args.n_jobs,
        chunk_size=args.chunk_size,
        float32=not args.float64,
        wavelets=args.wavelets,
        max_level=args.max_level,
        interpolate=not args.no_interpolate,
        n_interp_points=args.n_interp_points,
        argextremum_stats_col=argext_col,
        argextremum_compute_additional_stats=not args.no_argextremum_additional,
        od_col=args.od_col,
        od_iqr=args.od_iqr,
    )

    logger.info(f"Computed {len(features_df)} rows with {len(features_df.columns)} columns")
    logger.debug(f"Columns: {list(features_df.columns)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    features_df.write_parquet(output_file)
    logger.info(f"Saved to: {output_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

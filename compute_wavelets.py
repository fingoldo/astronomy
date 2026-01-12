"""
Compute wavelet features for the ZTF M-dwarf flares dataset.

This script extracts wavelet-based features from the target split and saves
them to a parquet file for later use in flare detection models.

Usage:
    python compute_wavelets.py --output-dir ./output --cache-dir ~/.cache/huggingface

Environment variables:
    HF_HOME: HuggingFace cache directory (default: ~/.cache/huggingface)
    ASTRO_DATA_DIR: Output directory for data files (default: ./data)
"""

import argparse
import logging
import os
from pathlib import Path

from astro_flares import extract_wavelet_features_sparingly

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
        description="Compute wavelet features for ZTF M-dwarf flares dataset",
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
        help="Dataset split to process",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="wavelets.parquet",
        help="Output filename",
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

    logger.info(f"Computing wavelet features for {DATASET_NAME} [{args.split}]")
    logger.info(f"Cache directory: {args.cache_dir}")
    logger.info(f"Output file: {output_file}")

    wavelet_df = extract_wavelet_features_sparingly(
        dataset_name=DATASET_NAME,
        split=args.split,
        hf_cache_dir=str(args.cache_dir),
        cache_dir=args.output_dir,
    )

    logger.info(f"Computed {len(wavelet_df)} rows with {len(wavelet_df.columns)} columns")
    logger.debug(f"Columns: {list(wavelet_df.columns)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    wavelet_df.write_parquet(output_file)
    logger.info(f"Saved to: {output_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

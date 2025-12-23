"""
Compute wavelet features for the ZTF M-dwarf flares dataset.

This script extracts wavelet-based features from the target split and saves
them to a parquet file for later use in flare detection models.
"""

from pathlib import Path
from datasets import load_dataset
from astro_flares import extract_wavelet_features_sparingly

# Configuration
DATASET_NAME = "snad-space/ztf-m-dwarf-flares-2025"
CACHE_DIR = r"R:\Caches\huggingface"
OUTPUT_DIR = Path(r"R:\Data\Astronomy")
OUTPUT_FILE = OUTPUT_DIR / "wavelets.parquet"


def main():
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, cache_dir=CACHE_DIR)

    target_dataset = dataset["target"]
    print(f"Target split: {len(target_dataset)} samples")

    print("Computing wavelet features (using all physical cores)...")
    wavelet_df = extract_wavelet_features_sparingly(
        target_dataset,
        wavelets=["haar", "db4", "sym4"],
        max_level=4,
        n_jobs=-1,
    )

    print(f"Computed {len(wavelet_df)} rows with {len(wavelet_df.columns)} columns")
    print(f"Columns: {wavelet_df.columns}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wavelet_df.write_parquet(OUTPUT_FILE)
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

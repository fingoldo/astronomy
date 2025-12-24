"""
Compute wavelet features for the ZTF M-dwarf flares dataset.

This script extracts wavelet-based features from the target split and saves
them to a parquet file for later use in flare detection models.
"""

from pathlib import Path
from astro_flares import extract_wavelet_features_sparingly

# Configuration
DATASET_NAME = "snad-space/ztf-m-dwarf-flares-2025"
HF_CACHE_DIR = r"R:\Caches\huggingface"
SPLIT = "target"
OUTPUT_DIR = Path(r"R:\Data\Astronomy")
OUTPUT_FILE = OUTPUT_DIR / "wavelets.parquet"


def main():
    print(f"Computing wavelet features for {DATASET_NAME} [{SPLIT}]...")
    print("Each worker will load the dataset independently from cache.")

    wavelet_df = extract_wavelet_features_sparingly(dataset_name=DATASET_NAME, split=SPLIT, hf_cache_dir=HF_CACHE_DIR, cache_dir=OUTPUT_DIR)

    print(f"Computed {len(wavelet_df)} rows with {len(wavelet_df.columns)} columns")
    print(f"Columns: {wavelet_df.columns}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wavelet_df.write_parquet(OUTPUT_FILE)
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

"""Tests for compute_wavelets module."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from compute_wavelets import (
    get_default_cache_dir,
    get_default_output_dir,
    parse_args,
)


class TestGetDefaultCacheDir:
    """Tests for get_default_cache_dir function."""

    def test_uses_hf_home_env_var(self, monkeypatch):
        """Should use HF_HOME environment variable when set."""
        monkeypatch.setenv("HF_HOME", "/custom/cache/dir")

        result = get_default_cache_dir()

        assert result == Path("/custom/cache/dir")

    def test_falls_back_to_home_cache(self, monkeypatch):
        """Should fall back to ~/.cache/huggingface when HF_HOME not set."""
        monkeypatch.delenv("HF_HOME", raising=False)

        result = get_default_cache_dir()

        assert result == Path.home() / ".cache" / "huggingface"


class TestGetDefaultOutputDir:
    """Tests for get_default_output_dir function."""

    def test_uses_astro_data_dir_env_var(self, monkeypatch):
        """Should use ASTRO_DATA_DIR environment variable when set."""
        monkeypatch.setenv("ASTRO_DATA_DIR", "/custom/output/dir")

        result = get_default_output_dir()

        assert result == Path("/custom/output/dir")

    def test_falls_back_to_cwd_data(self, monkeypatch):
        """Should fall back to ./data when ASTRO_DATA_DIR not set."""
        monkeypatch.delenv("ASTRO_DATA_DIR", raising=False)

        result = get_default_output_dir()

        assert result == Path.cwd() / "data"


class TestParseArgs:
    """Tests for parse_args function."""

    def test_default_values(self, monkeypatch):
        """Should use default values when no args provided."""
        monkeypatch.delenv("HF_HOME", raising=False)
        monkeypatch.delenv("ASTRO_DATA_DIR", raising=False)

        with patch("sys.argv", ["compute_wavelets.py"]):
            args = parse_args()

        assert args.split == "target"
        assert args.output_file == "wavelets.parquet"
        assert args.cache_dir == Path.home() / ".cache" / "huggingface"
        assert args.output_dir == Path.cwd() / "data"

    def test_custom_split(self, monkeypatch):
        """Should accept custom split argument."""
        monkeypatch.delenv("HF_HOME", raising=False)
        monkeypatch.delenv("ASTRO_DATA_DIR", raising=False)

        with patch("sys.argv", ["compute_wavelets.py", "--split", "train"]):
            args = parse_args()

        assert args.split == "train"

    def test_custom_output_file(self, monkeypatch):
        """Should accept custom output file argument."""
        monkeypatch.delenv("HF_HOME", raising=False)
        monkeypatch.delenv("ASTRO_DATA_DIR", raising=False)

        with patch("sys.argv", ["compute_wavelets.py", "--output-file", "custom.parquet"]):
            args = parse_args()

        assert args.output_file == "custom.parquet"

    def test_custom_cache_dir(self, monkeypatch):
        """Should accept custom cache directory argument."""
        monkeypatch.delenv("HF_HOME", raising=False)
        monkeypatch.delenv("ASTRO_DATA_DIR", raising=False)

        with patch("sys.argv", ["compute_wavelets.py", "--cache-dir", "/my/cache"]):
            args = parse_args()

        assert args.cache_dir == Path("/my/cache")

    def test_custom_output_dir(self, monkeypatch):
        """Should accept custom output directory argument."""
        monkeypatch.delenv("HF_HOME", raising=False)
        monkeypatch.delenv("ASTRO_DATA_DIR", raising=False)

        with patch("sys.argv", ["compute_wavelets.py", "--output-dir", "/my/output"]):
            args = parse_args()

        assert args.output_dir == Path("/my/output")

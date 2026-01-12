"""Tests for flare_labeller module."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from flare_labeller import (
    _ITER_PATTERN,
    _PROBABILITY_PATTERN,
    _ROW_INDEX_PATTERN,
    _SOURCE_ID_PATTERN,
    _discover_image_files,
    load_saved_state,
)


class TestRegexPatterns:
    """Tests for pre-compiled regex patterns."""

    def test_probability_pattern_basic(self):
        """Extract probability from standard filename."""
        filename = "added_ZTF123_MJD58795_row123_P52.67pct_cleaned.png"
        match = _PROBABILITY_PATTERN.search(filename)

        assert match is not None
        assert float(match.group(1)) == 52.67

    def test_probability_pattern_integer(self):
        """Extract integer probability."""
        filename = "sample_P50pct_test.png"  # Pattern requires trailing underscore
        match = _PROBABILITY_PATTERN.search(filename)

        assert match is not None
        assert float(match.group(1)) == 50

    def test_probability_pattern_no_match(self):
        """No match when pattern not present."""
        filename = "sample_without_probability.png"
        match = _PROBABILITY_PATTERN.search(filename)

        assert match is None

    def test_source_id_pattern_removed_prefix(self):
        """Extract source ID with removed_ prefix."""
        filename = "removed_ZTFDR567208300020011_MJD58795.44066_row58961985_P60.69pct_cleaned.png"
        match = _SOURCE_ID_PATTERN.search(filename)

        assert match is not None
        assert match.group(1) == "ZTFDR567208300020011_MJD58795.44066"

    def test_source_id_pattern_added_prefix(self):
        """Extract source ID with added_ prefix."""
        filename = "added_ZTF123_MJD58000.12345_row123_P50pct.png"
        match = _SOURCE_ID_PATTERN.search(filename)

        assert match is not None
        assert match.group(1) == "ZTF123_MJD58000.12345"

    def test_source_id_pattern_no_prefix(self):
        """Extract source ID without prefix."""
        filename = "ZTF123_MJD58000_row123_P50pct.png"
        match = _SOURCE_ID_PATTERN.search(filename)

        assert match is not None
        assert match.group(1) == "ZTF123_MJD58000"

    def test_row_index_pattern(self):
        """Extract row index from filename."""
        filename = "added_ZTF123_MJD58795_row47833_P50pct.png"
        match = _ROW_INDEX_PATTERN.search(filename)

        assert match is not None
        assert int(match.group(1)) == 47833

    def test_row_index_pattern_large_number(self):
        """Extract large row index."""
        filename = "sample_row58961985_test.png"
        match = _ROW_INDEX_PATTERN.search(filename)

        assert match is not None
        assert int(match.group(1)) == 58961985

    def test_iter_pattern(self):
        """Extract iteration number from path."""
        path = "sample_plots/iter030/pseudo_pos/sample.png"
        match = _ITER_PATTERN.search(path)

        assert match is not None
        assert int(match.group(1)) == 30

    def test_iter_pattern_three_digits(self):
        """Extract three-digit iteration number."""
        path = "iter123/sample.png"
        match = _ITER_PATTERN.search(path)

        assert match is not None
        assert int(match.group(1)) == 123


class TestLoadSavedState:
    """Tests for load_saved_state function."""

    def test_returns_none_when_no_file(self, tmp_path, monkeypatch):
        """Should return None when save file doesn't exist."""
        fake_save_file = tmp_path / ".nonexistent.json"
        monkeypatch.setattr("flare_labeller.SAVE_FILE", fake_save_file)

        result = load_saved_state()

        assert result is None

    def test_loads_valid_json(self, tmp_path, monkeypatch):
        """Should load valid JSON state file."""
        fake_save_file = tmp_path / ".test_state.json"
        state = {
            "folder": "/test/path",
            "pos_indices": [1, 2, 3],
            "neg_indices": [4, 5],
            "current_index": 10,
        }
        fake_save_file.write_text(json.dumps(state))
        monkeypatch.setattr("flare_labeller.SAVE_FILE", fake_save_file)

        result = load_saved_state()

        assert result is not None
        assert result["folder"] == "/test/path"
        assert result["pos_indices"] == [1, 2, 3]
        assert result["neg_indices"] == [4, 5]
        assert result["current_index"] == 10

    def test_handles_corrupted_json(self, tmp_path, monkeypatch):
        """Should return None for corrupted JSON."""
        fake_save_file = tmp_path / ".corrupted.json"
        fake_save_file.write_text("{invalid json")
        monkeypatch.setattr("flare_labeller.SAVE_FILE", fake_save_file)

        result = load_saved_state()

        assert result is None


class TestDiscoverImageFiles:
    """Tests for _discover_image_files function."""

    def test_empty_folder(self, tmp_path):
        """Should return empty list for folder with no images."""

        def mock_extract_prob(p):
            return None

        def mock_extract_id(p):
            return None

        files, seen = _discover_image_files(tmp_path, set(), mock_extract_prob, mock_extract_id)

        assert files == []
        assert seen == set()

    def test_direct_folder_mode(self, tmp_path):
        """Should find PNG files in direct folder mode."""
        # Create some PNG files
        (tmp_path / "image1.png").touch()
        (tmp_path / "image2.png").touch()
        (tmp_path / "not_image.txt").touch()

        def mock_extract_prob(p):
            return None

        def mock_extract_id(p):
            return None

        files, seen = _discover_image_files(tmp_path, set(), mock_extract_prob, mock_extract_id)

        assert len(files) == 2
        assert all(f.suffix == ".png" for f in files)

    def test_deduplication_by_source_id(self, tmp_path):
        """Should deduplicate files by source ID."""
        # Create PNG files
        (tmp_path / "iter001_ZTF123_row1.png").touch()
        (tmp_path / "iter002_ZTF123_row2.png").touch()  # Same source ID
        (tmp_path / "iter001_ZTF456_row3.png").touch()

        def mock_extract_prob(p):
            return None

        def mock_extract_id(p):
            if "ZTF123" in p.name:
                return "ZTF123"
            if "ZTF456" in p.name:
                return "ZTF456"
            return None

        files, seen = _discover_image_files(tmp_path, set(), mock_extract_prob, mock_extract_id)

        # Should only have 2 files (ZTF123 deduplicated)
        assert len(files) == 2
        assert "ZTF123" in seen
        assert "ZTF456" in seen

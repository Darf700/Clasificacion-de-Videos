"""Tests for modules.date_extractor - DateExtractor."""

import os
import tempfile
import time

import pytest

from modules.date_extractor import DateExtractor, FILENAME_PATTERNS


class TestDateExtractor:
    @pytest.fixture
    def extractor(self):
        return DateExtractor()

    # --- Metadata date strategy ---

    def test_metadata_date_iso(self, extractor):
        result = extractor.extract("/fake/path.mp4", metadata_date="2023-04-15T10:30:00Z")
        assert result["year"] == 2023
        assert result["month"] == 4
        assert result["month_name"] == "Abril"
        assert result["date_source"] == "exif"
        assert result["creation_date"] == "2023-04-15T10:30:00Z"

    def test_metadata_date_simple(self, extractor):
        result = extractor.extract("/fake/path.mp4", metadata_date="2021-12-25")
        assert result["year"] == 2021
        assert result["month"] == 12
        assert result["date_source"] == "exif"

    def test_metadata_date_none(self, extractor):
        result = extractor._try_metadata_date(None)
        assert result is None

    def test_metadata_date_invalid(self, extractor):
        result = extractor._try_metadata_date("not a date")
        assert result is None

    # --- Filename date strategy ---

    def test_filename_vid_yyyymmdd(self, extractor):
        result = extractor._try_filename_date("VID_20230415_123456.mp4")
        assert result == (2023, 4)

    def test_filename_img_yyyymmdd(self, extractor):
        result = extractor._try_filename_date("IMG-20210820-WA0001.mp4")
        assert result == (2021, 8)

    def test_filename_yyyymmdd_hhmmss(self, extractor):
        result = extractor._try_filename_date("20220115_093000.mp4")
        assert result == (2022, 1)

    def test_filename_yyyy_mm_dd(self, extractor):
        result = extractor._try_filename_date("video_2023-06-10_edit.mp4")
        assert result == (2023, 6)

    def test_filename_ddmmyyyy(self, extractor):
        result = extractor._try_filename_date("video_15042023.mp4")
        assert result == (2023, 4)

    def test_filename_screen_recording(self, extractor):
        result = extractor._try_filename_date("Screen Recording 2024-01-20 at 14.30.mp4")
        assert result == (2024, 1)

    def test_filename_unix_timestamp(self, extractor):
        result = extractor._try_filename_date("RPReplay_Final1681574400.mp4")
        assert result is not None
        assert result[0] == 2023  # April 2023

    def test_filename_no_date(self, extractor):
        result = extractor._try_filename_date("funny_cat_video.mp4")
        assert result is None

    def test_filename_invalid_month(self, extractor):
        # month=99 should fail validation
        result = extractor._try_filename_date("VID_20239915_120000.mp4")
        assert result is None

    # --- File modified date strategy ---

    def test_file_modified_date(self, extractor, tmp_dir):
        path = os.path.join(tmp_dir, "test.mp4")
        with open(path, "wb") as f:
            f.write(b"\x00" * 100)
        result = extractor._try_file_modified_date(path)
        assert result is not None
        assert 2000 <= result[0] <= 2030

    def test_file_modified_nonexistent(self, extractor):
        result = extractor._try_file_modified_date("/nonexistent/path.mp4")
        assert result is None

    # --- Full extract with priority ---

    def test_metadata_takes_priority_over_filename(self, extractor):
        # Even though filename says 2022, metadata says 2023
        result = extractor.extract(
            "/fake/VID_20220101_120000.mp4",
            metadata_date="2023-07-01T00:00:00Z",
        )
        assert result["year"] == 2023
        assert result["date_source"] == "exif"

    def test_fallback_to_unknown(self, extractor):
        result = extractor.extract(
            "/nonexistent/random_file.mp4",
            metadata_date=None,
        )
        assert result["date_source"] == "unknown"
        assert result["year"] is None
        assert result["month"] is None
        assert result["month_name"] is None

    # --- Validation ---

    def test_is_valid_date(self, extractor):
        assert extractor._is_valid_date(2023, 6)
        assert not extractor._is_valid_date(1999, 6)  # Too old
        assert not extractor._is_valid_date(2031, 6)  # Too far
        assert not extractor._is_valid_date(2023, 0)  # Invalid month
        assert not extractor._is_valid_date(2023, 13) # Invalid month

    # --- Month names ---

    def test_month_names_spanish(self, extractor):
        result = extractor._build_result(2023, 1, "exif", None)
        assert result["month_name"] == "Enero"
        result = extractor._build_result(2023, 12, "exif", None)
        assert result["month_name"] == "Diciembre"

    def test_custom_month_names(self):
        ext = DateExtractor(month_names={1: "January", 2: "February"})
        result = ext._build_result(2023, 1, "exif", None)
        assert result["month_name"] == "January"

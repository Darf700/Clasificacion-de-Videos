"""Tests for date extraction, theme classification, and file organization."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.date_extractor import DateExtractor
from modules.theme_classifier import ThemeClassifier
from modules.file_organizer import FileOrganizer


class TestDateExtractor:
    """Tests for DateExtractor."""

    @pytest.fixture
    def extractor(self):
        return DateExtractor()

    def test_filename_date_vid_pattern(self, extractor):
        result = extractor._parse_filename_date("VID_20230415_123456.mp4")
        assert result is not None
        assert result.year == 2023
        assert result.month == 4
        assert result.day == 15

    def test_filename_date_img_wa_pattern(self, extractor):
        result = extractor._parse_filename_date("IMG-20231225-WA0001.mp4")
        assert result is not None
        assert result.year == 2023
        assert result.month == 12

    def test_filename_date_iso_pattern(self, extractor):
        result = extractor._parse_filename_date("video_2024-01-15_edit.mp4")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1

    def test_filename_no_date(self, extractor):
        result = extractor._parse_filename_date("funny_cat_video.mp4")
        assert result is None

    def test_exif_date_parsing(self, extractor):
        result = extractor._parse_exif_date("2023-04-15T10:30:00.000000Z")
        assert result is not None
        assert result.year == 2023
        assert result.month == 4

    def test_extract_with_exif(self, extractor, tmp_path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")
        result = extractor.extract(video, exif_date="2023-06-15T12:00:00Z")
        assert result["date_source"] == "exif"
        assert result["year"] == 2023
        assert result["month"] == 6

    def test_extract_fallback_to_filename(self, extractor, tmp_path):
        video = tmp_path / "VID_20240101_120000.mp4"
        video.write_bytes(b"fake")
        result = extractor.extract(video, exif_date=None)
        assert result["date_source"] == "filename"
        assert result["year"] == 2024

    def test_extract_fallback_to_file_modified(self, extractor, tmp_path):
        video = tmp_path / "random_name.mp4"
        video.write_bytes(b"fake")
        result = extractor.extract(video)
        assert result["date_source"] == "file_modified"
        assert result["year"] is not None


class TestThemeClassifier:
    """Tests for ThemeClassifier."""

    @pytest.fixture
    def classifier(self):
        return ThemeClassifier(confidence_threshold=0.6, fallback_theme="Otros")

    def test_high_confidence(self, classifier):
        clip_result = {
            "primary_theme": "Comedia",
            "theme_confidence": 0.85,
            "all_scores": {"Comedia": 0.85, "Vlogs": 0.1, "Otros": 0.05},
        }
        result = classifier.classify(clip_result)
        assert result["theme"] == "Comedia"
        assert result["needs_review"] is False

    def test_low_confidence_fallback(self, classifier):
        clip_result = {
            "primary_theme": "Comedia",
            "theme_confidence": 0.3,
            "all_scores": {"Comedia": 0.3, "Vlogs": 0.25, "Otros": 0.2},
        }
        result = classifier.classify(clip_result)
        assert result["theme"] == "Otros"
        assert result["needs_review"] is True

    def test_empty_result(self, classifier):
        result = classifier.classify({})
        assert result["theme"] == "Otros"
        assert result["needs_review"] is True


class TestFileOrganizer:
    """Tests for FileOrganizer."""

    @pytest.fixture
    def organizer(self, tmp_path):
        return FileOrganizer(output_dir=str(tmp_path / "output"))

    def test_organize_with_date(self, organizer, tmp_path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video")
        result = organizer.organize(video, year=2023, month=4, theme="Comedia")
        assert result.exists()
        assert "2023" in str(result)
        assert "Abril" in str(result)
        assert "Comedia" in str(result)

    def test_organize_no_date(self, organizer, tmp_path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video")
        result = organizer.organize(video, year=None, month=None, theme="Otros")
        assert result.exists()
        assert "Sin_Fecha" in str(result)

    def test_conflict_resolution(self, organizer, tmp_path):
        video1 = tmp_path / "test.mp4"
        video1.write_bytes(b"video 1")
        result1 = organizer.organize(video1, year=2023, month=1, theme="Comedia")

        video2 = tmp_path / "test.mp4"
        video2.write_bytes(b"video 2")
        result2 = organizer.organize(video2, year=2023, month=1, theme="Comedia")

        assert result1 != result2
        assert result1.exists()
        assert result2.exists()

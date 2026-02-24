"""Tests for modules.metadata_extractor - MetadataExtractor (pure logic only)."""

import pytest

from modules.metadata_extractor import MetadataExtractor


class TestMetadataExtractor:
    @pytest.fixture
    def extractor(self):
        return MetadataExtractor()

    # --- FPS parsing ---

    def test_parse_fps_fraction(self, extractor):
        assert extractor._parse_fps("30000/1001") == pytest.approx(29.97, rel=0.01)

    def test_parse_fps_integer_string(self, extractor):
        assert extractor._parse_fps("30") == 30.0

    def test_parse_fps_zero_denominator(self, extractor):
        assert extractor._parse_fps("30/0") == 0.0

    def test_parse_fps_invalid(self, extractor):
        assert extractor._parse_fps("not_a_number") == 0.0

    def test_parse_fps_empty(self, extractor):
        assert extractor._parse_fps("") == 0.0

    # --- Aspect ratio ---

    def test_aspect_ratio_16_9(self, extractor):
        assert extractor._compute_aspect_ratio(1920, 1080) == "16:9"

    def test_aspect_ratio_9_16(self, extractor):
        assert extractor._compute_aspect_ratio(1080, 1920) == "9:16"

    def test_aspect_ratio_1_1(self, extractor):
        assert extractor._compute_aspect_ratio(1080, 1080) == "1:1"

    def test_aspect_ratio_4_3(self, extractor):
        assert extractor._compute_aspect_ratio(640, 480) == "4:3"

    def test_aspect_ratio_zero(self, extractor):
        assert extractor._compute_aspect_ratio(0, 1080) == "unknown"
        assert extractor._compute_aspect_ratio(1920, 0) == "unknown"

    def test_aspect_ratio_approximate_16_9(self, extractor):
        # 1280x720 is exactly 16:9
        assert extractor._compute_aspect_ratio(1280, 720) == "16:9"

    # --- Orientation ---

    def test_orientation_horizontal(self, extractor):
        assert extractor._classify_orientation(1920, 1080) == "horizontal"

    def test_orientation_vertical(self, extractor):
        assert extractor._classify_orientation(1080, 1920) == "vertical"

    def test_orientation_square(self, extractor):
        assert extractor._classify_orientation(1080, 1080) == "square"

    def test_orientation_zero(self, extractor):
        assert extractor._classify_orientation(0, 0) == "horizontal"

    # --- Format type ---

    def test_format_type_reel(self, extractor):
        assert extractor._classify_format_type(30) == "reel"
        assert extractor._classify_format_type(90) == "reel"

    def test_format_type_short(self, extractor):
        assert extractor._classify_format_type(91) == "short"
        assert extractor._classify_format_type(600) == "short"

    def test_format_type_long(self, extractor):
        assert extractor._classify_format_type(601) == "long"

    def test_format_type_zero(self, extractor):
        assert extractor._classify_format_type(0) == "reel"

    # --- Creation date extraction ---

    def test_extract_creation_date_from_tags(self, extractor):
        tags = {"creation_time": "2023-04-15T10:30:00.000000Z"}
        assert extractor._extract_creation_date(tags) == "2023-04-15T10:30:00.000000Z"

    def test_extract_creation_date_apple(self, extractor):
        tags = {"com.apple.quicktime.creationdate": "2023-06-01T12:00:00+0200"}
        assert "2023-06-01" in extractor._extract_creation_date(tags)

    def test_extract_creation_date_case_insensitive(self, extractor):
        tags = {"Creation_Time": "2023-01-01"}
        result = extractor._extract_creation_date(tags)
        assert result == "2023-01-01"

    def test_extract_creation_date_not_found(self, extractor):
        tags = {"unrelated_tag": "value"}
        assert extractor._extract_creation_date(tags) is None

    def test_extract_creation_date_empty(self, extractor):
        assert extractor._extract_creation_date({}) is None

    # --- Parse probe data ---

    def test_parse_probe_data_complete(self, extractor, tmp_dir):
        """Test parsing a realistic ffprobe output."""
        import os
        # Create a dummy file for file_size_bytes
        dummy = os.path.join(tmp_dir, "test.mp4")
        with open(dummy, "wb") as f:
            f.write(b"\x00" * 5000)

        probe_data = {
            "format": {
                "duration": "45.5",
                "bit_rate": "1500000",
                "tags": {"creation_time": "2023-06-15T10:00:00Z"},
            },
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "30/1",
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                },
            ],
        }
        result = extractor._parse_probe_data(probe_data, dummy)
        assert result["duration_seconds"] == 45.5
        assert result["width"] == 1920
        assert result["height"] == 1080
        assert result["codec_video"] == "h264"
        assert result["codec_audio"] == "aac"
        assert result["has_audio"] is True
        assert result["bitrate"] == 1500000
        assert result["fps"] == 30.0
        assert result["aspect_ratio"] == "16:9"
        assert result["orientation"] == "horizontal"
        assert result["format_type"] == "reel"
        assert result["creation_date"] == "2023-06-15T10:00:00Z"
        assert result["file_size_bytes"] == 5000

    def test_parse_probe_data_no_audio(self, extractor, tmp_dir):
        import os
        dummy = os.path.join(tmp_dir, "test.mp4")
        with open(dummy, "wb") as f:
            f.write(b"\x00")

        probe_data = {
            "format": {"duration": "10.0"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "width": 1080, "height": 1920, "r_frame_rate": "30/1"},
            ],
        }
        result = extractor._parse_probe_data(probe_data, dummy)
        assert result["has_audio"] is False
        assert result["codec_audio"] is None
        assert result["orientation"] == "vertical"

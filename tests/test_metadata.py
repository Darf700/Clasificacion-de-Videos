"""Tests for metadata extraction module."""

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.metadata_extractor import MetadataExtractor


@pytest.fixture
def extractor():
    return MetadataExtractor()


@pytest.fixture
def mock_ffprobe_output():
    return json.dumps({
        "format": {
            "duration": "30.5",
            "bit_rate": "2500000",
            "tags": {
                "creation_time": "2023-04-15T10:30:00.000000Z"
            },
        },
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1080,
                "height": 1920,
                "r_frame_rate": "30/1",
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
            },
        ],
    })


def test_parse_probe_data(extractor, mock_ffprobe_output, tmp_path):
    """Test parsing of FFprobe JSON output."""
    video_file = tmp_path / "test.mp4"
    video_file.write_bytes(b"fake video content")

    data = json.loads(mock_ffprobe_output)
    result = extractor._parse_probe_data(data, video_file)

    assert result["filename"] == "test.mp4"
    assert result["duration_seconds"] == 30.5
    assert result["width"] == 1080
    assert result["height"] == 1920
    assert result["codec_video"] == "h264"
    assert result["codec_audio"] == "aac"
    assert result["has_audio"] is True
    assert result["fps"] == 30.0
    assert result["aspect_ratio"] == "9:16"  # Vertical video


def test_parse_fps(extractor):
    """Test FPS parsing from fraction string."""
    assert extractor._parse_fps("30/1") == 30.0
    assert extractor._parse_fps("24000/1001") == pytest.approx(23.98, abs=0.01)
    assert extractor._parse_fps("0/0") is None


def test_find_stream(extractor, mock_ffprobe_output):
    """Test stream finding by type."""
    data = json.loads(mock_ffprobe_output)
    video_stream = extractor._find_stream(data, "video")
    assert video_stream is not None
    assert video_stream["codec_name"] == "h264"

    subtitle_stream = extractor._find_stream(data, "subtitle")
    assert subtitle_stream is None

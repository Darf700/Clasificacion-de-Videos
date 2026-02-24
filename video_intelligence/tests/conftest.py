"""Shared fixtures for Video Intelligence tests."""

import os
import tempfile

import pytest


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def schema_path():
    """Path to the official database schema."""
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "schemas",
        "database.sql",
    )


@pytest.fixture
def sample_video_data():
    """Minimal valid video record for database insertion."""
    return {
        "filename": "test_video.mp4",
        "original_path": "/videos/test_video.mp4",
        "file_hash": "abc123def456",
        "file_size_bytes": 1024000,
        "duration_seconds": 45.0,
        "width": 1920,
        "height": 1080,
        "codec_video": "h264",
        "has_audio": True,
        "format_type": "reel",
        "orientation": "horizontal",
        "date_source": "unknown",
    }

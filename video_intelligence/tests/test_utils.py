"""Tests for utils modules - hash_utils, video_utils, gpu_utils, logging_utils."""

import os
import tempfile

import pytest

from utils.hash_utils import calculate_md5
from utils.video_utils import discover_videos, format_duration, format_file_size, get_file_size
from utils.logging_utils import setup_logging, get_logger


# =========================================================================
# hash_utils
# =========================================================================

class TestHashUtils:
    def test_calculate_md5_known_value(self, tmp_dir):
        path = os.path.join(tmp_dir, "test.bin")
        with open(path, "wb") as f:
            f.write(b"hello world")
        result = calculate_md5(path)
        assert result == "5eb63bbbe01eeed093cb22bb8f5acdc3"  # known MD5

    def test_calculate_md5_empty_file(self, tmp_dir):
        path = os.path.join(tmp_dir, "empty.bin")
        with open(path, "wb") as f:
            pass
        result = calculate_md5(path)
        assert result == "d41d8cd98f00b204e9800998ecf8427e"  # MD5 of empty string

    def test_calculate_md5_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            calculate_md5("/nonexistent/file.bin")

    def test_calculate_md5_deterministic(self, tmp_dir):
        path = os.path.join(tmp_dir, "test.bin")
        with open(path, "wb") as f:
            f.write(b"test data")
        h1 = calculate_md5(path)
        h2 = calculate_md5(path)
        assert h1 == h2


# =========================================================================
# video_utils
# =========================================================================

class TestDiscoverVideos:
    def test_discover_mp4_files(self, tmp_dir):
        for name in ["a.mp4", "b.mp4", "c.txt", "d.mov"]:
            with open(os.path.join(tmp_dir, name), "w") as f:
                f.write("")
        videos = discover_videos(tmp_dir)
        assert len(videos) == 3  # .mp4 x2 + .mov x1
        assert all(os.path.isabs(v) for v in videos)

    def test_discover_custom_extensions(self, tmp_dir):
        for name in ["a.mp4", "b.avi", "c.webm"]:
            with open(os.path.join(tmp_dir, name), "w") as f:
                f.write("")
        videos = discover_videos(tmp_dir, extensions=[".avi"])
        assert len(videos) == 1
        assert videos[0].endswith(".avi")

    def test_discover_nonexistent_dir(self):
        result = discover_videos("/nonexistent/directory")
        assert result == []

    def test_discover_empty_dir(self, tmp_dir):
        result = discover_videos(tmp_dir)
        assert result == []

    def test_discover_sorted(self, tmp_dir):
        for name in ["z.mp4", "a.mp4", "m.mp4"]:
            with open(os.path.join(tmp_dir, name), "w") as f:
                f.write("")
        videos = discover_videos(tmp_dir)
        basenames = [os.path.basename(v) for v in videos]
        assert basenames == sorted(basenames)

    def test_discover_case_insensitive_extensions(self, tmp_dir):
        for name in ["a.MP4", "b.Mp4", "c.mp4"]:
            with open(os.path.join(tmp_dir, name), "w") as f:
                f.write("")
        videos = discover_videos(tmp_dir)
        assert len(videos) == 3

    def test_discover_skips_directories(self, tmp_dir):
        os.makedirs(os.path.join(tmp_dir, "subdir.mp4"))
        videos = discover_videos(tmp_dir)
        assert len(videos) == 0


class TestFormatDuration:
    def test_seconds_only(self):
        assert format_duration(45) == "45s"

    def test_minutes_and_seconds(self):
        assert format_duration(150) == "2m 30s"

    def test_hours_minutes_seconds(self):
        assert format_duration(5025) == "1h 23m 45s"

    def test_zero(self):
        assert format_duration(0) == "0s"

    def test_negative(self):
        assert format_duration(-5) == "0s"

    def test_exact_hour(self):
        assert format_duration(3600) == "1h 0s"


class TestFormatFileSize:
    def test_bytes(self):
        assert format_file_size(500) == "500.0 B"

    def test_kilobytes(self):
        assert format_file_size(2048) == "2.0 KB"

    def test_megabytes(self):
        assert format_file_size(5 * 1024 * 1024) == "5.0 MB"

    def test_gigabytes(self):
        assert format_file_size(int(1.5 * 1024**3)) == "1.5 GB"

    def test_zero(self):
        assert format_file_size(0) == "0.0 B"

    def test_negative(self):
        assert format_file_size(-1) == "0 B"


class TestGetFileSize:
    def test_existing_file(self, tmp_dir):
        path = os.path.join(tmp_dir, "test.bin")
        with open(path, "wb") as f:
            f.write(b"\x00" * 1234)
        assert get_file_size(path) == 1234

    def test_nonexistent_file(self):
        assert get_file_size("/nonexistent/file") == 0


# =========================================================================
# logging_utils
# =========================================================================

class TestLogging:
    def test_setup_logging_console_only(self):
        logger = setup_logging(log_dir=None, level="DEBUG", log_to_file=False)
        assert logger.name == "video_intelligence"

    def test_setup_logging_with_file(self, tmp_dir):
        logger = setup_logging(log_dir=tmp_dir, level="INFO", log_to_file=True)
        logger.info("test message")
        # Check log file was created
        log_files = [f for f in os.listdir(tmp_dir) if f.endswith(".log")]
        assert len(log_files) >= 1
        # Close file handlers so Windows can clean up the temp dir
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    def test_get_logger(self):
        logger = get_logger("test_module")
        assert logger.name == "video_intelligence.test_module"

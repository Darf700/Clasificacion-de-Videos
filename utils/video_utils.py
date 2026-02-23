"""Video helper functions."""

from pathlib import Path
from typing import Optional

from utils.logging_utils import get_logger

logger = get_logger("video_utils")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}


def is_video_file(path: str | Path) -> bool:
    """Check if a file is a video based on extension.

    Args:
        path: Path to check.

    Returns:
        True if the file has a video extension.
    """
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def scan_video_files(directory: str | Path) -> list[Path]:
    """Scan a directory for video files (non-recursive).

    Args:
        directory: Directory to scan.

    Returns:
        Sorted list of video file paths.
    """
    directory = Path(directory)
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []

    videos = sorted(
        [f for f in directory.iterdir() if f.is_file() and is_video_file(f)]
    )
    logger.info(f"Found {len(videos)} video files in {directory}")
    return videos


def classify_format(width: Optional[int], height: Optional[int], duration: Optional[float]) -> dict:
    """Classify video format based on dimensions and duration.

    Args:
        width: Video width in pixels.
        height: Video height in pixels.
        duration: Video duration in seconds.

    Returns:
        Dictionary with format_type and orientation.
    """
    result = {"format_type": "short", "orientation": "horizontal"}

    if width and height:
        if width > height:
            result["orientation"] = "horizontal"
        elif height > width:
            result["orientation"] = "vertical"
        else:
            result["orientation"] = "square"

    if duration:
        if duration <= 60:
            result["format_type"] = "reel"
        elif duration <= 180:
            result["format_type"] = "short"
        else:
            result["format_type"] = "long"

    return result

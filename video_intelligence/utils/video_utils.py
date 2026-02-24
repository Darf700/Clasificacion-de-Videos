"""Video helper functions."""

import os
from pathlib import Path
from typing import List, Optional

from utils.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def discover_videos(
    input_dir: str,
    extensions: Optional[List[str]] = None,
) -> List[str]:
    """Discover all video files in the input directory.

    Args:
        input_dir: Path to the directory to scan.
        extensions: List of file extensions to include (e.g., ['.mp4', '.mov']).
            Defaults to common video formats.

    Returns:
        Sorted list of absolute paths to video files found.
    """
    valid_ext = set(extensions) if extensions else DEFAULT_EXTENSIONS
    valid_ext = {e.lower() for e in valid_ext}

    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        return []

    videos = []
    for entry in input_path.iterdir():
        if entry.is_file() and entry.suffix.lower() in valid_ext:
            videos.append(str(entry.resolve()))

    videos.sort()
    logger.info("Discovered %d video(s) in %s", len(videos), input_dir)
    return videos


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string like '1h 23m 45s' or '2m 30s'.
    """
    if seconds < 0:
        return "0s"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def format_file_size(size_bytes: int) -> str:
    """Format file size in bytes to human-readable string.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Formatted string like '1.5 GB' or '234 MB'.
    """
    if size_bytes < 0:
        return "0 B"

    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0

    return f"{size_bytes:.1f} PB"


def get_file_size(file_path: str) -> int:
    """Get file size in bytes.

    Args:
        file_path: Path to the file.

    Returns:
        File size in bytes, or 0 if file doesn't exist.
    """
    try:
        return os.path.getsize(file_path)
    except OSError:
        return 0

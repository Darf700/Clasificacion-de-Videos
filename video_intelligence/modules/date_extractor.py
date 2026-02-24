"""Date extraction from video metadata, filenames, and file system.

Returns fields aligned with official schema:
- year, month, month_name
- date_source ('exif', 'filename', 'file_modified', 'unknown')
- creation_date
"""

import os
import re
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from dateutil import parser as dateutil_parser

from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Filename patterns for date extraction (ordered by specificity)
FILENAME_PATTERNS = [
    # VID_20230415_123456.mp4
    (r"VID[_-](\d{4})(\d{2})(\d{2})", "VID_YYYYMMDD"),
    # IMG-20230415-WA0001.mp4
    (r"IMG[_-](\d{4})(\d{2})(\d{2})", "IMG_YYYYMMDD"),
    # 20230415_123456.mp4
    (r"(\d{4})(\d{2})(\d{2})[_-]\d{6}", "YYYYMMDD_HHMMSS"),
    # 2023-04-15 or 2023_04_15
    (r"(\d{4})[_-](\d{2})[_-](\d{2})", "YYYY-MM-DD"),
    # video_15042023.mp4 (DD/MM/YYYY)
    (r"(\d{2})(\d{2})(\d{4})", "DDMMYYYY"),
    # RPReplay_Final1681574400.mp4 (Unix timestamp)
    (r"RPReplay[_\w]*?(\d{10})", "unix_timestamp"),
    # Screen Recording 2023-04-15
    (r"Screen[_ ]Recording[_ ](\d{4})[_-](\d{2})[_-](\d{2})", "screen_recording"),
    # Snapchat-1234567890.mp4
    (r"Snapchat[_-](\d{10,13})", "snapchat_timestamp"),
]


class DateExtractor:
    """Extracts dates from video files using multiple strategies.

    Priority order:
    1. EXIF/metadata creation date (most reliable)
    2. Filename patterns
    3. File modification date (fallback)

    Args:
        month_names: Dictionary mapping month numbers to Spanish names.
    """

    def __init__(self, month_names: Optional[Dict[int, str]] = None) -> None:
        self.month_names = month_names or {
            1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
            5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
            9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
        }

    def extract(
        self,
        video_path: str,
        metadata_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract date information from a video.

        Args:
            video_path: Path to the video file.
            metadata_date: Creation date from FFprobe metadata (ISO format).

        Returns:
            Dictionary matching schema fields:
                - year: int or None
                - month: int (1-12) or None
                - month_name: str (Spanish) or None
                - date_source: str ('exif', 'filename', 'file_modified', 'unknown')
                - creation_date: str (ISO format) or None
        """
        # Strategy 1: EXIF/metadata date
        result = self._try_metadata_date(metadata_date)
        if result:
            logger.debug("Date from metadata: %d/%d", result[0], result[1])
            return self._build_result(result[0], result[1], "exif", metadata_date)

        # Strategy 2: Filename patterns
        filename = os.path.basename(video_path)
        result = self._try_filename_date(filename)
        if result:
            logger.debug("Date from filename '%s': %d/%d", filename, result[0], result[1])
            return self._build_result(result[0], result[1], "filename", None)

        # Strategy 3: File modification date
        result = self._try_file_modified_date(video_path)
        if result:
            logger.debug("Date from file modification: %d/%d", result[0], result[1])
            return self._build_result(result[0], result[1], "file_modified", None)

        # No date found
        logger.warning("No date found for %s", filename)
        return self._build_result(None, None, "unknown", None)

    def _try_metadata_date(self, date_str: Optional[str]) -> Optional[Tuple[int, int]]:
        """Try to parse a date from metadata string.

        Args:
            date_str: ISO format date string from FFprobe.

        Returns:
            (year, month) tuple or None.
        """
        if not date_str:
            return None

        try:
            dt = dateutil_parser.parse(date_str)
            if self._is_valid_date(dt.year, dt.month):
                return (dt.year, dt.month)
        except (ValueError, OverflowError):
            logger.debug("Failed to parse metadata date: %s", date_str)

        return None

    def _try_filename_date(self, filename: str) -> Optional[Tuple[int, int]]:
        """Try to extract a date from the filename using known patterns.

        Args:
            filename: Video filename (without directory).

        Returns:
            (year, month) tuple or None.
        """
        for pattern, pattern_name in FILENAME_PATTERNS:
            match = re.search(pattern, filename, re.IGNORECASE)
            if not match:
                continue

            try:
                if pattern_name == "unix_timestamp":
                    timestamp = int(match.group(1))
                    dt = datetime.fromtimestamp(timestamp)
                    if self._is_valid_date(dt.year, dt.month):
                        return (dt.year, dt.month)

                elif pattern_name == "snapchat_timestamp":
                    timestamp = int(match.group(1))
                    # Handle millisecond timestamps
                    if timestamp > 9999999999:
                        timestamp = timestamp // 1000
                    dt = datetime.fromtimestamp(timestamp)
                    if self._is_valid_date(dt.year, dt.month):
                        return (dt.year, dt.month)

                elif pattern_name == "DDMMYYYY":
                    day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
                    if self._is_valid_date(year, month):
                        return (year, month)

                else:
                    year, month = int(match.group(1)), int(match.group(2))
                    if self._is_valid_date(year, month):
                        return (year, month)

            except (ValueError, OSError, OverflowError):
                continue

        return None

    def _try_file_modified_date(self, video_path: str) -> Optional[Tuple[int, int]]:
        """Use file modification date as last resort.

        Args:
            video_path: Path to the video file.

        Returns:
            (year, month) tuple or None.
        """
        try:
            mtime = os.path.getmtime(video_path)
            dt = datetime.fromtimestamp(mtime)
            if self._is_valid_date(dt.year, dt.month):
                return (dt.year, dt.month)
        except (OSError, ValueError, OverflowError):
            pass

        return None

    def _is_valid_date(self, year: int, month: int) -> bool:
        """Check if year and month are within reasonable range.

        Args:
            year: Year value.
            month: Month value (1-12).

        Returns:
            True if the date is valid and reasonable (2000-2030).
        """
        return 2000 <= year <= 2030 and 1 <= month <= 12

    def _build_result(
        self,
        year: Optional[int],
        month: Optional[int],
        source: str,
        creation_date: Optional[str],
    ) -> Dict[str, Any]:
        """Build the result dictionary matching schema fields.

        Args:
            year: Extracted year or None.
            month: Extracted month or None.
            source: Date source identifier.
            creation_date: Raw creation date string.

        Returns:
            Structured result dictionary.
        """
        month_name = self.month_names.get(month) if month else None

        return {
            "year": year,
            "month": month,
            "month_name": month_name,
            "date_source": source,
            "creation_date": creation_date,
        }

"""Date extraction from video files using multiple sources."""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from utils.logging_utils import get_logger

logger = get_logger("date_extractor")

# Common date patterns in filenames
FILENAME_PATTERNS = [
    # VID_20230415_123456
    (r"VID[_-](\d{4})(\d{2})(\d{2})", "%Y%m%d"),
    # IMG-20230415-WA0001
    (r"IMG[_-](\d{4})(\d{2})(\d{2})", "%Y%m%d"),
    # 20230415_123456
    (r"(\d{4})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})", "%Y%m%d_%H%M%S"),
    # 2023-04-15
    (r"(\d{4})-(\d{2})-(\d{2})", "%Y-%m-%d"),
    # 20230415
    (r"(\d{4})(\d{2})(\d{2})", "%Y%m%d"),
    # Screen Recording 2023-04-15
    (r"[Rr]ecording[_\s-]+(\d{4})[_-](\d{2})[_-](\d{2})", "%Y-%m-%d"),
]


class DateExtractor:
    """Extracts creation dates from video files using multiple strategies."""

    def __init__(self, date_sources: Optional[list[str]] = None):
        """Initialize date extractor.

        Args:
            date_sources: Priority order for date extraction sources.
        """
        self.date_sources = date_sources or ["exif", "filename", "file_modified"]

    def extract(
        self,
        video_path: str | Path,
        exif_date: Optional[str] = None,
    ) -> dict:
        """Extract the best available date from a video file.

        Args:
            video_path: Path to the video file.
            exif_date: Pre-extracted EXIF creation date string.

        Returns:
            Dictionary with creation_date, date_source, year, month, month_name.
        """
        video_path = Path(video_path)
        result = {
            "creation_date": None,
            "date_source": "unknown",
            "year": None,
            "month": None,
            "month_name": None,
        }

        for source in self.date_sources:
            extracted = None
            if source == "exif" and exif_date:
                extracted = self._parse_exif_date(exif_date)
                if extracted:
                    result["date_source"] = "exif"
            elif source == "filename":
                extracted = self._parse_filename_date(video_path.name)
                if extracted:
                    result["date_source"] = "filename"
            elif source == "file_modified":
                extracted = self._get_file_modified_date(video_path)
                if extracted:
                    result["date_source"] = "file_modified"

            if extracted:
                result["creation_date"] = extracted.strftime("%Y-%m-%d %H:%M:%S")
                result["year"] = extracted.year
                result["month"] = extracted.month
                break

        logger.debug(
            f"Date for {video_path.name}: {result['creation_date']} "
            f"(source: {result['date_source']})"
        )
        return result

    def _parse_exif_date(self, date_str: str) -> Optional[datetime]:
        """Parse EXIF date string.

        Args:
            date_str: EXIF date string in various formats.

        Returns:
            Parsed datetime or None.
        """
        formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
            "%Y:%m:%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        logger.debug(f"Could not parse EXIF date: {date_str}")
        return None

    def _parse_filename_date(self, filename: str) -> Optional[datetime]:
        """Extract date from filename using common patterns.

        Args:
            filename: Video filename.

        Returns:
            Parsed datetime or None.
        """
        for pattern, date_fmt in FILENAME_PATTERNS:
            match = re.search(pattern, filename)
            if match:
                try:
                    date_str = "".join(match.groups())
                    # Validate reasonable year range
                    year = int(match.group(1))
                    if 2000 <= year <= 2030:
                        month = int(match.group(2))
                        day = int(match.group(3)) if len(match.groups()) >= 3 else 1
                        if 1 <= month <= 12 and 1 <= day <= 31:
                            return datetime(year, month, day)
                except (ValueError, IndexError):
                    continue
        return None

    def _get_file_modified_date(self, path: Path) -> Optional[datetime]:
        """Get file modification date.

        Args:
            path: Path to the file.

        Returns:
            File modification datetime or None.
        """
        try:
            mtime = os.path.getmtime(path)
            return datetime.fromtimestamp(mtime)
        except OSError:
            return None

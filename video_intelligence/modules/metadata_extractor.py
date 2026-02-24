"""Video metadata extraction using FFprobe.

Returns fields aligned with the official schema:
- codec_video, codec_audio (not video_codec/audio_codec)
- bitrate (not bitrate_kbps)
- aspect_ratio, format_type, orientation
"""

import json
import os
import subprocess
from typing import Any, Dict, Optional

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class MetadataExtractor:
    """Extracts technical metadata from video files using FFprobe.

    Provides information about duration, resolution, codecs, bitrate,
    and derives format classification (reel/short/long, orientation).
    """

    def extract(self, video_path: str) -> Dict[str, Any]:
        """Extract metadata from a video file.

        Args:
            video_path: Absolute path to the video file.

        Returns:
            Dictionary matching schema columns:
                - duration_seconds: float
                - width: int
                - height: int
                - aspect_ratio: str (e.g., '9:16')
                - fps: float
                - bitrate: int (bps)
                - codec_video: str
                - codec_audio: str | None
                - format_type: str ('reel', 'short', 'long')
                - orientation: str ('vertical', 'horizontal', 'square')
                - has_audio: bool
                - file_size_bytes: int
                - creation_date: str | None (ISO format from metadata)

        Raises:
            FileNotFoundError: If the video file doesn't exist.
            RuntimeError: If FFprobe fails to extract metadata.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        probe_data = self._run_ffprobe(video_path)
        metadata = self._parse_probe_data(probe_data, video_path)

        logger.debug(
            "Metadata for %s: %dx%d, %.1fs, %s, %s, %s",
            os.path.basename(video_path),
            metadata.get("width", 0),
            metadata.get("height", 0),
            metadata.get("duration_seconds", 0),
            metadata.get("codec_video", "unknown"),
            metadata.get("format_type", "?"),
            metadata.get("orientation", "?"),
        )

        return metadata

    def _run_ffprobe(self, video_path: str) -> Dict[str, Any]:
        """Run ffprobe and return parsed JSON output.

        Args:
            video_path: Path to the video file.

        Returns:
            Parsed JSON output from ffprobe.

        Raises:
            RuntimeError: If ffprobe command fails.
        """
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"FFprobe failed for {video_path}: {result.stderr.strip()}"
                )

            return json.loads(result.stdout)

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"FFprobe timed out for {video_path}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse FFprobe output for {video_path}: {e}")
        except FileNotFoundError:
            raise RuntimeError(
                "FFprobe not found. Please install FFmpeg: sudo apt install ffmpeg"
            )

    def _parse_probe_data(
        self, probe_data: Dict[str, Any], video_path: str
    ) -> Dict[str, Any]:
        """Parse ffprobe JSON output into structured metadata.

        Args:
            probe_data: Raw ffprobe JSON output.
            video_path: Original video path (for file size).

        Returns:
            Parsed metadata dictionary with schema-aligned field names.
        """
        metadata: Dict[str, Any] = {
            "file_size_bytes": os.path.getsize(video_path),
        }

        # Parse format-level data
        fmt = probe_data.get("format", {})
        metadata["duration_seconds"] = float(fmt.get("duration", 0))

        bitrate = fmt.get("bit_rate")
        metadata["bitrate"] = int(bitrate) if bitrate else 0

        # Extract creation date from format tags
        tags = fmt.get("tags", {})
        metadata["creation_date"] = self._extract_creation_date(tags)

        # Parse streams
        video_stream = None
        audio_stream = None

        for stream in probe_data.get("streams", []):
            codec_type = stream.get("codec_type")
            if codec_type == "video" and video_stream is None:
                video_stream = stream
            elif codec_type == "audio" and audio_stream is None:
                audio_stream = stream

        # Video stream data
        if video_stream:
            width = int(video_stream.get("width", 0))
            height = int(video_stream.get("height", 0))
            metadata["width"] = width
            metadata["height"] = height
            metadata["codec_video"] = video_stream.get("codec_name", "unknown")
            metadata["fps"] = self._parse_fps(video_stream.get("r_frame_rate", "0/1"))

            # Derived fields
            metadata["aspect_ratio"] = self._compute_aspect_ratio(width, height)
            metadata["orientation"] = self._classify_orientation(width, height)

            # Try stream-level creation date if format-level missing
            if not metadata["creation_date"]:
                stream_tags = video_stream.get("tags", {})
                metadata["creation_date"] = self._extract_creation_date(stream_tags)
        else:
            metadata["width"] = 0
            metadata["height"] = 0
            metadata["codec_video"] = "unknown"
            metadata["fps"] = 0.0
            metadata["aspect_ratio"] = "unknown"
            metadata["orientation"] = "horizontal"

        # Audio stream data
        if audio_stream:
            metadata["codec_audio"] = audio_stream.get("codec_name", "unknown")
            metadata["has_audio"] = True
        else:
            metadata["codec_audio"] = None
            metadata["has_audio"] = False

        # Format type classification based on duration
        metadata["format_type"] = self._classify_format_type(
            metadata["duration_seconds"]
        )

        return metadata

    def _parse_fps(self, fps_str: str) -> float:
        """Parse FPS from ffprobe fraction format (e.g., '30000/1001').

        Args:
            fps_str: FPS string in 'num/den' format.

        Returns:
            FPS as a float.
        """
        try:
            if "/" in fps_str:
                num, den = fps_str.split("/")
                den_val = float(den)
                if den_val == 0:
                    return 0.0
                return round(float(num) / den_val, 2)
            return round(float(fps_str), 2)
        except (ValueError, ZeroDivisionError):
            return 0.0

    def _extract_creation_date(self, tags: Dict[str, str]) -> Optional[str]:
        """Extract creation date from metadata tags.

        Args:
            tags: Dictionary of metadata tags.

        Returns:
            ISO format date string, or None if not found.
        """
        date_keys = [
            "creation_time",
            "date",
            "com.apple.quicktime.creationdate",
        ]

        for key in date_keys:
            for tag_key, tag_value in tags.items():
                if tag_key.lower() == key.lower() and tag_value:
                    return tag_value.strip()

        return None

    def _compute_aspect_ratio(self, width: int, height: int) -> str:
        """Compute simplified aspect ratio string.

        Args:
            width: Video width in pixels.
            height: Video height in pixels.

        Returns:
            Aspect ratio string like '16:9', '9:16', '1:1'.
        """
        if width == 0 or height == 0:
            return "unknown"

        from math import gcd

        divisor = gcd(width, height)
        w = width // divisor
        h = height // divisor

        # Simplify common ratios
        common_ratios = {
            (16, 9): "16:9",
            (9, 16): "9:16",
            (4, 3): "4:3",
            (3, 4): "3:4",
            (1, 1): "1:1",
            (21, 9): "21:9",
            (18, 9): "18:9",
        }

        if (w, h) in common_ratios:
            return common_ratios[(w, h)]

        # Approximate to nearest common ratio
        ratio = width / height
        if abs(ratio - 16 / 9) < 0.1:
            return "16:9"
        elif abs(ratio - 9 / 16) < 0.1:
            return "9:16"
        elif abs(ratio - 4 / 3) < 0.1:
            return "4:3"
        elif abs(ratio - 1.0) < 0.1:
            return "1:1"

        return f"{w}:{h}"

    def _classify_orientation(self, width: int, height: int) -> str:
        """Classify video orientation.

        Args:
            width: Video width in pixels.
            height: Video height in pixels.

        Returns:
            'vertical', 'horizontal', or 'square'.
        """
        if width == 0 or height == 0:
            return "horizontal"

        ratio = width / height
        if abs(ratio - 1.0) < 0.05:
            return "square"
        elif ratio < 1.0:
            return "vertical"
        else:
            return "horizontal"

    def _classify_format_type(self, duration: float) -> str:
        """Classify video format type based on duration.

        Args:
            duration: Video duration in seconds.

        Returns:
            'reel' (<= 90s), 'short' (<= 600s), or 'long' (> 600s).
        """
        if duration <= 90:
            return "reel"
        elif duration <= 600:
            return "short"
        else:
            return "long"

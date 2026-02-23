"""Video metadata extraction using FFprobe."""

import json
import subprocess
from pathlib import Path
from typing import Any, Optional

from utils.logging_utils import get_logger

logger = get_logger("metadata")


class MetadataExtractor:
    """Extracts technical metadata from video files using FFprobe."""

    def extract(self, video_path: str | Path) -> dict[str, Any]:
        """Extract metadata from a video file.

        Args:
            video_path: Path to the video file.

        Returns:
            Dictionary with extracted metadata fields.

        Raises:
            RuntimeError: If FFprobe fails.
        """
        video_path = Path(video_path)
        logger.debug(f"Extracting metadata: {video_path.name}")

        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "quiet",
                    "-print_format", "json",
                    "-show_format",
                    "-show_streams",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                raise RuntimeError(f"FFprobe error: {result.stderr}")

            probe_data = json.loads(result.stdout)
            return self._parse_probe_data(probe_data, video_path)

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"FFprobe timed out for {video_path.name}")
        except json.JSONDecodeError:
            raise RuntimeError(f"Failed to parse FFprobe output for {video_path.name}")

    def _parse_probe_data(self, data: dict, video_path: Path) -> dict[str, Any]:
        """Parse FFprobe JSON output into a flat metadata dict.

        Args:
            data: Raw FFprobe JSON output.
            video_path: Path to the video file.

        Returns:
            Parsed metadata dictionary.
        """
        metadata: dict[str, Any] = {
            "filename": video_path.name,
            "original_path": str(video_path),
            "file_size_bytes": video_path.stat().st_size,
        }

        # Parse format info
        fmt = data.get("format", {})
        metadata["duration_seconds"] = float(fmt.get("duration", 0))
        metadata["bitrate"] = int(fmt.get("bit_rate", 0)) if fmt.get("bit_rate") else None

        # Parse video stream
        video_stream = self._find_stream(data, "video")
        if video_stream:
            metadata["width"] = video_stream.get("width")
            metadata["height"] = video_stream.get("height")
            metadata["codec_video"] = video_stream.get("codec_name")
            metadata["fps"] = self._parse_fps(video_stream.get("r_frame_rate", "0/1"))

            w, h = metadata["width"], metadata["height"]
            if w and h:
                from math import gcd
                g = gcd(w, h)
                metadata["aspect_ratio"] = f"{w // g}:{h // g}"

        # Parse audio stream
        audio_stream = self._find_stream(data, "audio")
        metadata["has_audio"] = audio_stream is not None
        if audio_stream:
            metadata["codec_audio"] = audio_stream.get("codec_name")

        # Extract creation date from format tags
        tags = fmt.get("tags", {})
        creation_time = tags.get("creation_time") or tags.get("date")
        if creation_time:
            metadata["exif_creation_date"] = creation_time

        logger.debug(
            f"Metadata: {metadata['filename']} - "
            f"{metadata.get('width')}x{metadata.get('height')}, "
            f"{metadata.get('duration_seconds', 0):.1f}s"
        )
        return metadata

    def _find_stream(self, data: dict, codec_type: str) -> Optional[dict]:
        """Find the first stream of a given type.

        Args:
            data: FFprobe data.
            codec_type: 'video' or 'audio'.

        Returns:
            Stream dictionary or None.
        """
        for stream in data.get("streams", []):
            if stream.get("codec_type") == codec_type:
                return stream
        return None

    def _parse_fps(self, fps_str: str) -> Optional[float]:
        """Parse FPS from a fraction string like '30/1'.

        Args:
            fps_str: FPS fraction string.

        Returns:
            FPS as float or None.
        """
        try:
            parts = fps_str.split("/")
            if len(parts) == 2 and int(parts[1]) != 0:
                return round(int(parts[0]) / int(parts[1]), 2)
            return float(parts[0])
        except (ValueError, ZeroDivisionError):
            return None

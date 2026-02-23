"""Frame extraction from video files."""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from utils.logging_utils import get_logger

logger = get_logger("frame_sampler")


class FrameSampler:
    """Extracts frames from video files using FFmpeg."""

    def __init__(self, frames_per_video: int = 30):
        """Initialize frame sampler.

        Args:
            frames_per_video: Number of frames to extract per video.
        """
        self.frames_per_video = frames_per_video

    def extract_frames(
        self,
        video_path: str | Path,
        duration: Optional[float] = None,
        temp_dir: Optional[str] = None,
    ) -> list[np.ndarray]:
        """Extract uniformly distributed frames from a video.

        Args:
            video_path: Path to the video file.
            duration: Video duration in seconds (avoids re-probing).
            temp_dir: Temporary directory for frame files.

        Returns:
            List of frames as numpy arrays (RGB).
        """
        video_path = Path(video_path)
        logger.debug(f"Extracting {self.frames_per_video} frames from {video_path.name}")

        if duration is None or duration <= 0:
            duration = self._get_duration(video_path)

        if duration <= 0:
            logger.warning(f"Cannot determine duration for {video_path.name}")
            return []

        # Calculate timestamps for uniform sampling
        timestamps = [
            duration * i / self.frames_per_video
            for i in range(self.frames_per_video)
        ]

        frames = []
        with tempfile.TemporaryDirectory(dir=temp_dir) as tmpdir:
            for i, ts in enumerate(timestamps):
                frame = self._extract_single_frame(video_path, ts, tmpdir, i)
                if frame is not None:
                    frames.append(frame)

        logger.debug(f"Extracted {len(frames)} frames from {video_path.name}")
        return frames

    def extract_thumbnail(
        self,
        video_path: str | Path,
        output_path: str | Path,
        duration: Optional[float] = None,
        size: tuple[int, int] = (320, 180),
    ) -> bool:
        """Extract a thumbnail from the middle of the video.

        Args:
            video_path: Path to the video file.
            output_path: Path to save the thumbnail.
            duration: Video duration in seconds.
            size: Thumbnail size (width, height).

        Returns:
            True if thumbnail was created successfully.
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if duration is None or duration <= 0:
            duration = self._get_duration(video_path)

        mid_point = max(0, duration / 2)

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss", str(mid_point),
                    "-i", str(video_path),
                    "-vframes", "1",
                    "-vf", f"scale={size[0]}:{size[1]}:force_original_aspect_ratio=decrease,pad={size[0]}:{size[1]}:(ow-iw)/2:(oh-ih)/2",
                    "-q:v", "5",
                    str(output_path),
                ],
                capture_output=True,
                timeout=15,
            )
            return output_path.exists()
        except (subprocess.TimeoutExpired, Exception) as e:
            logger.warning(f"Thumbnail extraction failed for {video_path.name}: {e}")
            return False

    def _extract_single_frame(
        self, video_path: Path, timestamp: float, tmpdir: str, index: int
    ) -> Optional[np.ndarray]:
        """Extract a single frame at a given timestamp.

        Args:
            video_path: Path to the video file.
            timestamp: Time in seconds.
            tmpdir: Temporary directory.
            index: Frame index for naming.

        Returns:
            Frame as numpy array (RGB) or None on failure.
        """
        output_file = Path(tmpdir) / f"frame_{index:04d}.jpg"
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss", str(timestamp),
                    "-i", str(video_path),
                    "-vframes", "1",
                    "-q:v", "2",
                    str(output_file),
                ],
                capture_output=True,
                timeout=10,
            )
            if output_file.exists():
                img = Image.open(output_file).convert("RGB")
                return np.array(img)
        except Exception as e:
            logger.debug(f"Failed to extract frame at {timestamp:.1f}s: {e}")
        return None

    def _get_duration(self, video_path: Path) -> float:
        """Get video duration using FFprobe.

        Args:
            video_path: Path to the video file.

        Returns:
            Duration in seconds.
        """
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "quiet",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return float(result.stdout.strip())
        except (ValueError, subprocess.TimeoutExpired):
            return 0.0

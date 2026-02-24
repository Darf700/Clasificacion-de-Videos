"""Key frame extraction from video files."""

import os
import tempfile
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class FrameSampler:
    """Extracts evenly distributed frames from video files.

    Samples N frames uniformly across the video duration and optionally
    generates a thumbnail from the middle frame.

    Args:
        num_frames: Number of frames to sample per video.
        thumbnail_dir: Directory to store video thumbnails.
    """

    def __init__(self, num_frames: int = 30, thumbnail_dir: Optional[str] = None) -> None:
        self.num_frames = num_frames
        self.thumbnail_dir = thumbnail_dir
        if self.thumbnail_dir:
            os.makedirs(self.thumbnail_dir, exist_ok=True)

    def sample_frames(
        self,
        video_path: str,
        save_dir: Optional[str] = None,
    ) -> List[np.ndarray]:
        """Sample frames uniformly from a video.

        Args:
            video_path: Path to the video file.
            save_dir: Optional directory to save frames as JPEG files.

        Returns:
            List of frames as numpy arrays (BGR format).

        Raises:
            RuntimeError: If the video cannot be opened.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if total_frames <= 0:
                raise RuntimeError(f"Video has no frames: {video_path}")

            # Calculate frame indices to sample
            num_to_sample = min(self.num_frames, total_frames)
            indices = np.linspace(0, total_frames - 1, num_to_sample, dtype=int)

            frames = []
            frame_metadata = []

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames.append(frame)
                    timestamp = float(idx) / fps if fps > 0 else 0.0
                    frame_metadata.append({
                        "frame_number": int(idx),
                        "timestamp": timestamp,
                    })
        finally:
            cap.release()

        if not frames:
            raise RuntimeError(f"No frames could be read from: {video_path}")

        # Save frames to disk if requested
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            for i, frame in enumerate(frames):
                frame_path = os.path.join(save_dir, f"frame_{i:04d}.jpg")
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        logger.debug(
            "Sampled %d/%d frames from %s",
            len(frames),
            total_frames,
            os.path.basename(video_path),
        )

        return frames

    def sample_frames_with_metadata(
        self, video_path: str
    ) -> List[Tuple[np.ndarray, int, float]]:
        """Sample frames and return them with position metadata.

        Args:
            video_path: Path to the video file.

        Returns:
            List of (frame, frame_number, timestamp_seconds) tuples.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if total_frames <= 0:
            cap.release()
            raise RuntimeError(f"Video has no frames: {video_path}")

        num_to_sample = min(self.num_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, num_to_sample, dtype=int)

        results = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                timestamp = float(idx) / fps if fps > 0 else 0.0
                results.append((frame, int(idx), timestamp))

        cap.release()

        if not results:
            raise RuntimeError(f"No frames could be read from: {video_path}")

        logger.debug(
            "Sampled %d frames with metadata from %s",
            len(results),
            os.path.basename(video_path),
        )
        return results

    def generate_thumbnail(
        self,
        video_path: str,
        video_id: Optional[int] = None,
    ) -> Optional[str]:
        """Generate a thumbnail from the middle frame of the video.

        Args:
            video_path: Path to the video file.
            video_id: Optional database ID for naming the thumbnail.

        Returns:
            Path to the saved thumbnail, or None on failure.
        """
        if not self.thumbnail_dir:
            return None

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            mid_frame = total_frames // 2

            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                return None

            # Resize to thumbnail size (320px wide, maintain aspect ratio)
            height, width = frame.shape[:2]
            thumb_width = 320
            thumb_height = int(height * (thumb_width / width))
            thumbnail = cv2.resize(frame, (thumb_width, thumb_height))

            # Save thumbnail
            name = f"video_{video_id}" if video_id else os.path.splitext(os.path.basename(video_path))[0]
            thumb_path = os.path.join(self.thumbnail_dir, f"{name}.jpg")
            cv2.imwrite(thumb_path, thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 80])

            logger.debug("Thumbnail saved: %s", thumb_path)
            return thumb_path

        except Exception as e:
            logger.warning("Thumbnail generation failed for %s: %s", video_path, e)
            return None

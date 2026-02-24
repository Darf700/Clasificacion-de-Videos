"""OCR text extraction from video frames using EasyOCR.

Returns OCR records aligned with official schema:
- text_content (not text)
- timestamp_seconds (not frame_timestamp)
- bbox as (bbox_x, bbox_y, bbox_width, bbox_height)
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.gpu_utils import get_device
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class OCRProcessor:
    """Extracts text from video frames using EasyOCR.

    Processes sampled frames to detect and extract visible text
    (subtitles, overlays, screen text, etc.).

    Args:
        languages: List of language codes for OCR (e.g., ['es', 'en']).
        confidence_threshold: Minimum confidence to keep a detection.
        device: Compute device. Auto-detected if None.
        max_frames: Maximum number of frames to OCR per video.
    """

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
        max_frames: int = 10,
    ) -> None:
        self.languages = languages or ["es", "en"]
        self.confidence_threshold = confidence_threshold
        self.device = device or get_device()
        self.max_frames = max_frames
        self.reader = None

    def load_model(self) -> None:
        """Load the EasyOCR reader."""
        import easyocr

        gpu = self.device == "cuda"
        self.reader = easyocr.Reader(
            self.languages,
            gpu=gpu,
            verbose=False,
        )
        logger.info("EasyOCR loaded for languages: %s (GPU=%s)", self.languages, gpu)

    def extract_text(
        self,
        frames: List[Tuple[np.ndarray, int, float]],
        video_id: int,
    ) -> List[Dict[str, Any]]:
        """Extract text from a set of video frames.

        Processes a subset of frames (every 5th frame) to avoid
        redundant OCR on similar consecutive frames.

        Args:
            frames: List of (frame, frame_number, timestamp) tuples.
            video_id: Database ID of the video.

        Returns:
            List of OCR result dictionaries matching schema:
                - video_id: int
                - frame_number: int
                - timestamp_seconds: float
                - text_content: str
                - language: str
                - confidence: float
                - bbox_x, bbox_y, bbox_width, bbox_height: int
        """
        if self.reader is None:
            self.load_model()

        # Sample frames evenly, capped by max_frames to reduce redundancy
        if len(frames) <= self.max_frames:
            sampled = frames
        else:
            indices = np.linspace(0, len(frames) - 1, self.max_frames, dtype=int)
            sampled = [frames[i] for i in indices]

        all_texts = []
        seen_texts = set()  # Deduplicate identical text across frames

        for frame, frame_number, timestamp in sampled:
            try:
                results = self.reader.readtext(frame)

                for bbox, text, confidence in results:
                    if confidence < self.confidence_threshold:
                        continue

                    text = text.strip()
                    if not text or len(text) < 2:
                        continue

                    # Deduplicate: skip if we've seen this exact text
                    text_key = text.lower()
                    if text_key in seen_texts:
                        continue
                    seen_texts.add(text_key)

                    # Convert bbox polygon to (x, y, width, height)
                    points = np.array(bbox)
                    x = int(np.min(points[:, 0]))
                    y = int(np.min(points[:, 1]))
                    w = int(np.max(points[:, 0]) - x)
                    h = int(np.max(points[:, 1]) - y)

                    ocr_record = {
                        "video_id": video_id,
                        "frame_number": frame_number,
                        "timestamp_seconds": round(timestamp, 3),
                        "text_content": text,
                        "language": self._guess_language(text),
                        "confidence": round(float(confidence), 4),
                        "bbox_x": x,
                        "bbox_y": y,
                        "bbox_width": w,
                        "bbox_height": h,
                    }
                    all_texts.append(ocr_record)

            except Exception as e:
                logger.warning("OCR failed on frame %d: %s", frame_number, e)
                continue

        logger.debug(
            "OCR extracted %d unique texts from %d frames for video_id=%d",
            len(all_texts),
            len(sampled),
            video_id,
        )
        return all_texts

    def _guess_language(self, text: str) -> str:
        """Simple language guess based on character patterns.

        Args:
            text: Text string to analyze.

        Returns:
            Language code ('es' or 'en').
        """
        spanish_chars = set("áéíóúñ¿¡")
        if any(c in text.lower() for c in spanish_chars):
            return "es"
        return "en"

    def unload_model(self) -> None:
        """Unload EasyOCR to free memory."""
        if self.reader is not None:
            del self.reader
            self.reader = None
            logger.info("EasyOCR reader unloaded")

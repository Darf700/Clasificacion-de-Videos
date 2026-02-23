"""OCR text extraction from video frames using EasyOCR."""

from typing import Optional

import numpy as np

from utils.logging_utils import get_logger

logger = get_logger("ocr")


class OCRProcessor:
    """Extracts text from video frames using EasyOCR."""

    def __init__(
        self,
        languages: list[str] = None,
        gpu: bool = True,
        confidence_threshold: float = 0.5,
        max_frames: int = 10,
    ):
        """Initialize OCR processor.

        Args:
            languages: List of language codes (e.g., ['es', 'en']).
            gpu: Whether to use GPU.
            confidence_threshold: Minimum confidence to keep text.
            max_frames: Maximum frames to process per video.
        """
        self.languages = languages or ["es", "en"]
        self.gpu = gpu
        self.confidence_threshold = confidence_threshold
        self.max_frames = max_frames
        self.reader = None

    def load_model(self) -> None:
        """Load the EasyOCR reader."""
        import easyocr

        logger.info(f"Loading EasyOCR for languages: {self.languages}")
        self.reader = easyocr.Reader(self.languages, gpu=self.gpu)
        logger.info("EasyOCR loaded")

    def extract_text(
        self,
        frames: list[np.ndarray],
        video_duration: float = 0.0,
    ) -> list[dict]:
        """Extract text from video frames.

        Args:
            frames: List of frames as numpy arrays (RGB).
            video_duration: Video duration for timestamp calculation.

        Returns:
            List of OCR result dictionaries.
        """
        if not frames:
            return []

        if self.reader is None:
            self.load_model()

        # Sample frames evenly
        total = len(frames)
        if total > self.max_frames:
            indices = np.linspace(0, total - 1, self.max_frames, dtype=int)
        else:
            indices = range(total)

        all_results = []

        for idx in indices:
            frame = frames[idx]
            timestamp = (video_duration * idx / total) if total > 0 else 0

            try:
                detections = self.reader.readtext(frame)

                for bbox, text, conf in detections:
                    if conf < self.confidence_threshold:
                        continue

                    # Convert bbox points to x, y, w, h
                    points = np.array(bbox)
                    x = int(points[:, 0].min())
                    y = int(points[:, 1].min())
                    w = int(points[:, 0].max() - x)
                    h = int(points[:, 1].max() - y)

                    all_results.append({
                        "frame_number": int(idx),
                        "timestamp_seconds": round(timestamp, 2),
                        "text_content": text.strip(),
                        "confidence": round(float(conf), 4),
                        "bbox_x": x,
                        "bbox_y": y,
                        "bbox_width": w,
                        "bbox_height": h,
                    })

            except Exception as e:
                logger.debug(f"OCR failed on frame {idx}: {e}")

        # Deduplicate similar texts from consecutive frames
        results = self._deduplicate(all_results)
        logger.debug(f"OCR found {len(results)} text regions")
        return results

    def _deduplicate(self, results: list[dict]) -> list[dict]:
        """Remove duplicate text detections across frames.

        Args:
            results: List of OCR result dicts.

        Returns:
            Deduplicated results, keeping highest confidence.
        """
        if not results:
            return []

        seen: dict[str, dict] = {}
        for r in results:
            text = r["text_content"].lower().strip()
            if text in seen:
                if r["confidence"] > seen[text]["confidence"]:
                    seen[text] = r
            else:
                seen[text] = r

        return list(seen.values())

    def unload_model(self) -> None:
        """Unload the OCR reader."""
        self.reader = None
        logger.debug("EasyOCR unloaded")

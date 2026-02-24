"""Face detection and embedding extraction using InsightFace.

Returns face records aligned with official schema:
- bbox as (bbox_x, bbox_y, bbox_width, bbox_height)
- face_size classification (small/medium/large)
- is_frontal detection
- timestamp_seconds (not frame_timestamp)
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.gpu_utils import clear_gpu_memory, get_device
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class FaceDetector:
    """Detects faces and generates embeddings using InsightFace.

    Uses RetinaFace for detection and ArcFace for 512-dimensional
    face embeddings.

    Args:
        detection_threshold: Minimum confidence for face detection.
        device: Compute device ('cuda' or 'cpu'). Auto-detected if None.
    """

    def __init__(
        self,
        detection_threshold: float = 0.8,
        device: Optional[str] = None,
    ) -> None:
        self.detection_threshold = detection_threshold
        self.device = device or get_device()
        self.model = None

    def load_model(self) -> None:
        """Load the InsightFace analysis model."""
        from insightface.app import FaceAnalysis

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device == "cuda"
            else ["CPUExecutionProvider"]
        )

        self.model = FaceAnalysis(
            name="buffalo_l",
            providers=providers,
        )
        self.model.prepare(ctx_id=0 if self.device == "cuda" else -1)
        logger.info("InsightFace model loaded on %s", self.device)

    def detect_faces(
        self,
        frames: List[Tuple[np.ndarray, int, float]],
        video_id: int,
    ) -> List[Dict[str, Any]]:
        """Detect faces in a list of frames.

        Args:
            frames: List of (frame, frame_number, timestamp) tuples.
            video_id: Database ID of the video.

        Returns:
            List of face dictionaries matching schema:
                - video_id: int
                - frame_number: int
                - timestamp_seconds: float
                - bbox_x, bbox_y, bbox_width, bbox_height: int
                - confidence: float
                - face_size: str ('small', 'medium', 'large')
                - is_frontal: bool
                - embedding: bytes (512D vector serialized)
        """
        if self.model is None:
            self.load_model()

        all_faces = []

        for frame, frame_number, timestamp in frames:
            try:
                frame_h, frame_w = frame.shape[:2]
                frame_area = frame_h * frame_w
                detections = self.model.get(frame)

                for det in detections:
                    score = float(det.det_score)
                    if score < self.detection_threshold:
                        continue

                    bbox = det.bbox.astype(float)
                    embedding = det.embedding

                    # Convert from (x1, y1, x2, y2) to (x, y, width, height)
                    x1, y1, x2, y2 = bbox
                    bbox_x = int(round(x1))
                    bbox_y = int(round(y1))
                    bbox_width = int(round(x2 - x1))
                    bbox_height = int(round(y2 - y1))

                    # Classify face size relative to frame
                    face_area = bbox_width * bbox_height
                    face_size = self._classify_face_size(face_area, frame_area)

                    # Estimate if frontal based on bbox aspect ratio and landmarks
                    is_frontal = self._estimate_frontal(det, bbox_width, bbox_height)

                    face_record = {
                        "video_id": video_id,
                        "frame_number": frame_number,
                        "timestamp_seconds": round(timestamp, 3),
                        "bbox_x": bbox_x,
                        "bbox_y": bbox_y,
                        "bbox_width": bbox_width,
                        "bbox_height": bbox_height,
                        "confidence": round(score, 4),
                        "face_size": face_size,
                        "is_frontal": is_frontal,
                        "embedding": embedding.tobytes() if embedding is not None else None,
                    }
                    all_faces.append(face_record)

            except Exception as e:
                logger.warning(
                    "Face detection failed on frame %d: %s", frame_number, e
                )
                continue

        logger.debug(
            "Detected %d faces in %d frames for video_id=%d",
            len(all_faces),
            len(frames),
            video_id,
        )
        return all_faces

    def _classify_face_size(self, face_area: int, frame_area: int) -> str:
        """Classify face size relative to the frame.

        Args:
            face_area: Area of the face bounding box in pixels.
            frame_area: Total frame area in pixels.

        Returns:
            'small', 'medium', or 'large'.
        """
        if frame_area == 0:
            return "medium"

        ratio = face_area / frame_area
        if ratio < 0.02:
            return "small"
        elif ratio < 0.10:
            return "medium"
        else:
            return "large"

    def _estimate_frontal(self, detection: Any, bbox_w: int, bbox_h: int) -> bool:
        """Estimate if a face is roughly frontal.

        Uses facial landmark positions if available, otherwise
        falls back to aspect ratio heuristic.

        Args:
            detection: InsightFace detection object.
            bbox_w: Bounding box width.
            bbox_h: Bounding box height.

        Returns:
            True if the face appears to be frontal.
        """
        try:
            # InsightFace provides 5 key landmarks:
            # [left_eye, right_eye, nose, left_mouth, right_mouth]
            landmarks = detection.kps
            if landmarks is not None and len(landmarks) >= 5:
                left_eye = landmarks[0]
                right_eye = landmarks[1]
                nose = landmarks[2]

                # Check horizontal symmetry of nose relative to eyes
                eye_center_x = (left_eye[0] + right_eye[0]) / 2
                eye_distance = abs(right_eye[0] - left_eye[0])

                if eye_distance > 0:
                    nose_offset = abs(nose[0] - eye_center_x) / eye_distance
                    # Frontal if nose is roughly centered between eyes
                    return nose_offset < 0.25
        except (AttributeError, IndexError, TypeError):
            pass

        # Fallback: frontal faces tend to have width/height ratio near 0.7-1.0
        if bbox_h > 0:
            ratio = bbox_w / bbox_h
            return 0.6 < ratio < 1.1

        # Zero-height bbox is degenerate — cannot determine orientation
        return False

    def extract_face_thumbnail(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        output_path: str,
    ) -> Optional[str]:
        """Extract and save a face crop from a frame.

        Args:
            frame: Video frame as numpy array (BGR).
            bbox: Bounding box (x, y, width, height).
            output_path: Path to save the face image.

        Returns:
            Path to saved image, or None on failure.
        """
        try:
            x, y, w, h = bbox
            frame_h, frame_w = frame.shape[:2]

            # Add padding (20% of face size)
            pad_x = int(w * 0.2)
            pad_y = int(h * 0.2)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(frame_w, x + w + pad_x)
            y2 = min(frame_h, y + h + pad_y)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                return None

            face_crop = cv2.resize(face_crop, (112, 112))
            cv2.imwrite(output_path, face_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return output_path

        except Exception as e:
            logger.warning("Face thumbnail extraction failed: %s", e)
            return None

    def unload_model(self) -> None:
        """Unload InsightFace model to free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
            clear_gpu_memory()
            logger.info("InsightFace model unloaded")

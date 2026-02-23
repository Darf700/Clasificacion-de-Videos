"""Face detection and embedding extraction using InsightFace."""

from typing import Optional

import numpy as np

from utils.gpu_utils import clear_gpu_memory
from utils.logging_utils import get_logger

logger = get_logger("face_detector")


class FaceDetector:
    """Detects faces and generates embeddings using InsightFace."""

    def __init__(self, detection_threshold: float = 0.8, device: str = "cuda"):
        """Initialize face detector.

        Args:
            detection_threshold: Minimum confidence for face detection.
            device: Device to use ('cuda' or 'cpu').
        """
        self.detection_threshold = detection_threshold
        self.device = device
        self.app = None

    def load_model(self) -> None:
        """Load the InsightFace model."""
        from insightface.app import FaceAnalysis

        logger.info("Loading InsightFace model")
        ctx_id = 0 if self.device == "cuda" else -1
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device == "cuda"
            else ["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        logger.info("InsightFace model loaded")

    def detect_faces(
        self,
        frames: list[np.ndarray],
        video_duration: float = 0.0,
    ) -> list[dict]:
        """Detect faces in a list of frames.

        Args:
            frames: List of frames as numpy arrays (RGB).
            video_duration: Video duration for timestamp calculation.

        Returns:
            List of face detection dictionaries.
        """
        if not frames:
            return []

        if self.app is None:
            self.load_model()

        all_faces = []
        total_frames = len(frames)

        for i, frame in enumerate(frames):
            timestamp = (video_duration * i / total_frames) if total_frames > 0 else 0

            try:
                # InsightFace expects BGR
                frame_bgr = frame[:, :, ::-1] if frame.shape[2] == 3 else frame
                detected = self.app.get(frame_bgr)

                for face in detected:
                    if face.det_score < self.detection_threshold:
                        continue

                    bbox = face.bbox.astype(int)
                    bw = int(bbox[2] - bbox[0])
                    bh = int(bbox[3] - bbox[1])

                    # Classify face size
                    face_area = bw * bh
                    frame_area = frame.shape[0] * frame.shape[1]
                    ratio = face_area / frame_area if frame_area > 0 else 0

                    if ratio > 0.1:
                        face_size = "large"
                    elif ratio > 0.02:
                        face_size = "medium"
                    else:
                        face_size = "small"

                    face_data = {
                        "frame_number": i,
                        "timestamp_seconds": round(timestamp, 2),
                        "bbox_x": int(bbox[0]),
                        "bbox_y": int(bbox[1]),
                        "bbox_width": bw,
                        "bbox_height": bh,
                        "confidence": round(float(face.det_score), 4),
                        "face_size": face_size,
                        "is_frontal": self._is_frontal(face),
                        "embedding": face.normed_embedding.tobytes()
                        if hasattr(face, "normed_embedding") and face.normed_embedding is not None
                        else None,
                    }
                    all_faces.append(face_data)

            except Exception as e:
                logger.debug(f"Face detection failed on frame {i}: {e}")

        logger.debug(f"Detected {len(all_faces)} faces across {total_frames} frames")
        return all_faces

    def _is_frontal(self, face) -> bool:
        """Check if a face is roughly frontal based on landmarks.

        Args:
            face: InsightFace detection result.

        Returns:
            True if face appears frontal.
        """
        try:
            if hasattr(face, "pose") and face.pose is not None:
                # Yaw angle close to 0 means frontal
                yaw = abs(face.pose[1]) if len(face.pose) > 1 else 0
                return yaw < 30
            if hasattr(face, "kps") and face.kps is not None:
                # Check if left and right eye are roughly symmetric
                kps = face.kps
                left_eye = kps[0]
                right_eye = kps[1]
                nose = kps[2]
                dist_left = abs(nose[0] - left_eye[0])
                dist_right = abs(nose[0] - right_eye[0])
                if max(dist_left, dist_right) > 0:
                    ratio = min(dist_left, dist_right) / max(dist_left, dist_right)
                    return ratio > 0.5
        except Exception:
            pass
        return True

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        self.app = None
        clear_gpu_memory()
        logger.debug("InsightFace model unloaded")

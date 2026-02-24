"""Tests for modules.face_detector - FaceDetector (pure logic methods)."""

from unittest.mock import MagicMock

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="opencv not installed")

from modules.face_detector import FaceDetector


class TestFaceDetector:
    @pytest.fixture
    def detector(self):
        return FaceDetector(detection_threshold=0.8)

    # --- _classify_face_size ---

    def test_face_size_small(self, detector):
        frame_area = 1920 * 1080
        face_area = int(frame_area * 0.01)  # 1% of frame
        assert detector._classify_face_size(face_area, frame_area) == "small"

    def test_face_size_medium(self, detector):
        frame_area = 1920 * 1080
        face_area = int(frame_area * 0.05)  # 5% of frame
        assert detector._classify_face_size(face_area, frame_area) == "medium"

    def test_face_size_large(self, detector):
        frame_area = 1920 * 1080
        face_area = int(frame_area * 0.15)  # 15% of frame
        assert detector._classify_face_size(face_area, frame_area) == "large"

    def test_face_size_zero_frame(self, detector):
        assert detector._classify_face_size(100, 0) == "medium"

    def test_face_size_boundary_small_medium(self, detector):
        frame_area = 10000
        face_area = 200  # exactly 2%
        assert detector._classify_face_size(face_area, frame_area) == "medium"

    def test_face_size_boundary_medium_large(self, detector):
        frame_area = 10000
        face_area = 1000  # exactly 10%
        assert detector._classify_face_size(face_area, frame_area) == "large"

    # --- _estimate_frontal ---

    def test_frontal_with_symmetric_landmarks(self, detector):
        det = MagicMock()
        det.kps = np.array([
            [100, 100],  # left_eye
            [200, 100],  # right_eye
            [150, 150],  # nose (centered)
            [120, 200],  # left_mouth
            [180, 200],  # right_mouth
        ], dtype=np.float32)
        assert detector._estimate_frontal(det, 100, 130) == True

    def test_non_frontal_with_asymmetric_landmarks(self, detector):
        det = MagicMock()
        det.kps = np.array([
            [100, 100],
            [200, 100],
            [100, 150],  # nose far left
            [120, 200],
            [180, 200],
        ], dtype=np.float32)
        assert detector._estimate_frontal(det, 100, 130) == False

    def test_frontal_fallback_bbox_ratio(self, detector):
        """Without landmarks, use bbox aspect ratio."""
        det = MagicMock()
        det.kps = None
        # Width/height ratio ~0.8 -> frontal
        assert detector._estimate_frontal(det, 80, 100) is True

    def test_non_frontal_fallback_bbox_narrow(self, detector):
        det = MagicMock()
        det.kps = None
        # Very narrow bbox (ratio 0.3) -> not frontal
        assert detector._estimate_frontal(det, 30, 100) is False

    def test_zero_height_bbox_returns_false(self, detector):
        """Degenerate zero-height bbox should return False, not True."""
        det = MagicMock()
        det.kps = None
        assert detector._estimate_frontal(det, 100, 0) is False

    def test_frontal_with_exception_in_landmarks(self, detector):
        """If landmarks processing raises, fall back to bbox ratio."""
        det = MagicMock()
        det.kps = "not an array"  # Will cause IndexError
        # Fallback: 80/100 = 0.8 -> frontal
        assert detector._estimate_frontal(det, 80, 100) is True

    # --- extract_face_thumbnail ---

    def test_extract_face_thumbnail(self, detector, tmp_dir):
        import os
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        output = os.path.join(tmp_dir, "face.jpg")
        result = detector.extract_face_thumbnail(frame, (100, 100, 50, 60), output)
        assert result == output
        assert os.path.exists(output)

    def test_extract_face_thumbnail_empty_crop(self, detector, tmp_dir):
        import os
        frame = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        output = os.path.join(tmp_dir, "face.jpg")
        # bbox that results in empty crop
        result = detector.extract_face_thumbnail(frame, (100, 100, 0, 0), output)
        assert result is None

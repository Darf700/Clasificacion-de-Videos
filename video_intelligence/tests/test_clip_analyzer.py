"""Tests for modules.clip_analyzer - CLIPAnalyzer (score computation logic)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
cv2 = pytest.importorskip("cv2", reason="opencv not installed")

from modules.clip_analyzer import CLIPAnalyzer


class TestCLIPAnalyzerScoreComputation:
    """Test the _compute_similarity method directly."""

    @pytest.fixture
    def analyzer(self):
        return CLIPAnalyzer()

    def test_identical_vectors_max_similarity(self, analyzer):
        vec = torch.randn(1, 512)
        vec = vec / vec.norm(dim=-1, keepdim=True)
        scores = analyzer._compute_similarity(vec, vec)
        assert len(scores) == 1
        assert scores[0] == pytest.approx(1.0, abs=0.01)

    def test_opposite_vectors_min_similarity(self, analyzer):
        vec = torch.randn(1, 512)
        vec = vec / vec.norm(dim=-1, keepdim=True)
        scores = analyzer._compute_similarity(vec, -vec)
        assert len(scores) == 1
        assert scores[0] == pytest.approx(0.0, abs=0.01)

    def test_orthogonal_vectors_mid_similarity(self, analyzer):
        # Create two orthogonal unit vectors
        v1 = torch.zeros(1, 512)
        v1[0, 0] = 1.0
        v2 = torch.zeros(1, 512)
        v2[0, 1] = 1.0
        scores = analyzer._compute_similarity(v1, v2)
        assert len(scores) == 1
        assert scores[0] == pytest.approx(0.5, abs=0.01)

    def test_batch_similarity(self, analyzer):
        images = torch.randn(5, 512)
        images = images / images.norm(dim=-1, keepdim=True)
        text = torch.randn(1, 512)
        text = text / text.norm(dim=-1, keepdim=True)
        scores = analyzer._compute_similarity(images, text)
        assert len(scores) == 5
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_scores_always_in_0_1(self, analyzer):
        """Clamping should ensure scores are always in [0, 1]."""
        for _ in range(10):
            img = torch.randn(3, 512)
            txt = torch.randn(1, 512)
            scores = analyzer._compute_similarity(img, txt)
            assert all(0.0 <= s <= 1.0 for s in scores)

    def test_single_image_returns_list(self, analyzer):
        img = torch.randn(1, 512)
        txt = torch.randn(1, 512)
        scores = analyzer._compute_similarity(img, txt)
        assert isinstance(scores, list)
        assert len(scores) == 1

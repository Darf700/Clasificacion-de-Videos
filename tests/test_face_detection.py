"""Tests for face detection and clustering modules."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.face_clusterer import FaceClusterer


class TestFaceClusterer:
    """Tests for FaceClusterer."""

    @pytest.fixture
    def clusterer(self):
        return FaceClusterer(eps=0.6, min_samples=2)

    def test_cluster_empty(self, clusterer):
        result = clusterer.cluster([])
        assert result == {}

    def test_cluster_too_few(self, clusterer):
        records = [{"id": 1, "embedding": np.random.randn(512).astype(np.float32).tobytes()}]
        result = clusterer.cluster(records)
        assert result == {}

    def test_cluster_similar_faces(self, clusterer):
        """Test that similar embeddings get clustered together."""
        base = np.random.randn(512).astype(np.float32)
        base = base / np.linalg.norm(base)

        records = []
        for i in range(5):
            # Add small noise to create similar embeddings
            noisy = base + np.random.randn(512).astype(np.float32) * 0.05
            noisy = noisy / np.linalg.norm(noisy)
            records.append({"id": i, "embedding": noisy.astype(np.float32).tobytes()})

        result = clusterer.cluster(records)
        # All should be in the same cluster
        clusters = set(result.values())
        non_noise = clusters - {-1}
        assert len(non_noise) >= 1

    def test_cluster_different_faces(self, clusterer):
        """Test that different embeddings get different clusters."""
        records = []
        for i in range(10):
            # Create distinct embeddings for two groups
            if i < 5:
                base = np.ones(512, dtype=np.float32)
            else:
                base = -np.ones(512, dtype=np.float32)
            noise = np.random.randn(512).astype(np.float32) * 0.05
            emb = base + noise
            emb = emb / np.linalg.norm(emb)
            records.append({"id": i, "embedding": emb.astype(np.float32).tobytes()})

        result = clusterer.cluster(records)
        clusters = set(v for v in result.values() if v != -1)
        assert len(clusters) >= 2

    def test_build_cluster_stats(self, clusterer):
        face_cluster_map = {1: 0, 2: 0, 3: 1, 4: -1}
        face_records = [
            {"id": 1, "video_id": 10, "embedding": b"...", "confidence": 0.95},
            {"id": 2, "video_id": 10, "embedding": b"...", "confidence": 0.90},
            {"id": 3, "video_id": 20, "embedding": b"...", "confidence": 0.85},
            {"id": 4, "video_id": 20, "embedding": b"...", "confidence": 0.80},
        ]
        stats = clusterer.build_cluster_stats(face_cluster_map, face_records)
        assert len(stats) == 2  # cluster 0 and 1 (not -1)

        cluster_0 = next(s for s in stats if s["cluster_id"] == 0)
        assert cluster_0["face_count"] == 2

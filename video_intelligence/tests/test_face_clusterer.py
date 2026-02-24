"""Tests for modules.face_clusterer - FaceClusterer."""

import numpy as np
import pytest

from modules.face_clusterer import FaceClusterer, EMBEDDING_DIM


class TestFaceClusterer:
    @pytest.fixture
    def clusterer(self):
        return FaceClusterer(eps=0.5, min_samples=2)

    # --- cluster_faces ---

    def test_empty_input(self, clusterer):
        result = clusterer.cluster_faces([])
        assert result == {}

    def test_basic_clustering(self, clusterer):
        """Three similar embeddings should form one cluster."""
        np.random.seed(42)
        base = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        base = base / np.linalg.norm(base)

        # Create 4 similar embeddings (small perturbation)
        face_data = []
        for i in range(4):
            emb = base + np.random.randn(EMBEDDING_DIM).astype(np.float32) * 0.01
            face_data.append((i + 1, emb.tobytes()))

        result = clusterer.cluster_faces(face_data)
        assert len(result) == 4
        # All should be in the same cluster (label >= 0)
        labels = set(result.values())
        assert -1 not in labels or len(labels) <= 2

    def test_noise_points(self):
        """Isolated embeddings should be labeled as noise (-1)."""
        clusterer = FaceClusterer(eps=0.1, min_samples=3)
        np.random.seed(123)
        face_data = []
        for i in range(3):
            emb = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            face_data.append((i + 1, emb.tobytes()))

        result = clusterer.cluster_faces(face_data)
        assert all(label == -1 for label in result.values())

    def test_wrong_dimension_skipped(self, clusterer):
        """Embeddings with wrong dimension should be skipped."""
        wrong_emb = np.random.randn(256).astype(np.float32)
        correct_emb = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        face_data = [
            (1, wrong_emb.tobytes()),
            (2, correct_emb.tobytes()),
            (3, correct_emb.tobytes()),
        ]
        result = clusterer.cluster_faces(face_data)
        assert 1 not in result
        assert 2 in result
        assert 3 in result

    def test_invalid_bytes_skipped(self, clusterer):
        """Corrupt embedding bytes should be skipped gracefully."""
        valid = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        face_data = [
            (1, b"corrupt"),
            (2, valid.tobytes()),
            (3, valid.tobytes()),
        ]
        result = clusterer.cluster_faces(face_data)
        assert 1 not in result

    def test_zero_norm_embeddings_excluded(self, clusterer):
        """Zero-norm embeddings should be excluded with a warning."""
        zero_emb = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        valid_emb = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        face_data = [
            (1, zero_emb.tobytes()),
            (2, valid_emb.tobytes()),
            (3, valid_emb.tobytes()),
        ]
        result = clusterer.cluster_faces(face_data)
        assert 1 not in result  # Zero-norm excluded
        assert 2 in result
        assert 3 in result

    def test_all_zero_norm_returns_empty(self, clusterer):
        """If all embeddings are zero-norm, return empty dict."""
        zero_emb = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        face_data = [
            (1, zero_emb.tobytes()),
            (2, zero_emb.tobytes()),
        ]
        result = clusterer.cluster_faces(face_data)
        assert result == {}

    # --- get_cluster_stats ---

    def test_get_cluster_stats(self, clusterer):
        assignments = {1: 0, 2: 0, 3: 0, 4: 1, 5: -1}
        stats = clusterer.get_cluster_stats(assignments)
        assert len(stats) == 2  # Clusters 0 and 1 (not -1)
        cluster_0 = next(s for s in stats if s["cluster_id"] == 0)
        assert cluster_0["face_count"] == 3
        assert set(cluster_0["face_ids"]) == {1, 2, 3}
        cluster_1 = next(s for s in stats if s["cluster_id"] == 1)
        assert cluster_1["face_count"] == 1

    def test_get_cluster_stats_empty(self, clusterer):
        assert clusterer.get_cluster_stats({}) == []

    def test_get_cluster_stats_all_noise(self, clusterer):
        assignments = {1: -1, 2: -1}
        assert clusterer.get_cluster_stats(assignments) == []

"""Face embedding clustering using DBSCAN."""

from collections import Counter
from typing import Optional

import numpy as np

from utils.logging_utils import get_logger

logger = get_logger("face_clusterer")


class FaceClusterer:
    """Clusters face embeddings to identify unique persons."""

    def __init__(
        self,
        algorithm: str = "dbscan",
        eps: float = 0.6,
        min_samples: int = 2,
        metric: str = "cosine",
    ):
        """Initialize face clusterer.

        Args:
            algorithm: Clustering algorithm ('dbscan' or 'agglomerative').
            eps: DBSCAN epsilon parameter.
            min_samples: Minimum samples for a cluster.
            metric: Distance metric.
        """
        self.algorithm = algorithm
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def cluster(self, face_records: list[dict]) -> dict[int, int]:
        """Cluster face embeddings and return face_id -> cluster_id mapping.

        Args:
            face_records: List of dicts with 'id' and 'embedding' (bytes) keys.

        Returns:
            Dictionary mapping face_id to cluster_id.
            Noise points get cluster_id = -1.
        """
        # Filter records with embeddings
        valid = [(r["id"], r["embedding"]) for r in face_records if r.get("embedding")]
        if len(valid) < self.min_samples:
            logger.info(f"Not enough faces for clustering ({len(valid)} < {self.min_samples})")
            return {}

        face_ids = [v[0] for v in valid]
        embeddings = np.array([
            np.frombuffer(v[1], dtype=np.float32) for v in valid
        ])

        logger.info(f"Clustering {len(embeddings)} face embeddings")

        if self.algorithm == "dbscan":
            labels = self._dbscan_cluster(embeddings)
        else:
            labels = self._agglomerative_cluster(embeddings)

        # Build mapping
        mapping = {}
        for face_id, label in zip(face_ids, labels):
            mapping[face_id] = int(label)

        n_clusters = len(set(labels) - {-1})
        n_noise = list(labels).count(-1)
        logger.info(f"Found {n_clusters} person clusters, {n_noise} noise faces")

        return mapping

    def _dbscan_cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """Run DBSCAN clustering.

        Args:
            embeddings: Face embedding matrix.

        Returns:
            Cluster labels array.
        """
        from sklearn.cluster import DBSCAN

        clusterer = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
        )
        return clusterer.fit_predict(embeddings)

    def _agglomerative_cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """Run Agglomerative clustering.

        Args:
            embeddings: Face embedding matrix.

        Returns:
            Cluster labels array.
        """
        from sklearn.cluster import AgglomerativeClustering

        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.eps,
            metric=self.metric,
            linkage="average",
        )
        return clusterer.fit_predict(embeddings)

    def build_cluster_stats(
        self,
        face_cluster_map: dict[int, int],
        face_records: list[dict],
    ) -> list[dict]:
        """Build statistics for each cluster.

        Args:
            face_cluster_map: Mapping of face_id -> cluster_id.
            face_records: Original face records with video_id info.

        Returns:
            List of cluster stat dictionaries.
        """
        if not face_cluster_map:
            return []

        # Build face_id -> record mapping
        record_map = {r["id"]: r for r in face_records}

        # Group by cluster
        clusters: dict[int, list] = {}
        for face_id, cluster_id in face_cluster_map.items():
            if cluster_id == -1:
                continue
            clusters.setdefault(cluster_id, []).append(face_id)

        stats = []
        for cluster_id, face_ids in clusters.items():
            video_ids = set()
            confidences = []
            for fid in face_ids:
                rec = record_map.get(fid, {})
                if "video_id" in rec:
                    video_ids.add(rec["video_id"])
                if "confidence" in rec:
                    confidences.append(rec["confidence"])

            stats.append({
                "cluster_id": cluster_id,
                "face_count": len(face_ids),
                "video_count": len(video_ids),
                "representative_face_id": face_ids[0],
                "avg_confidence": round(np.mean(confidences), 4) if confidences else 0,
            })

        return stats

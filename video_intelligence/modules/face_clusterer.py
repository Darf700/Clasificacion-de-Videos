"""Face embedding clustering using DBSCAN."""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

from utils.logging_utils import get_logger

logger = get_logger(__name__)

# ArcFace embedding dimension
EMBEDDING_DIM = 512


class FaceClusterer:
    """Clusters face embeddings to group the same person across videos.

    Uses DBSCAN on cosine distance to find groups of similar faces
    without requiring a pre-defined number of clusters.

    Args:
        eps: Maximum distance between two samples in a cluster.
            Lower = stricter matching. Range: 0.0-1.0.
        min_samples: Minimum faces needed to form a cluster.
        thumbnail_dir: Directory to save cluster representative images.
    """

    def __init__(
        self,
        eps: float = 0.65,
        min_samples: int = 3,
        thumbnail_dir: Optional[str] = None,
    ) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.thumbnail_dir = thumbnail_dir
        if thumbnail_dir:
            os.makedirs(thumbnail_dir, exist_ok=True)

    def cluster_faces(
        self,
        face_data: List[Tuple[int, bytes]],
    ) -> Dict[int, int]:
        """Cluster face embeddings using DBSCAN.

        Args:
            face_data: List of (face_id, embedding_bytes) tuples from database.

        Returns:
            Dictionary mapping face_id -> cluster_id.
            Noise points (unclustered) are assigned cluster_id = -1.
        """
        if not face_data:
            logger.info("No faces to cluster")
            return {}

        # Deserialize embeddings
        face_ids = []
        embeddings = []

        for face_id, emb_bytes in face_data:
            try:
                embedding = np.frombuffer(emb_bytes, dtype=np.float32)
                if embedding.shape[0] == EMBEDDING_DIM:
                    face_ids.append(face_id)
                    embeddings.append(embedding)
                else:
                    logger.warning(
                        "Face %d has wrong embedding dimension: %d",
                        face_id,
                        embedding.shape[0],
                    )
            except Exception as e:
                logger.warning("Failed to deserialize embedding for face %d: %s", face_id, e)

        if not embeddings:
            logger.warning("No valid embeddings found for clustering")
            return {}

        # Normalize embeddings for cosine distance
        embeddings_array = np.vstack(embeddings)
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)

        # Detect and warn about zero-norm embeddings (degenerate)
        zero_mask = (norms.squeeze() == 0)
        n_zero = int(np.sum(zero_mask))
        if n_zero > 0:
            zero_ids = [face_ids[i] for i in np.where(zero_mask)[0]]
            logger.warning(
                "%d face(s) have zero-norm embeddings (will be excluded): %s",
                n_zero,
                zero_ids,
            )
            # Filter out zero-norm embeddings — they can't be meaningfully clustered
            valid_mask = ~zero_mask
            embeddings_array = embeddings_array[valid_mask]
            norms = norms[valid_mask]
            face_ids = [fid for fid, keep in zip(face_ids, valid_mask) if keep]

            if len(face_ids) == 0:
                logger.warning("No valid embeddings remain after filtering zero-norms")
                return {}

        embeddings_normalized = embeddings_array / norms

        # Run DBSCAN with cosine metric
        logger.info(
            "Clustering %d face embeddings (eps=%.2f, min_samples=%d)",
            len(embeddings_normalized),
            self.eps,
            self.min_samples,
        )

        clustering = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric="cosine",
            n_jobs=-1,
        )
        labels = clustering.fit_predict(embeddings_normalized)

        # Build face_id -> cluster_id mapping
        assignments = {}
        for face_id, label in zip(face_ids, labels):
            assignments[face_id] = int(label)

        # Log cluster statistics
        unique_labels = set(labels)
        n_clusters = len(unique_labels - {-1})
        n_noise = int(np.sum(labels == -1))
        logger.info(
            "Clustering complete: %d clusters, %d noise points",
            n_clusters,
            n_noise,
        )

        return assignments

    def get_cluster_stats(
        self, assignments: Dict[int, int]
    ) -> List[Dict[str, Any]]:
        """Generate statistics for each cluster.

        Args:
            assignments: face_id -> cluster_id mapping.

        Returns:
            List of cluster info dictionaries with:
                - cluster_id: int
                - face_count: int
                - face_ids: list of face IDs
        """
        clusters: Dict[int, List[int]] = {}

        for face_id, cluster_id in assignments.items():
            if cluster_id == -1:
                continue
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(face_id)

        stats = []
        for cluster_id, face_ids in sorted(clusters.items()):
            stats.append({
                "cluster_id": cluster_id,
                "face_count": len(face_ids),
                "face_ids": face_ids,
            })

        return stats

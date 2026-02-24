"""SQLite database operations for Video Intelligence System.

Matches the official DATABASE_SCHEMA.sql with:
- videos (with format_type, orientation, aspect_ratio, etc.)
- faces (bbox as x/y/width/height, face_size, is_frontal)
- person_clusters (cluster_id PK, first/last seen tracking)
- video_persons (many-to-many relationship)
- transcriptions
- ocr_texts (text_content, bbox as x/y/width/height)
- visual_tags (tag, source)
- processing_log (batch_id TEXT, JSON distributions)
- FTS5 virtual tables for transcriptions and OCR
- Views: videos_full, person_clusters_stats, recent_batches
"""

import json
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Valid column names per table (whitelist for SQL injection prevention)
_VALID_COLUMNS = {
    "videos": {
        "id", "filename", "original_path", "final_path", "file_hash",
        "file_size_bytes", "duration_seconds", "width", "height",
        "aspect_ratio", "fps", "bitrate", "codec_video", "codec_audio",
        "format_type", "orientation", "has_audio", "has_dialogue",
        "audio_language", "has_text_overlay", "creation_date", "date_source",
        "year", "month", "month_name", "primary_theme", "theme_confidence",
        "secondary_themes", "face_count", "unique_persons", "processed_date",
        "processing_duration_seconds", "batch_id", "error_message",
        "needs_review", "created_at", "updated_at",
    },
    "faces": {
        "id", "video_id", "frame_number", "timestamp_seconds",
        "bbox_x", "bbox_y", "bbox_width", "bbox_height",
        "confidence", "face_size", "is_frontal", "embedding",
        "cluster_id", "created_at",
    },
    "person_clusters": {
        "cluster_id", "face_count", "video_count",
        "representative_face_id", "avg_confidence", "label",
        "first_seen_video_id", "last_seen_video_id",
        "first_seen_date", "last_seen_date", "created_at", "updated_at",
    },
    "transcriptions": {
        "id", "video_id", "full_text", "language",
        "confidence", "word_count", "processing_time_seconds", "created_at",
    },
    "ocr_texts": {
        "id", "video_id", "frame_number", "timestamp_seconds",
        "text_content", "language", "confidence",
        "bbox_x", "bbox_y", "bbox_width", "bbox_height", "created_at",
    },
    "visual_tags": {
        "id", "video_id", "tag", "confidence", "source", "created_at",
    },
    "processing_log": {
        "id", "batch_id", "batch_number", "videos_found", "videos_new",
        "videos_processed", "videos_failed", "videos_skipped",
        "processing_start", "processing_end", "duration_seconds",
        "themes_distribution", "years_distribution", "errors",
        "report_path", "export_path", "status", "created_at",
    },
}

# Regex for safe SQL identifier (alphanumeric + underscore only)
_SAFE_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_columns(columns: List[str], table: str) -> None:
    """Validate that column names are safe and exist in the schema.

    Args:
        columns: List of column names to validate.
        table: Table name to check against.

    Raises:
        ValueError: If any column name is invalid or not in the schema.
    """
    valid = _VALID_COLUMNS.get(table)
    for col in columns:
        if not _SAFE_IDENTIFIER.match(col):
            raise ValueError(f"Invalid column name: '{col}'")
        if valid and col not in valid:
            raise ValueError(f"Unknown column '{col}' for table '{table}'")


class DatabaseManager:
    """Manages all SQLite database operations.

    Provides transactional inserts, duplicate detection, and query methods
    aligned with the official schema including FTS5, views, and triggers.

    Args:
        db_path: Path to the SQLite database file.
        schema_path: Path to the SQL schema file for initialization.
    """

    def __init__(self, db_path: str, schema_path: str) -> None:
        self.db_path = db_path
        self.schema_path = schema_path
        self._ensure_directory()
        self._init_database()

    def _ensure_directory(self) -> None:
        """Create the database directory if it doesn't exist."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

    def _init_database(self) -> None:
        """Initialize database with schema if tables don't exist."""
        if not os.path.exists(self.schema_path):
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")

        with open(self.schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        with self._connect() as conn:
            conn.executescript(schema_sql)
            logger.info("Database initialized at %s", self.db_path)

    @contextmanager
    def _connect(self):
        """Context manager for database connections.

        Yields:
            sqlite3.Connection with row_factory set to sqlite3.Row.
        """
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _build_insert_sql(self, table: str, data: Dict[str, Any]) -> Tuple[str, list]:
        """Build a safe INSERT SQL statement with column validation.

        Args:
            table: Target table name.
            data: Column-value mapping.

        Returns:
            Tuple of (sql_string, values_list).
        """
        cols = list(data.keys())
        _validate_columns(cols, table)
        columns = ", ".join(cols)
        placeholders = ", ".join(["?"] * len(cols))
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        return sql, list(data.values())

    def _build_update_sql(self, table: str, updates: Dict[str, Any],
                          where_col: str, where_val: Any) -> Tuple[str, list]:
        """Build a safe UPDATE SQL statement with column validation.

        Args:
            table: Target table name.
            updates: Column-value mapping for SET clause.
            where_col: Column for WHERE clause.
            where_val: Value for WHERE clause.

        Returns:
            Tuple of (sql_string, values_list).
        """
        cols = list(updates.keys())
        _validate_columns(cols, table)
        _validate_columns([where_col], table)
        set_clause = ", ".join([f"{k} = ?" for k in cols])
        sql = f"UPDATE {table} SET {set_clause} WHERE {where_col} = ?"
        return sql, list(updates.values()) + [where_val]

    # =========================================================================
    # Duplicate Detection
    # =========================================================================

    def video_exists(self, file_hash: str) -> bool:
        """Check if a video with this hash already exists.

        Args:
            file_hash: MD5 hash of the video file.

        Returns:
            True if the hash exists in the database.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM videos WHERE file_hash = ?", (file_hash,)
            ).fetchone()
            return row is not None

    # =========================================================================
    # Video CRUD
    # =========================================================================

    def insert_video(self, video_data: Dict[str, Any]) -> int:
        """Insert a new video record.

        Args:
            video_data: Dictionary with columns matching the 'videos' table.
                Required: filename, original_path, file_hash.

        Returns:
            The row ID of the inserted video.
        """
        sql, values = self._build_insert_sql("videos", video_data)

        with self._connect() as conn:
            cursor = conn.execute(sql, values)
            video_id = cursor.lastrowid
            logger.debug("Inserted video id=%d: %s", video_id, video_data.get("filename", ""))
            return video_id

    def update_video(self, video_id: int, updates: Dict[str, Any]) -> None:
        """Update a video record.

        Args:
            video_id: ID of the video to update.
            updates: Dictionary of column=value pairs to update.
        """
        sql, values = self._build_update_sql("videos", updates, "id", video_id)

        with self._connect() as conn:
            conn.execute(sql, values)
            logger.debug("Updated video id=%d", video_id)

    def get_video_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Get a video record by its file hash.

        Args:
            file_hash: MD5 hash of the video.

        Returns:
            Dictionary of video data, or None if not found.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM videos WHERE file_hash = ?", (file_hash,)
            ).fetchone()
            return dict(row) if row else None

    def get_video_by_id(self, video_id: int) -> Optional[Dict[str, Any]]:
        """Get a video record by its ID.

        Args:
            video_id: Database ID of the video.

        Returns:
            Dictionary of video data, or None if not found.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM videos WHERE id = ?", (video_id,)
            ).fetchone()
            return dict(row) if row else None

    # =========================================================================
    # Face Operations
    # =========================================================================

    def insert_faces(self, faces: List[Dict[str, Any]]) -> List[int]:
        """Insert multiple face detection records in a single transaction.

        Args:
            faces: List of face dictionaries.

        Returns:
            List of inserted row IDs.
        """
        if not faces:
            return []

        ids = []
        with self._connect() as conn:
            for face in faces:
                sql, values = self._build_insert_sql("faces", face)
                cursor = conn.execute(sql, values)
                ids.append(cursor.lastrowid)

        logger.debug("Inserted %d face records", len(ids))
        return ids

    def get_all_face_embeddings(self) -> List[Tuple[int, bytes]]:
        """Get all face embeddings for clustering.

        Returns:
            List of (face_id, embedding_blob) tuples.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, embedding FROM faces WHERE embedding IS NOT NULL"
            ).fetchall()
            return [(row["id"], row["embedding"]) for row in rows]

    def get_faces_by_video(self, video_id: int) -> List[Dict[str, Any]]:
        """Get all faces detected in a specific video.

        Args:
            video_id: Database ID of the video.

        Returns:
            List of face dictionaries.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM faces WHERE video_id = ?", (video_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def update_face_clusters(self, assignments: Dict[int, int]) -> None:
        """Update cluster_id for multiple faces in a single transaction.

        Args:
            assignments: Mapping of face_id -> cluster_id.
        """
        if not assignments:
            return

        with self._connect() as conn:
            conn.executemany(
                "UPDATE faces SET cluster_id = ? WHERE id = ?",
                [(cluster_id, face_id) for face_id, cluster_id in assignments.items()],
            )
        logger.debug("Updated cluster assignments for %d faces", len(assignments))

    # =========================================================================
    # Person Cluster Operations
    # =========================================================================

    def insert_person_cluster(self, cluster_data: Dict[str, Any]) -> int:
        """Insert a new person cluster record.

        Args:
            cluster_data: Dictionary with person_clusters columns.

        Returns:
            cluster_id of the inserted record.
        """
        sql, values = self._build_insert_sql("person_clusters", cluster_data)

        with self._connect() as conn:
            cursor = conn.execute(sql, values)
            return cursor.lastrowid

    def update_person_cluster(self, cluster_id: int, updates: Dict[str, Any]) -> None:
        """Update a person cluster record.

        Args:
            cluster_id: The cluster ID to update.
            updates: Dictionary of column=value pairs.
        """
        sql, values = self._build_update_sql("person_clusters", updates, "cluster_id", cluster_id)

        with self._connect() as conn:
            conn.execute(sql, values)

    # =========================================================================
    # Video-Person Relationship
    # =========================================================================

    def upsert_video_person(self, video_id: int, cluster_id: int,
                            appearance_count: int, total_frames: int,
                            avg_confidence: float) -> None:
        """Insert or update a video-person relationship.

        Args:
            video_id: Database ID of the video.
            cluster_id: Cluster ID of the person.
            appearance_count: Number of face detections in this video.
            total_frames: Total frames analyzed.
            avg_confidence: Average detection confidence.
        """
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO video_persons
                   (video_id, cluster_id, appearance_count, total_frames, avg_confidence)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(video_id, cluster_id) DO UPDATE SET
                       appearance_count = excluded.appearance_count,
                       total_frames = excluded.total_frames,
                       avg_confidence = excluded.avg_confidence""",
                (video_id, cluster_id, appearance_count, total_frames, avg_confidence),
            )

    # =========================================================================
    # Transcription Operations
    # =========================================================================

    def insert_transcription(self, transcription: Dict[str, Any]) -> int:
        """Insert a transcription record.

        Args:
            transcription: Dictionary with transcription data.

        Returns:
            Row ID of the inserted record.
        """
        sql, values = self._build_insert_sql("transcriptions", transcription)
        # Use OR REPLACE for upsert behavior on video_id
        sql = sql.replace("INSERT INTO", "INSERT OR REPLACE INTO", 1)

        with self._connect() as conn:
            cursor = conn.execute(sql, values)
            logger.debug("Inserted transcription for video_id=%s", transcription.get("video_id"))
            return cursor.lastrowid

    # =========================================================================
    # OCR Operations
    # =========================================================================

    def insert_ocr_texts(self, ocr_texts: List[Dict[str, Any]]) -> List[int]:
        """Insert multiple OCR text records in a single transaction.

        Args:
            ocr_texts: List of OCR result dictionaries.

        Returns:
            List of inserted row IDs.
        """
        if not ocr_texts:
            return []

        ids = []
        with self._connect() as conn:
            for text_rec in ocr_texts:
                sql, values = self._build_insert_sql("ocr_texts", text_rec)
                cursor = conn.execute(sql, values)
                ids.append(cursor.lastrowid)

        logger.debug("Inserted %d OCR text records", len(ids))
        return ids

    # =========================================================================
    # Visual Tags Operations
    # =========================================================================

    def insert_visual_tags(self, tags: List[Dict[str, Any]]) -> List[int]:
        """Insert visual tag (CLIP theme) records in a single transaction.

        Args:
            tags: List of visual tag dictionaries.

        Returns:
            List of inserted row IDs.
        """
        if not tags:
            return []

        ids = []
        with self._connect() as conn:
            for tag in tags:
                sql, values = self._build_insert_sql("visual_tags", tag)
                cursor = conn.execute(sql, values)
                ids.append(cursor.lastrowid)

        logger.debug("Inserted %d visual tag records", len(ids))
        return ids

    # =========================================================================
    # Processing Log Operations
    # =========================================================================

    def create_batch_log(self, videos_found: int) -> str:
        """Create a new batch processing log entry.

        Uses a single transaction for atomic read-then-insert.

        Args:
            videos_found: Total number of videos discovered.

        Returns:
            The batch_id (UUID string) for this processing run.
        """
        batch_id = str(uuid.uuid4())[:8]

        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(batch_number), 0) + 1 as next_num FROM processing_log"
            ).fetchone()
            batch_number = row["next_num"]

            conn.execute(
                """INSERT INTO processing_log
                   (batch_id, batch_number, videos_found, videos_new, videos_processed,
                    videos_failed, videos_skipped, processing_start, status)
                   VALUES (?, ?, ?, 0, 0, 0, 0, ?, 'running')""",
                (batch_id, batch_number, videos_found, datetime.now().isoformat()),
            )

        logger.info("Created batch log #%d (id=%s) for %d videos", batch_number, batch_id, videos_found)
        return batch_id

    def update_batch_log(self, batch_id: str, updates: Dict[str, Any]) -> None:
        """Update a batch processing log entry.

        Args:
            batch_id: The batch ID string to update.
            updates: Dictionary of column=value pairs.
        """
        sql, values = self._build_update_sql("processing_log", updates, "batch_id", batch_id)

        with self._connect() as conn:
            conn.execute(sql, values)

    def increment_batch_counter(self, batch_id: str, counter: str) -> None:
        """Increment a counter in the batch log.

        Args:
            batch_id: The batch ID string.
            counter: Column name to increment (videos_processed, videos_failed, etc.).
        """
        allowed = {"videos_processed", "videos_failed", "videos_skipped", "videos_new"}
        if counter not in allowed:
            raise ValueError(f"Invalid counter column: '{counter}'")

        sql = f"UPDATE processing_log SET {counter} = {counter} + 1 WHERE batch_id = ?"
        with self._connect() as conn:
            conn.execute(sql, (batch_id,))

    # =========================================================================
    # Statistics & Reporting Queries
    # =========================================================================

    def get_batch_stats(self, batch_id: str) -> Dict[str, Any]:
        """Get statistics for a specific batch.

        Args:
            batch_id: The batch ID to query.

        Returns:
            Dictionary with batch statistics.
        """
        with self._connect() as conn:
            # Theme distribution
            rows = conn.execute(
                """SELECT primary_theme, COUNT(*) as cnt
                   FROM videos WHERE batch_id = ? AND primary_theme IS NOT NULL
                   GROUP BY primary_theme ORDER BY cnt DESC""",
                (batch_id,),
            ).fetchall()
            theme_dist = {row["primary_theme"]: row["cnt"] for row in rows}

            # Year/month distribution
            rows = conn.execute(
                """SELECT year, month, month_name, COUNT(*) as cnt
                   FROM videos WHERE batch_id = ?
                   GROUP BY year, month
                   ORDER BY year, month""",
                (batch_id,),
            ).fetchall()
            date_dist = [
                {
                    "year": row["year"],
                    "month": row["month"],
                    "month_name": row["month_name"],
                    "count": row["cnt"],
                }
                for row in rows
            ]

            # Face cluster count
            face_clusters = conn.execute(
                """SELECT COUNT(DISTINCT f.cluster_id) as cnt
                   FROM faces f JOIN videos v ON f.video_id = v.id
                   WHERE v.batch_id = ? AND f.cluster_id IS NOT NULL AND f.cluster_id >= 0""",
                (batch_id,),
            ).fetchone()["cnt"]

            # Videos with speech
            speech_count = conn.execute(
                """SELECT COUNT(*) as cnt
                   FROM transcriptions t JOIN videos v ON t.video_id = v.id
                   WHERE v.batch_id = ?""",
                (batch_id,),
            ).fetchone()["cnt"]

            # Needs review count
            review_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM videos WHERE batch_id = ? AND needs_review = 1",
                (batch_id,),
            ).fetchone()["cnt"]

            # Error count
            error_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM videos WHERE batch_id = ? AND error_message IS NOT NULL",
                (batch_id,),
            ).fetchone()["cnt"]

            # Total faces detected
            total_faces = conn.execute(
                """SELECT COUNT(*) as cnt FROM faces f
                   JOIN videos v ON f.video_id = v.id
                   WHERE v.batch_id = ?""",
                (batch_id,),
            ).fetchone()["cnt"]

            return {
                "theme_distribution": theme_dist,
                "date_distribution": date_dist,
                "unique_persons": face_clusters,
                "videos_with_speech": speech_count,
                "needs_review": review_count,
                "errors": error_count,
                "total_faces_detected": total_faces,
            }

    def get_all_videos_for_export(self, batch_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all video records for CSV export.

        Args:
            batch_id: Optional batch ID to filter by.

        Returns:
            List of video dictionaries.
        """
        with self._connect() as conn:
            if batch_id:
                rows = conn.execute(
                    "SELECT * FROM videos WHERE batch_id = ? ORDER BY id",
                    (batch_id,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM videos ORDER BY id").fetchall()

            return [dict(row) for row in rows]

    def get_total_video_count(self) -> int:
        """Get total number of videos in the database.

        Returns:
            Total count of video records.
        """
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM videos").fetchone()
            return row["cnt"]

    # =========================================================================
    # Full-Text Search
    # =========================================================================

    def search_transcriptions(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search transcriptions using FTS5 full-text search.

        Args:
            query: Search query (supports FTS5 syntax).
            limit: Maximum results to return.

        Returns:
            List of matching video records with transcription text.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT v.id, v.filename, v.primary_theme, t.full_text,
                          highlight(transcriptions_fts, 1, '<b>', '</b>') as highlight
                   FROM transcriptions_fts
                   JOIN transcriptions t ON transcriptions_fts.rowid = t.id
                   JOIN videos v ON t.video_id = v.id
                   WHERE transcriptions_fts MATCH ?
                   LIMIT ?""",
                (query, limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def search_ocr_texts(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search OCR texts using FTS5 full-text search.

        Args:
            query: Search query (supports FTS5 syntax).
            limit: Maximum results to return.

        Returns:
            List of matching video records with OCR text.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT v.id, v.filename, v.primary_theme, o.text_content,
                          highlight(ocr_texts_fts, 1, '<b>', '</b>') as highlight
                   FROM ocr_texts_fts
                   JOIN ocr_texts o ON ocr_texts_fts.rowid = o.id
                   JOIN videos v ON o.video_id = v.id
                   WHERE ocr_texts_fts MATCH ?
                   LIMIT ?""",
                (query, limit),
            ).fetchall()
            return [dict(row) for row in rows]

    # =========================================================================
    # View Queries
    # =========================================================================

    def get_videos_full(self, batch_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query the videos_full view.

        Args:
            batch_id: Optional batch filter.

        Returns:
            List of enriched video records.
        """
        with self._connect() as conn:
            if batch_id:
                rows = conn.execute(
                    "SELECT * FROM videos_full WHERE batch_id = ? ORDER BY id",
                    (batch_id,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM videos_full ORDER BY id").fetchall()
            return [dict(row) for row in rows]

    def get_person_clusters_stats(self) -> List[Dict[str, Any]]:
        """Query the person_clusters_stats view.

        Returns:
            List of cluster records with computed statistics.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM person_clusters_stats ORDER BY actual_face_count DESC"
            ).fetchall()
            return [dict(row) for row in rows]

    def get_recent_batches(self) -> List[Dict[str, Any]]:
        """Query the recent_batches view.

        Returns:
            List of recent batch processing records.
        """
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM recent_batches").fetchall()
            return [dict(row) for row in rows]

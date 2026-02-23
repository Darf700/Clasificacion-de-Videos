"""SQLite database operations for the Video Intelligence System."""

import json
import os
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from utils.logging_utils import get_logger

logger = get_logger("db_manager")


class DatabaseManager:
    """Manages all SQLite database operations."""

    def __init__(self, db_path: str, wal_mode: bool = True, auto_backup: bool = True,
                 backup_dir: Optional[str] = None):
        """Initialize database manager.

        Args:
            db_path: Path to the SQLite database file.
            wal_mode: Enable WAL mode for better concurrency.
            auto_backup: Back up database before each run.
            backup_dir: Directory for database backups.
        """
        self.db_path = Path(db_path)
        self.wal_mode = wal_mode
        self.auto_backup = auto_backup
        self.backup_dir = Path(backup_dir) if backup_dir else self.db_path.parent / "backups"
        self._connection: Optional[sqlite3.Connection] = None

    def initialize(self) -> None:
        """Initialize the database: create dirs, backup, create schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        if self.auto_backup and self.db_path.exists():
            self._backup_database()

        self._connection = self._create_connection()
        self._execute_schema()
        logger.info(f"Database initialized at {self.db_path}")

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection."""
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        if self.wal_mode:
            conn.execute("PRAGMA journal_mode = WAL")
        return conn

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        if self._connection is None:
            self._connection = self._create_connection()
        try:
            yield self._connection
        except Exception:
            self._connection.rollback()
            raise

    def _execute_schema(self) -> None:
        """Execute the database schema SQL file."""
        schema_path = Path(__file__).parent.parent / "schemas" / "database.sql"
        if not schema_path.exists():
            logger.error(f"Schema file not found: {schema_path}")
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        with open(schema_path, "r") as f:
            schema_sql = f.read()

        with self.get_connection() as conn:
            conn.executescript(schema_sql)
            conn.commit()
        logger.debug("Database schema applied")

    def _backup_database(self) -> None:
        """Create a backup of the database."""
        if not self.db_path.exists():
            return
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"analysis_backup_{timestamp}.db"
        shutil.copy2(self.db_path, backup_path)
        logger.info(f"Database backed up to {backup_path}")

        # Keep only last 5 backups
        backups = sorted(self.backup_dir.glob("analysis_backup_*.db"))
        for old_backup in backups[:-5]:
            old_backup.unlink()

    def video_exists(self, file_hash: str) -> bool:
        """Check if a video with the given hash already exists.

        Args:
            file_hash: MD5 hash of the video file.

        Returns:
            True if video already exists in database.
        """
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT id FROM videos WHERE file_hash = ?", (file_hash,)
            ).fetchone()
            return row is not None

    def insert_video(self, video_data: dict) -> int:
        """Insert a new video record.

        Args:
            video_data: Dictionary with video metadata fields.

        Returns:
            The ID of the inserted video.
        """
        columns = ", ".join(video_data.keys())
        placeholders = ", ".join(["?"] * len(video_data))
        sql = f"INSERT INTO videos ({columns}) VALUES ({placeholders})"

        with self.get_connection() as conn:
            cursor = conn.execute(sql, list(video_data.values()))
            conn.commit()
            video_id = cursor.lastrowid
            logger.debug(f"Inserted video id={video_id}: {video_data.get('filename')}")
            return video_id

    def update_video(self, video_id: int, updates: dict) -> None:
        """Update a video record.

        Args:
            video_id: ID of the video to update.
            updates: Dictionary of fields to update.
        """
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        sql = f"UPDATE videos SET {set_clause} WHERE id = ?"

        with self.get_connection() as conn:
            conn.execute(sql, list(updates.values()) + [video_id])
            conn.commit()

    def insert_faces(self, faces: list[dict]) -> None:
        """Insert face detection results.

        Args:
            faces: List of face detection dictionaries.
        """
        if not faces:
            return

        columns = ", ".join(faces[0].keys())
        placeholders = ", ".join(["?"] * len(faces[0]))
        sql = f"INSERT INTO faces ({columns}) VALUES ({placeholders})"

        with self.get_connection() as conn:
            conn.executemany(sql, [list(f.values()) for f in faces])
            conn.commit()
        logger.debug(f"Inserted {len(faces)} face records")

    def insert_transcription(self, transcription_data: dict) -> int:
        """Insert a transcription record.

        Args:
            transcription_data: Transcription data dictionary.

        Returns:
            ID of the inserted record.
        """
        columns = ", ".join(transcription_data.keys())
        placeholders = ", ".join(["?"] * len(transcription_data))
        sql = f"INSERT INTO transcriptions ({columns}) VALUES ({placeholders})"

        with self.get_connection() as conn:
            cursor = conn.execute(sql, list(transcription_data.values()))
            conn.commit()
            return cursor.lastrowid

    def insert_ocr_texts(self, ocr_records: list[dict]) -> None:
        """Insert OCR text extraction results.

        Args:
            ocr_records: List of OCR result dictionaries.
        """
        if not ocr_records:
            return

        columns = ", ".join(ocr_records[0].keys())
        placeholders = ", ".join(["?"] * len(ocr_records[0]))
        sql = f"INSERT INTO ocr_texts ({columns}) VALUES ({placeholders})"

        with self.get_connection() as conn:
            conn.executemany(sql, [list(r.values()) for r in ocr_records])
            conn.commit()
        logger.debug(f"Inserted {len(ocr_records)} OCR records")

    def insert_visual_tags(self, tags: list[dict]) -> None:
        """Insert visual tag (CLIP) results.

        Args:
            tags: List of visual tag dictionaries.
        """
        if not tags:
            return

        columns = ", ".join(tags[0].keys())
        placeholders = ", ".join(["?"] * len(tags[0]))
        sql = f"INSERT INTO visual_tags ({columns}) VALUES ({placeholders})"

        with self.get_connection() as conn:
            conn.executemany(sql, [list(t.values()) for t in tags])
            conn.commit()

    def get_all_face_embeddings(self) -> list[dict]:
        """Get all face embeddings for clustering.

        Returns:
            List of dicts with id, video_id, embedding, cluster_id.
        """
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT id, video_id, embedding, cluster_id FROM faces WHERE embedding IS NOT NULL"
            ).fetchall()
            return [dict(row) for row in rows]

    def update_face_clusters(self, face_cluster_map: dict[int, int]) -> None:
        """Update cluster_id for faces.

        Args:
            face_cluster_map: Mapping of face_id -> cluster_id.
        """
        with self.get_connection() as conn:
            for face_id, cluster_id in face_cluster_map.items():
                conn.execute(
                    "UPDATE faces SET cluster_id = ? WHERE id = ?",
                    (cluster_id, face_id),
                )
            conn.commit()
        logger.debug(f"Updated clusters for {len(face_cluster_map)} faces")

    def upsert_person_cluster(self, cluster_data: dict) -> None:
        """Insert or update a person cluster.

        Args:
            cluster_data: Cluster data dictionary.
        """
        with self.get_connection() as conn:
            existing = conn.execute(
                "SELECT cluster_id FROM person_clusters WHERE cluster_id = ?",
                (cluster_data["cluster_id"],),
            ).fetchone()

            if existing:
                set_clause = ", ".join([f"{k} = ?" for k in cluster_data.keys()])
                conn.execute(
                    f"UPDATE person_clusters SET {set_clause} WHERE cluster_id = ?",
                    list(cluster_data.values()) + [cluster_data["cluster_id"]],
                )
            else:
                columns = ", ".join(cluster_data.keys())
                placeholders = ", ".join(["?"] * len(cluster_data))
                conn.execute(
                    f"INSERT INTO person_clusters ({columns}) VALUES ({placeholders})",
                    list(cluster_data.values()),
                )
            conn.commit()

    def upsert_video_person(self, video_id: int, cluster_id: int,
                            appearance_count: int, avg_confidence: float) -> None:
        """Insert or update a video-person relationship.

        Args:
            video_id: Video ID.
            cluster_id: Person cluster ID.
            appearance_count: Number of appearances.
            avg_confidence: Average detection confidence.
        """
        with self.get_connection() as conn:
            conn.execute(
                """INSERT INTO video_persons (video_id, cluster_id, appearance_count, avg_confidence)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(video_id, cluster_id) DO UPDATE SET
                   appearance_count = excluded.appearance_count,
                   avg_confidence = excluded.avg_confidence""",
                (video_id, cluster_id, appearance_count, avg_confidence),
            )
            conn.commit()

    def insert_processing_log(self, log_data: dict) -> None:
        """Insert a processing log entry.

        Args:
            log_data: Processing log data dictionary.
        """
        # Serialize JSON fields
        for key in ("themes_distribution", "years_distribution", "errors"):
            if key in log_data and not isinstance(log_data[key], str):
                log_data[key] = json.dumps(log_data[key], ensure_ascii=False)

        columns = ", ".join(log_data.keys())
        placeholders = ", ".join(["?"] * len(log_data))
        sql = f"INSERT OR REPLACE INTO processing_log ({columns}) VALUES ({placeholders})"

        with self.get_connection() as conn:
            conn.execute(sql, list(log_data.values()))
            conn.commit()

    def get_next_batch_number(self) -> int:
        """Get the next batch number.

        Returns:
            Next sequential batch number.
        """
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT MAX(batch_number) as max_num FROM processing_log"
            ).fetchone()
            return (row["max_num"] or 0) + 1

    def get_video_count_by_theme(self, batch_id: Optional[str] = None) -> dict[str, int]:
        """Get count of videos per theme.

        Args:
            batch_id: Optional batch filter.

        Returns:
            Dictionary mapping theme names to counts.
        """
        with self.get_connection() as conn:
            if batch_id:
                rows = conn.execute(
                    "SELECT primary_theme, COUNT(*) as cnt FROM videos WHERE batch_id = ? GROUP BY primary_theme",
                    (batch_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT primary_theme, COUNT(*) as cnt FROM videos GROUP BY primary_theme"
                ).fetchall()
            return {row["primary_theme"]: row["cnt"] for row in rows}

    def get_video_count_by_year(self, batch_id: Optional[str] = None) -> dict[str, int]:
        """Get count of videos per year.

        Args:
            batch_id: Optional batch filter.

        Returns:
            Dictionary mapping year to count.
        """
        with self.get_connection() as conn:
            if batch_id:
                rows = conn.execute(
                    "SELECT year, COUNT(*) as cnt FROM videos WHERE batch_id = ? GROUP BY year",
                    (batch_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT year, COUNT(*) as cnt FROM videos GROUP BY year"
                ).fetchall()
            return {str(row["year"]): row["cnt"] for row in rows}

    def get_batch_videos(self, batch_id: str) -> list[dict]:
        """Get all videos from a batch.

        Args:
            batch_id: Batch identifier.

        Returns:
            List of video record dictionaries.
        """
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM videos WHERE batch_id = ?", (batch_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.debug("Database connection closed")

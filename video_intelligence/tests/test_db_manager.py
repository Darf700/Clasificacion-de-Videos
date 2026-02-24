"""Tests for modules.db_manager - DatabaseManager."""

import os
import numpy as np
import pytest

from modules.db_manager import DatabaseManager, _validate_columns


# =========================================================================
# Column validation (SQL injection prevention)
# =========================================================================

class TestValidateColumns:
    def test_valid_columns_accepted(self):
        _validate_columns(["filename", "file_hash", "original_path"], "videos")

    def test_invalid_identifier_rejected(self):
        with pytest.raises(ValueError, match="Invalid column name"):
            _validate_columns(["valid; DROP TABLE videos--"], "videos")

    def test_unknown_column_rejected(self):
        with pytest.raises(ValueError, match="Unknown column"):
            _validate_columns(["nonexistent_column"], "videos")

    def test_sql_injection_via_column_name(self):
        with pytest.raises(ValueError):
            _validate_columns(["id; DROP TABLE videos"], "videos")

    def test_empty_column_name_rejected(self):
        with pytest.raises(ValueError):
            _validate_columns([""], "videos")

    def test_column_starting_with_number_rejected(self):
        with pytest.raises(ValueError):
            _validate_columns(["1column"], "videos")

    def test_all_tables_have_whitelists(self):
        """Every table used in the app should have a column whitelist."""
        from modules.db_manager import _VALID_COLUMNS
        expected_tables = {
            "videos", "faces", "person_clusters", "transcriptions",
            "ocr_texts", "visual_tags", "processing_log",
        }
        assert expected_tables.issubset(set(_VALID_COLUMNS.keys()))


# =========================================================================
# DatabaseManager CRUD
# =========================================================================

class TestDatabaseManager:
    @pytest.fixture
    def db(self, tmp_dir, schema_path):
        db_path = os.path.join(tmp_dir, "test.db")
        return DatabaseManager(db_path, schema_path)

    def test_init_creates_database_file(self, db, tmp_dir):
        assert os.path.exists(os.path.join(tmp_dir, "test.db"))

    def test_insert_and_retrieve_video(self, db, sample_video_data):
        vid = db.insert_video(sample_video_data)
        assert vid > 0
        row = db.get_video_by_id(vid)
        assert row is not None
        assert row["filename"] == "test_video.mp4"
        assert row["file_hash"] == "abc123def456"

    def test_video_exists_by_hash(self, db, sample_video_data):
        assert not db.video_exists("abc123def456")
        db.insert_video(sample_video_data)
        assert db.video_exists("abc123def456")

    def test_get_video_by_hash(self, db, sample_video_data):
        db.insert_video(sample_video_data)
        row = db.get_video_by_hash("abc123def456")
        assert row is not None
        assert row["filename"] == "test_video.mp4"

    def test_get_video_by_hash_not_found(self, db):
        assert db.get_video_by_hash("nonexistent") is None

    def test_update_video(self, db, sample_video_data):
        vid = db.insert_video(sample_video_data)
        db.update_video(vid, {"primary_theme": "Comedia", "theme_confidence": 0.85})
        row = db.get_video_by_id(vid)
        assert row["primary_theme"] == "Comedia"
        assert row["theme_confidence"] == 0.85

    def test_update_video_rejects_invalid_column(self, db, sample_video_data):
        vid = db.insert_video(sample_video_data)
        with pytest.raises(ValueError, match="Unknown column"):
            db.update_video(vid, {"hacked_column": "oops"})

    def test_duplicate_hash_raises(self, db, sample_video_data):
        db.insert_video(sample_video_data)
        with pytest.raises(Exception):
            db.insert_video(sample_video_data)

    def test_total_video_count(self, db, sample_video_data):
        assert db.get_total_video_count() == 0
        db.insert_video(sample_video_data)
        assert db.get_total_video_count() == 1

    # --- Face operations ---

    def test_insert_and_query_faces(self, db, sample_video_data):
        vid = db.insert_video(sample_video_data)
        embedding = np.random.randn(512).astype(np.float32).tobytes()
        faces = [
            {
                "video_id": vid,
                "frame_number": 10,
                "timestamp_seconds": 1.5,
                "bbox_x": 100, "bbox_y": 200,
                "bbox_width": 50, "bbox_height": 60,
                "confidence": 0.95,
                "face_size": "medium",
                "is_frontal": True,
                "embedding": embedding,
            },
        ]
        ids = db.insert_faces(faces)
        assert len(ids) == 1

        result = db.get_faces_by_video(vid)
        assert len(result) == 1
        assert result[0]["confidence"] == 0.95

    def test_insert_faces_empty_list(self, db):
        assert db.insert_faces([]) == []

    def test_get_all_face_embeddings(self, db, sample_video_data):
        vid = db.insert_video(sample_video_data)
        emb = np.random.randn(512).astype(np.float32).tobytes()
        db.insert_faces([{
            "video_id": vid, "frame_number": 0, "timestamp_seconds": 0.0,
            "bbox_x": 0, "bbox_y": 0, "bbox_width": 50, "bbox_height": 50,
            "confidence": 0.9, "face_size": "medium", "is_frontal": True,
            "embedding": emb,
        }])
        embeddings = db.get_all_face_embeddings()
        assert len(embeddings) == 1
        assert isinstance(embeddings[0], tuple)
        assert len(embeddings[0]) == 2

    def test_update_face_clusters(self, db, sample_video_data):
        vid = db.insert_video(sample_video_data)
        emb = np.random.randn(512).astype(np.float32).tobytes()
        ids = db.insert_faces([{
            "video_id": vid, "frame_number": 0, "timestamp_seconds": 0.0,
            "bbox_x": 0, "bbox_y": 0, "bbox_width": 50, "bbox_height": 50,
            "confidence": 0.9, "face_size": "medium", "is_frontal": True,
            "embedding": emb,
        }])
        db.update_face_clusters({ids[0]: 42})
        faces = db.get_faces_by_video(vid)
        assert faces[0]["cluster_id"] == 42

    def test_update_face_clusters_empty(self, db):
        db.update_face_clusters({})  # should not raise

    # --- Transcription operations ---

    def test_insert_transcription(self, db, sample_video_data):
        vid = db.insert_video(sample_video_data)
        tid = db.insert_transcription({
            "video_id": vid,
            "full_text": "Hola mundo",
            "language": "es",
            "confidence": 0.92,
            "word_count": 2,
            "processing_time_seconds": 3.5,
        })
        assert tid > 0

    # --- OCR operations ---

    def test_insert_ocr_texts(self, db, sample_video_data):
        vid = db.insert_video(sample_video_data)
        ids = db.insert_ocr_texts([{
            "video_id": vid,
            "frame_number": 5,
            "timestamp_seconds": 2.0,
            "text_content": "Subtitulo de prueba",
            "language": "es",
            "confidence": 0.88,
            "bbox_x": 10, "bbox_y": 400,
            "bbox_width": 300, "bbox_height": 30,
        }])
        assert len(ids) == 1

    def test_insert_ocr_texts_empty(self, db):
        assert db.insert_ocr_texts([]) == []

    # --- Visual tags operations ---

    def test_insert_visual_tags(self, db, sample_video_data):
        vid = db.insert_video(sample_video_data)
        ids = db.insert_visual_tags([
            {"video_id": vid, "tag": "Comedia", "confidence": 0.85, "source": "clip_auto"},
            {"video_id": vid, "tag": "Musica", "confidence": 0.70, "source": "clip_auto"},
        ])
        assert len(ids) == 2

    def test_insert_visual_tags_empty(self, db):
        assert db.insert_visual_tags([]) == []

    # --- Batch log operations ---

    def test_create_and_update_batch_log(self, db):
        batch_id = db.create_batch_log(50)
        assert isinstance(batch_id, str)
        assert len(batch_id) == 8

        db.update_batch_log(batch_id, {"status": "completed", "videos_processed": 48})

    def test_increment_batch_counter(self, db):
        batch_id = db.create_batch_log(10)
        db.increment_batch_counter(batch_id, "videos_processed")
        db.increment_batch_counter(batch_id, "videos_processed")
        db.increment_batch_counter(batch_id, "videos_failed")

    def test_increment_batch_counter_invalid_column(self, db):
        batch_id = db.create_batch_log(10)
        with pytest.raises(ValueError, match="Invalid counter"):
            db.increment_batch_counter(batch_id, "hacked_column")

    def test_batch_number_auto_increments(self, db):
        b1 = db.create_batch_log(5)
        b2 = db.create_batch_log(10)
        assert b1 != b2

    # --- Person cluster operations ---

    def test_insert_person_cluster(self, db, sample_video_data):
        vid = db.insert_video(sample_video_data)
        cid = db.insert_person_cluster({
            "face_count": 5,
            "video_count": 2,
            "avg_confidence": 0.90,
            "first_seen_video_id": vid,
            "last_seen_video_id": vid,
        })
        assert cid > 0

    def test_upsert_video_person(self, db, sample_video_data):
        vid = db.insert_video(sample_video_data)
        cid = db.insert_person_cluster({
            "face_count": 5, "video_count": 1, "avg_confidence": 0.9,
        })
        db.upsert_video_person(vid, cid, 3, 30, 0.92)
        # Upsert again (should update, not fail)
        db.upsert_video_person(vid, cid, 5, 30, 0.95)

    # --- FTS5 ---

    def test_fts_transcription_search(self, db, sample_video_data):
        vid = db.insert_video(sample_video_data)
        db.insert_transcription({
            "video_id": vid,
            "full_text": "El gato negro corre por el parque",
            "language": "es",
            "confidence": 0.9,
            "word_count": 7,
            "processing_time_seconds": 2.0,
        })
        results = db.search_transcriptions("gato")
        assert len(results) == 1
        assert "gato" in results[0]["full_text"]

    def test_fts_ocr_search(self, db, sample_video_data):
        vid = db.insert_video(sample_video_data)
        db.insert_ocr_texts([{
            "video_id": vid, "frame_number": 0, "timestamp_seconds": 0.0,
            "text_content": "SUBSCRIBE NOW", "language": "en",
            "confidence": 0.95,
            "bbox_x": 0, "bbox_y": 0, "bbox_width": 100, "bbox_height": 20,
        }])
        results = db.search_ocr_texts("SUBSCRIBE")
        assert len(results) == 1

    # --- Statistics ---

    def test_get_batch_stats(self, db, sample_video_data):
        batch_id = db.create_batch_log(1)
        sample_video_data["batch_id"] = batch_id
        db.insert_video(sample_video_data)
        stats = db.get_batch_stats(batch_id)
        assert "theme_distribution" in stats
        assert "date_distribution" in stats
        assert "unique_persons" in stats

    # --- Views ---

    def test_get_videos_full(self, db, sample_video_data):
        db.insert_video(sample_video_data)
        rows = db.get_videos_full()
        assert len(rows) == 1

    def test_get_recent_batches(self, db):
        db.create_batch_log(5)
        batches = db.get_recent_batches()
        assert len(batches) == 1

    # --- Export ---

    def test_get_all_videos_for_export(self, db, sample_video_data):
        db.insert_video(sample_video_data)
        rows = db.get_all_videos_for_export()
        assert len(rows) == 1
        assert rows[0]["filename"] == "test_video.mp4"

    def test_get_all_videos_for_export_with_batch_filter(self, db, sample_video_data):
        batch_id = db.create_batch_log(1)
        sample_video_data["batch_id"] = batch_id
        db.insert_video(sample_video_data)
        rows = db.get_all_videos_for_export(batch_id)
        assert len(rows) == 1
        rows_empty = db.get_all_videos_for_export("nonexistent")
        assert len(rows_empty) == 0

-- Video Intelligence System - Database Schema
-- SQLite 3

CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    original_path TEXT NOT NULL,
    final_path TEXT,
    file_hash TEXT UNIQUE NOT NULL,
    file_size_bytes INTEGER,
    duration_seconds REAL,
    width INTEGER,
    height INTEGER,
    aspect_ratio TEXT,
    fps REAL,
    bitrate INTEGER,
    codec_video TEXT,
    codec_audio TEXT,
    format_type TEXT CHECK(format_type IN ('reel', 'short', 'long')),
    orientation TEXT CHECK(orientation IN ('vertical', 'horizontal', 'square')),
    has_audio BOOLEAN,
    has_dialogue BOOLEAN,
    audio_language TEXT,
    has_text_overlay BOOLEAN,
    creation_date DATETIME,
    date_source TEXT CHECK(date_source IN ('exif', 'filename', 'file_modified', 'unknown')),
    year INTEGER,
    month INTEGER,
    month_name TEXT,
    primary_theme TEXT,
    theme_confidence REAL,
    secondary_themes TEXT,
    face_count INTEGER DEFAULT 0,
    unique_persons INTEGER DEFAULT 0,
    processed_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    processing_duration_seconds REAL,
    batch_id TEXT,
    error_message TEXT,
    needs_review BOOLEAN DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_videos_hash ON videos(file_hash);
CREATE INDEX IF NOT EXISTS idx_videos_theme ON videos(primary_theme);
CREATE INDEX IF NOT EXISTS idx_videos_date ON videos(year, month);
CREATE INDEX IF NOT EXISTS idx_videos_processed ON videos(processed_date);
CREATE INDEX IF NOT EXISTS idx_videos_batch ON videos(batch_id);

CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL,
    frame_number INTEGER,
    timestamp_seconds REAL,
    bbox_x INTEGER,
    bbox_y INTEGER,
    bbox_width INTEGER,
    bbox_height INTEGER,
    confidence REAL,
    face_size TEXT CHECK(face_size IN ('small', 'medium', 'large')),
    is_frontal BOOLEAN,
    embedding BLOB,
    cluster_id INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_faces_video ON faces(video_id);
CREATE INDEX IF NOT EXISTS idx_faces_cluster ON faces(cluster_id);

CREATE TABLE IF NOT EXISTS person_clusters (
    cluster_id INTEGER PRIMARY KEY AUTOINCREMENT,
    face_count INTEGER DEFAULT 0,
    video_count INTEGER DEFAULT 0,
    representative_face_id INTEGER,
    avg_confidence REAL,
    label TEXT,
    first_seen_video_id INTEGER,
    last_seen_video_id INTEGER,
    first_seen_date DATETIME,
    last_seen_date DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (representative_face_id) REFERENCES faces(id)
);

CREATE TABLE IF NOT EXISTS video_persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL,
    cluster_id INTEGER NOT NULL,
    appearance_count INTEGER DEFAULT 0,
    total_frames INTEGER DEFAULT 0,
    avg_confidence REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE,
    FOREIGN KEY (cluster_id) REFERENCES person_clusters(cluster_id),
    UNIQUE(video_id, cluster_id)
);

CREATE INDEX IF NOT EXISTS idx_video_persons_video ON video_persons(video_id);
CREATE INDEX IF NOT EXISTS idx_video_persons_cluster ON video_persons(cluster_id);

CREATE TABLE IF NOT EXISTS transcriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL,
    full_text TEXT,
    language TEXT,
    confidence REAL,
    word_count INTEGER,
    processing_time_seconds REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_transcriptions_video ON transcriptions(video_id);

CREATE TABLE IF NOT EXISTS ocr_texts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL,
    frame_number INTEGER,
    timestamp_seconds REAL,
    text_content TEXT,
    language TEXT,
    confidence REAL,
    bbox_x INTEGER,
    bbox_y INTEGER,
    bbox_width INTEGER,
    bbox_height INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ocr_video ON ocr_texts(video_id);

CREATE TABLE IF NOT EXISTS visual_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL,
    tag TEXT NOT NULL,
    confidence REAL,
    source TEXT DEFAULT 'clip_auto',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_visual_tags_video ON visual_tags(video_id);
CREATE INDEX IF NOT EXISTS idx_visual_tags_tag ON visual_tags(tag);

CREATE TABLE IF NOT EXISTS processing_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT UNIQUE NOT NULL,
    batch_number INTEGER,
    videos_found INTEGER,
    videos_new INTEGER,
    videos_processed INTEGER,
    videos_failed INTEGER,
    videos_skipped INTEGER,
    processing_start DATETIME,
    processing_end DATETIME,
    duration_seconds REAL,
    themes_distribution TEXT,
    years_distribution TEXT,
    errors TEXT,
    report_path TEXT,
    export_path TEXT,
    status TEXT CHECK(status IN ('running', 'completed', 'failed', 'interrupted')),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_processing_log_batch ON processing_log(batch_id);
CREATE INDEX IF NOT EXISTS idx_processing_log_date ON processing_log(processing_start);

-- Triggers
CREATE TRIGGER IF NOT EXISTS update_videos_timestamp
AFTER UPDATE ON videos
BEGIN
    UPDATE videos SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_person_clusters_timestamp
AFTER UPDATE ON person_clusters
BEGIN
    UPDATE person_clusters SET updated_at = CURRENT_TIMESTAMP WHERE cluster_id = NEW.cluster_id;
END;

-- Views
CREATE VIEW IF NOT EXISTS videos_full AS
SELECT
    v.*,
    t.full_text as transcription,
    t.language as transcription_language,
    COUNT(DISTINCT f.cluster_id) as unique_persons_count,
    COUNT(DISTINCT vt.tag) as tag_count
FROM videos v
LEFT JOIN transcriptions t ON v.id = t.video_id
LEFT JOIN faces f ON v.id = f.video_id
LEFT JOIN visual_tags vt ON v.id = vt.video_id
GROUP BY v.id;

CREATE VIEW IF NOT EXISTS recent_batches AS
SELECT
    batch_id,
    batch_number,
    videos_processed,
    duration_seconds,
    processing_start,
    status
FROM processing_log
ORDER BY processing_start DESC
LIMIT 20;

-- FTS5 for full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS transcriptions_fts USING fts5(
    video_id UNINDEXED,
    full_text,
    content=transcriptions,
    content_rowid=id
);

CREATE TRIGGER IF NOT EXISTS transcriptions_fts_insert AFTER INSERT ON transcriptions
BEGIN
    INSERT INTO transcriptions_fts(rowid, video_id, full_text)
    VALUES (NEW.id, NEW.video_id, NEW.full_text);
END;

CREATE TRIGGER IF NOT EXISTS transcriptions_fts_delete AFTER DELETE ON transcriptions
BEGIN
    DELETE FROM transcriptions_fts WHERE rowid = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS transcriptions_fts_update AFTER UPDATE ON transcriptions
BEGIN
    DELETE FROM transcriptions_fts WHERE rowid = OLD.id;
    INSERT INTO transcriptions_fts(rowid, video_id, full_text)
    VALUES (NEW.id, NEW.video_id, NEW.full_text);
END;

CREATE VIRTUAL TABLE IF NOT EXISTS ocr_texts_fts USING fts5(
    video_id UNINDEXED,
    text_content,
    content=ocr_texts,
    content_rowid=id
);

CREATE TRIGGER IF NOT EXISTS ocr_texts_fts_insert AFTER INSERT ON ocr_texts
BEGIN
    INSERT INTO ocr_texts_fts(rowid, video_id, text_content)
    VALUES (NEW.id, NEW.video_id, NEW.text_content);
END;

CREATE TRIGGER IF NOT EXISTS ocr_texts_fts_delete AFTER DELETE ON ocr_texts
BEGIN
    DELETE FROM ocr_texts_fts WHERE rowid = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS ocr_texts_fts_update AFTER UPDATE ON ocr_texts
BEGIN
    DELETE FROM ocr_texts_fts WHERE rowid = OLD.id;
    INSERT INTO ocr_texts_fts(rowid, video_id, text_content)
    VALUES (NEW.id, NEW.video_id, NEW.text_content);
END;

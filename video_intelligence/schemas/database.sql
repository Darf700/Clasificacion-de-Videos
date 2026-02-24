-- Video Intelligence System - Database Schema
-- SQLite 3

-- ============================================================================
-- VIDEOS - Main table
-- ============================================================================
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- File information
    filename TEXT NOT NULL,
    original_path TEXT NOT NULL,
    final_path TEXT,
    file_hash TEXT UNIQUE NOT NULL,
    file_size_bytes INTEGER,

    -- Technical metadata
    duration_seconds REAL,
    width INTEGER,
    height INTEGER,
    aspect_ratio TEXT,
    fps REAL,
    bitrate INTEGER,
    codec_video TEXT,
    codec_audio TEXT,

    -- Format classification
    format_type TEXT CHECK(format_type IN ('reel', 'short', 'long')),
    orientation TEXT CHECK(orientation IN ('vertical', 'horizontal', 'square')),

    -- Audio information
    has_audio BOOLEAN,
    has_dialogue BOOLEAN,
    audio_language TEXT,

    -- Visual information
    has_text_overlay BOOLEAN,

    -- Date information
    creation_date DATETIME,
    date_source TEXT CHECK(date_source IN ('exif', 'filename', 'file_modified', 'unknown')),
    year INTEGER,
    month INTEGER,
    month_name TEXT,

    -- Classification
    primary_theme TEXT,
    theme_confidence REAL,
    secondary_themes TEXT,  -- JSON array

    -- Face information
    face_count INTEGER DEFAULT 0,
    unique_persons INTEGER DEFAULT 0,

    -- Processing metadata
    processed_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    processing_duration_seconds REAL,
    batch_id TEXT,
    error_message TEXT,
    needs_review BOOLEAN DEFAULT 0,

    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for videos
CREATE INDEX IF NOT EXISTS idx_videos_hash ON videos(file_hash);
CREATE INDEX IF NOT EXISTS idx_videos_theme ON videos(primary_theme);
CREATE INDEX IF NOT EXISTS idx_videos_date ON videos(year, month);
CREATE INDEX IF NOT EXISTS idx_videos_processed ON videos(processed_date);
CREATE INDEX IF NOT EXISTS idx_videos_batch ON videos(batch_id);

-- ============================================================================
-- FACES - Face detection results
-- ============================================================================
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL,

    -- Frame information
    frame_number INTEGER,
    timestamp_seconds REAL,

    -- Bounding box
    bbox_x INTEGER,
    bbox_y INTEGER,
    bbox_width INTEGER,
    bbox_height INTEGER,

    -- Detection metadata
    confidence REAL,
    face_size TEXT CHECK(face_size IN ('small', 'medium', 'large')),
    is_frontal BOOLEAN,

    -- Embedding (stored as BLOB)
    embedding BLOB,

    -- Clustering
    cluster_id INTEGER,

    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
);

-- Indexes for faces
CREATE INDEX IF NOT EXISTS idx_faces_video ON faces(video_id);
CREATE INDEX IF NOT EXISTS idx_faces_cluster ON faces(cluster_id);

-- ============================================================================
-- PERSON_CLUSTERS - Face clustering results
-- ============================================================================
CREATE TABLE IF NOT EXISTS person_clusters (
    cluster_id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Statistics
    face_count INTEGER DEFAULT 0,
    video_count INTEGER DEFAULT 0,

    -- Representative face (for thumbnail)
    representative_face_id INTEGER,

    -- Confidence
    avg_confidence REAL,

    -- User-assigned label
    label TEXT,

    -- First/last appearance
    first_seen_video_id INTEGER,
    last_seen_video_id INTEGER,
    first_seen_date DATETIME,
    last_seen_date DATETIME,

    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (representative_face_id) REFERENCES faces(id)
);

-- ============================================================================
-- VIDEO_PERSONS - Many-to-many relationship
-- ============================================================================
CREATE TABLE IF NOT EXISTS video_persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL,
    cluster_id INTEGER NOT NULL,

    -- Statistics for this video
    appearance_count INTEGER DEFAULT 0,  -- How many times in this video
    total_frames INTEGER DEFAULT 0,
    avg_confidence REAL,

    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE,
    FOREIGN KEY (cluster_id) REFERENCES person_clusters(cluster_id),

    UNIQUE(video_id, cluster_id)
);

-- Indexes for video_persons
CREATE INDEX IF NOT EXISTS idx_video_persons_video ON video_persons(video_id);
CREATE INDEX IF NOT EXISTS idx_video_persons_cluster ON video_persons(cluster_id);

-- ============================================================================
-- TRANSCRIPTIONS - Whisper audio transcriptions
-- ============================================================================
CREATE TABLE IF NOT EXISTS transcriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL,

    -- Transcription
    full_text TEXT,
    language TEXT,

    -- Metadata
    confidence REAL,
    word_count INTEGER,
    processing_time_seconds REAL,

    -- Timestamps (segments stored separately if needed)
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
);

-- Index for transcriptions
CREATE INDEX IF NOT EXISTS idx_transcriptions_video ON transcriptions(video_id);
CREATE INDEX IF NOT EXISTS idx_transcriptions_text ON transcriptions(full_text);

-- ============================================================================
-- OCR_TEXTS - Text extracted from video frames
-- ============================================================================
CREATE TABLE IF NOT EXISTS ocr_texts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL,

    -- Frame information
    frame_number INTEGER,
    timestamp_seconds REAL,

    -- OCR result
    text_content TEXT,
    language TEXT,
    confidence REAL,

    -- Bounding box
    bbox_x INTEGER,
    bbox_y INTEGER,
    bbox_width INTEGER,
    bbox_height INTEGER,

    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
);

-- Indexes for ocr_texts
CREATE INDEX IF NOT EXISTS idx_ocr_video ON ocr_texts(video_id);
CREATE INDEX IF NOT EXISTS idx_ocr_text ON ocr_texts(text_content);

-- ============================================================================
-- VISUAL_TAGS - CLIP categorization results
-- ============================================================================
CREATE TABLE IF NOT EXISTS visual_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL,

    -- Tag information
    tag TEXT NOT NULL,
    confidence REAL,
    source TEXT DEFAULT 'clip_auto',  -- clip_auto, user_manual, etc.

    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
);

-- Indexes for visual_tags
CREATE INDEX IF NOT EXISTS idx_visual_tags_video ON visual_tags(video_id);
CREATE INDEX IF NOT EXISTS idx_visual_tags_tag ON visual_tags(tag);

-- ============================================================================
-- PROCESSING_LOG - Batch processing history
-- ============================================================================
CREATE TABLE IF NOT EXISTS processing_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Batch information
    batch_id TEXT UNIQUE NOT NULL,
    batch_number INTEGER,

    -- Statistics
    videos_found INTEGER,
    videos_new INTEGER,
    videos_processed INTEGER,
    videos_failed INTEGER,
    videos_skipped INTEGER,

    -- Performance
    processing_start DATETIME,
    processing_end DATETIME,
    duration_seconds REAL,

    -- Results
    themes_distribution TEXT,  -- JSON object
    years_distribution TEXT,   -- JSON object
    errors TEXT,               -- JSON array of errors

    -- Paths
    report_path TEXT,
    export_path TEXT,

    -- Status
    status TEXT CHECK(status IN ('running', 'completed', 'failed', 'interrupted')),

    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Index for processing_log
CREATE INDEX IF NOT EXISTS idx_processing_log_batch ON processing_log(batch_id);
CREATE INDEX IF NOT EXISTS idx_processing_log_date ON processing_log(processing_start);

-- ============================================================================
-- TRIGGERS - Auto-update timestamps
-- ============================================================================
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

-- ============================================================================
-- VIEWS - Convenient queries
-- ============================================================================

-- Videos with full information
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

-- Person clusters with statistics
CREATE VIEW IF NOT EXISTS person_clusters_stats AS
SELECT
    pc.*,
    COUNT(DISTINCT vp.video_id) as actual_video_count,
    COUNT(f.id) as actual_face_count
FROM person_clusters pc
LEFT JOIN video_persons vp ON pc.cluster_id = vp.cluster_id
LEFT JOIN faces f ON pc.cluster_id = f.cluster_id
GROUP BY pc.cluster_id;

-- Recent processing batches
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

-- ============================================================================
-- INDEXES FOR FULL-TEXT SEARCH (FTS5)
-- ============================================================================

-- Full-text search on transcriptions
CREATE VIRTUAL TABLE IF NOT EXISTS transcriptions_fts USING fts5(
    video_id UNINDEXED,
    full_text,
    content=transcriptions,
    content_rowid=id
);

-- Triggers to keep FTS in sync
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

-- Full-text search on OCR texts
CREATE VIRTUAL TABLE IF NOT EXISTS ocr_texts_fts USING fts5(
    video_id UNINDEXED,
    text_content,
    content=ocr_texts,
    content_rowid=id
);

-- Triggers for OCR FTS
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

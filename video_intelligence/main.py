#!/usr/bin/env python3
"""Video Intelligence System - Main Orchestrator.

Processes video files through a multi-stage AI analysis pipeline:
1. Discovery & Metadata extraction
2. Frame sampling
3. Face detection & clustering
4. CLIP visual categorization
5. Whisper audio transcription
6. OCR text extraction
7. Date extraction
8. Theme classification
9. File organization
10. Reporting

Usage:
    python main.py                    # Process all videos in input folder
    python main.py --config my.yaml   # Use custom config
    python main.py --dry-run          # Preview without moving files
    python main.py --report-only ID   # Generate report from existing data
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from modules.clip_analyzer import CLIPAnalyzer
from modules.date_extractor import DateExtractor
from modules.db_manager import DatabaseManager
from modules.face_clusterer import FaceClusterer
from modules.face_detector import FaceDetector
from modules.file_organizer import FileOrganizer
from modules.frame_sampler import FrameSampler
from modules.metadata_extractor import MetadataExtractor
from modules.ocr_processor import OCRProcessor
from modules.theme_classifier import ThemeClassifier
from modules.whisper_transcriber import WhisperTranscriber
from utils.hash_utils import calculate_md5
from utils.logging_utils import get_logger, setup_logging
from utils.video_utils import discover_videos, format_duration, format_file_size


class VideoIntelligence:
    """Main orchestrator for the video analysis pipeline.

    Coordinates all processing modules to analyze, classify, and
    organize video files. Reads all settings from the nested config.yaml.

    Args:
        config_path: Path to the YAML configuration file.
        dry_run: If True, preview organization without moving files.
    """

    def __init__(self, config_path: str, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self.config = self._load_config(config_path)

        # Resolve paths (database/log paths may be relative to analysis dir)
        self.analysis_dir = self.config["paths"]["analysis"]

        # Setup logging
        log_cfg = self.config["logging"]
        log_dir = log_cfg.get("log_dir", "_ANALYSIS/logs")
        if not os.path.isabs(log_dir):
            log_dir = os.path.join(self.analysis_dir, os.path.basename(log_dir))

        self.logger = setup_logging(
            log_dir=log_dir,
            level=log_cfg.get("level", "INFO"),
            log_to_file=log_cfg.get("file_logging", True),
            max_size_mb=log_cfg.get("max_bytes", 10485760) // (1024 * 1024),
        )
        self.log = get_logger("orchestrator")

        # Initialize database
        db_cfg = self.config["database"]
        db_path = db_cfg.get("path", "_ANALYSIS/database/analysis.db")
        if not os.path.isabs(db_path):
            db_path = os.path.join(self.analysis_dir, os.path.relpath(db_path, "_ANALYSIS"))
        schema_path = os.path.join(PROJECT_ROOT, "schemas", "database.sql")

        # Auto-backup database before run
        if db_cfg.get("auto_backup", False) and os.path.exists(db_path):
            self._backup_database(db_path, db_cfg)

        self.db = DatabaseManager(db_path, schema_path)

        # Initialize modules
        self._init_modules()

        # Processing stats
        self.stats = {
            "total": 0,
            "processed": 0,
            "skipped": 0,
            "duplicates_moved": 0,
            "errors": 0,
            "error_details": [],
            "start_time": None,
        }

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate configuration from YAML file.

        Args:
            config_path: Path to config.yaml.

        Returns:
            Configuration dictionary.
        """
        if not os.path.exists(config_path):
            print(f"ERROR: Config file not found: {config_path}")
            sys.exit(1)

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return config

    def _backup_database(self, db_path: str, db_cfg: Dict[str, Any]) -> None:
        """Create a backup of the database before processing.

        Args:
            db_path: Path to the current database file.
            db_cfg: Database config section.
        """
        backup_dir = db_cfg.get("backup_dir", "_ANALYSIS/database/backups")
        if not os.path.isabs(backup_dir):
            backup_dir = os.path.join(self.analysis_dir, os.path.relpath(backup_dir, "_ANALYSIS"))
        os.makedirs(backup_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"analysis_backup_{timestamp}.db")
        shutil.copy2(db_path, backup_path)
        get_logger("orchestrator").info("Database backup: %s", backup_path)

    def _init_modules(self) -> None:
        """Initialize all processing modules from nested config."""
        cfg = self.config
        models = cfg["models"]
        analysis = cfg["analysis"]
        org = cfg["organization"]
        thumbs = cfg.get("thumbnails", {})
        perf = cfg.get("performance", {})

        # Thumbnail directories
        video_thumb_dir = os.path.join(self.analysis_dir, "thumbnails", "videos")
        face_thumb_dir = os.path.join(self.analysis_dir, "thumbnails", "faces")

        # Core modules (always loaded)
        self.metadata_extractor = MetadataExtractor()
        self.date_extractor = DateExtractor(
            month_names=org["month_names"],
        )
        self.frame_sampler = FrameSampler(
            num_frames=analysis["frames"]["count_per_video"],
            thumbnail_dir=video_thumb_dir if thumbs.get("generate_video_thumbs", True) else None,
        )
        self.file_organizer = FileOrganizer(
            output_base=cfg["paths"]["output"],
            operation=org["operation"],
            no_date_folder=org["no_date_folder"],
            no_theme_folder=analysis["themes"]["fallback_theme"],
            month_names=org["month_names"],
        )
        self.theme_classifier = ThemeClassifier(
            confidence_threshold=analysis["themes"]["confidence_threshold"],
            fallback_theme=analysis["themes"]["fallback_theme"],
        )

        # AI modules (lazy loaded on first use)
        self.clip_analyzer = CLIPAnalyzer(
            model_name=models["clip"]["model_name"],
            device=models["clip"].get("device"),
            batch_size=perf.get("batch", {}).get("frame_batch_size", cfg["processing"]["gpu_batch_size"]),
        )
        self.face_detector = FaceDetector(
            detection_threshold=models["face"]["detection_threshold"],
            device=models["face"].get("device"),
        )
        self.face_clusterer = FaceClusterer(
            eps=analysis["face_clustering"]["eps"],
            min_samples=analysis["face_clustering"]["min_samples"],
            thumbnail_dir=face_thumb_dir if thumbs.get("generate_face_thumbs", True) else None,
        )
        self.whisper_transcriber = WhisperTranscriber(
            model_name=models["whisper"]["model_name"],
            device=models["whisper"].get("device"),
            audio_sample_duration=analysis["audio"]["sample_duration"],
        )
        self.ocr_processor = OCRProcessor(
            languages=models["ocr"].get("languages", ["es", "en"]),
            confidence_threshold=models["ocr"]["confidence_threshold"],
            max_frames=analysis["text_extraction"]["max_frames"],
        )

    def run(self) -> None:
        """Execute the full processing pipeline."""
        self.stats["start_time"] = time.time()
        self.log.info("=" * 60)
        self.log.info("Video Intelligence System - Starting")
        self.log.info("=" * 60)

        if self.dry_run:
            self.log.info("DRY RUN mode - no files will be moved")

        # Ensure directories exist
        self._ensure_directories()

        # Step 1: Discover videos
        input_dir = self.config["paths"]["input"]
        extensions = self.config["processing"].get("video_extensions")
        video_paths = discover_videos(input_dir, extensions)

        if not video_paths:
            self.log.info("No videos found in %s", input_dir)
            return

        self.stats["total"] = len(video_paths)
        self.log.info("Found %d video(s) to process", len(video_paths))

        # Create batch log
        batch_id = self.db.create_batch_log(len(video_paths))

        # Error handling config
        error_cfg = self.config.get("error_handling", {})
        max_errors = error_cfg.get("max_errors_per_batch", 10)
        continue_on_error = error_cfg.get("continue_on_error", True)

        # Step 2: Process each video
        self.log.info("-" * 40)
        self.log.info("Phase 1: Processing individual videos")
        self.log.info("-" * 40)

        for video_path in tqdm(video_paths, desc="Processing videos", unit="video"):
            self._process_single_video(video_path, batch_id)

            # Check error limit
            if self.stats["errors"] >= max_errors and not continue_on_error:
                self.log.error(
                    "Max errors reached (%d). Aborting batch.", max_errors
                )
                break

        # Step 3: Face clustering (across all videos in batch)
        self.log.info("-" * 40)
        self.log.info("Phase 2: Face clustering")
        self.log.info("-" * 40)
        self._run_face_clustering(batch_id)

        # Step 4: Unload models
        self._unload_models()

        # Step 5: Generate reports
        elapsed = time.time() - self.stats["start_time"]
        self.log.info("-" * 40)
        self.log.info("Phase 3: Generating reports")
        self.log.info("-" * 40)

        report_path = None
        csv_path = None
        reporting = self.config.get("reporting", {})

        if reporting.get("generate_text_report", True):
            report_path = self._generate_report(batch_id, elapsed)
        if reporting.get("generate_csv_export", True):
            csv_path = self._export_csv(batch_id)

        # Get distributions as JSON for batch log
        batch_stats = self.db.get_batch_stats(batch_id)

        # Update batch log
        self.db.update_batch_log(batch_id, {
            "processing_end": datetime.now().isoformat(),
            "videos_new": self.stats["processed"],
            "videos_processed": self.stats["processed"],
            "videos_failed": self.stats["errors"],
            "videos_skipped": self.stats["skipped"],
            "duration_seconds": round(elapsed, 2),
            "themes_distribution": json.dumps(batch_stats.get("theme_distribution", {})),
            "years_distribution": json.dumps(
                {f"{d['year']}/{d['month']}": d["count"] for d in batch_stats.get("date_distribution", [])}
            ),
            "errors": json.dumps(self.stats["error_details"][-50:]),
            "report_path": report_path,
            "export_path": csv_path,
            "status": "completed",
        })

        # Print summary
        self._print_summary(elapsed)

    def _process_single_video(self, video_path: str, batch_id: str) -> None:
        """Process a single video through the full pipeline.

        Args:
            video_path: Path to the video file.
            batch_id: Current batch ID (text UUID).
        """
        filename = os.path.basename(video_path)
        video_id = None
        video_start = time.time()

        try:
            # --- Step 1: Hash & duplicate check ---
            file_hash = calculate_md5(video_path)

            if self.db.video_exists(file_hash):
                dup_cfg = self.config.get("duplicates", {})
                dup_action = dup_cfg.get("action", "skip")

                if dup_action == "move_to_folder" and not self.dry_run:
                    dup_folder = self.config["organization"].get("special_folders", {}).get(
                        "duplicates", "_ESPECIALES/Duplicados"
                    )
                    dup_path = os.path.join(self.config["paths"]["output"], dup_folder)
                    os.makedirs(dup_path, exist_ok=True)
                    shutil.move(video_path, os.path.join(dup_path, filename))
                    self.log.info("DUPLICATE (moved): %s -> %s", filename, dup_folder)
                    self.stats["duplicates_moved"] += 1
                else:
                    self.log.info("SKIP (duplicate): %s", filename)

                self.stats["skipped"] += 1
                self.db.increment_batch_counter(batch_id, "videos_skipped")
                return

            # --- Step 2: Metadata extraction ---
            metadata = self.metadata_extractor.extract(video_path)

            # --- Step 3: Date extraction ---
            date_info = self.date_extractor.extract(
                video_path, metadata.get("creation_date")
            )

            # --- Step 4: Insert initial video record ---
            video_data = {
                "filename": filename,
                "original_path": video_path,
                "file_hash": file_hash,
                "file_size_bytes": metadata["file_size_bytes"],
                "duration_seconds": metadata["duration_seconds"],
                "width": metadata["width"],
                "height": metadata["height"],
                "aspect_ratio": metadata["aspect_ratio"],
                "fps": metadata["fps"],
                "bitrate": metadata["bitrate"],
                "codec_video": metadata["codec_video"],
                "codec_audio": metadata["codec_audio"],
                "format_type": metadata["format_type"],
                "orientation": metadata["orientation"],
                "has_audio": metadata["has_audio"],
                "creation_date": date_info["creation_date"],
                "date_source": date_info["date_source"],
                "year": date_info["year"],
                "month": date_info["month"],
                "month_name": date_info["month_name"],
                "batch_id": batch_id,
            }
            video_id = self.db.insert_video(video_data)

            # --- Step 5: Frame extraction ---
            frames_with_meta = self.frame_sampler.sample_frames_with_metadata(video_path)
            frames = [f[0] for f in frames_with_meta]

            # Generate thumbnail
            self.frame_sampler.generate_thumbnail(video_path, video_id)

            # --- Step 6: CLIP analysis & theme classification ---
            theme_prompts = self.config["themes"]["prompts"]
            clip_scores = self.clip_analyzer.analyze_frames(frames, theme_prompts)

            primary_theme, theme_confidence, needs_review = (
                self.theme_classifier.classify(clip_scores)
            )

            # Build secondary themes JSON
            secondary_themes = self.theme_classifier.get_secondary_themes_json(
                clip_scores, primary_theme
            )

            # Insert visual tags
            visual_tags = self.theme_classifier.build_visual_tags(video_id, clip_scores)
            self.db.insert_visual_tags(visual_tags)

            # --- Step 7: Face detection ---
            face_records = self.face_detector.detect_faces(frames_with_meta, video_id)
            face_count = len(face_records)
            if face_records:
                self.db.insert_faces(face_records)

            # --- Step 8: Whisper transcription ---
            has_dialogue = False
            audio_language = None
            transcription = None

            if self.config["analysis"]["audio"].get("enable_transcription", True):
                transcription = self.whisper_transcriber.transcribe(
                    video_path, video_id, metadata["has_audio"]
                )
                if transcription:
                    has_dialogue = True
                    audio_language = transcription["language"]
                    self.db.insert_transcription(transcription)

            # --- Step 9: OCR ---
            has_text_overlay = False
            if self.config["analysis"]["text_extraction"].get("enable", True):
                ocr_texts = self.ocr_processor.extract_text(frames_with_meta, video_id)
                has_text_overlay = len(ocr_texts) > 0
                if ocr_texts:
                    self.db.insert_ocr_texts(ocr_texts)

            # --- Step 10: File organization ---
            if not self.dry_run:
                new_path = self.file_organizer.organize(
                    video_path,
                    date_info["year"],
                    date_info["month"],
                    primary_theme,
                )
            else:
                new_path = self.file_organizer.get_target_preview(
                    filename,
                    date_info["year"],
                    date_info["month"],
                    primary_theme,
                )
                self.log.info("DRY RUN -> %s", new_path)

            # --- Step 11: Update database ---
            processing_duration = time.time() - video_start

            self.db.update_video(video_id, {
                "final_path": new_path,
                "primary_theme": primary_theme,
                "theme_confidence": round(theme_confidence, 4),
                "secondary_themes": secondary_themes,
                "has_dialogue": has_dialogue,
                "audio_language": audio_language,
                "has_text_overlay": has_text_overlay,
                "face_count": face_count,
                "needs_review": needs_review,
                "processing_duration_seconds": round(processing_duration, 2),
                "error_message": None,
            })

            self.stats["processed"] += 1
            self.db.increment_batch_counter(batch_id, "videos_processed")

            self.log.info(
                "OK: %s -> %s/%s/%s (%.2f) [%.1fs]",
                filename,
                date_info.get("year", "?"),
                date_info.get("month_name", "?"),
                primary_theme,
                theme_confidence,
                processing_duration,
            )

        except Exception as e:
            self.stats["errors"] += 1
            self.stats["error_details"].append({
                "filename": filename,
                "error": str(e)[:300],
            })
            self.db.increment_batch_counter(batch_id, "videos_failed")

            self.log.error("ERROR processing %s: %s", filename, e, exc_info=True)

            # Move failed video to review folder if configured
            error_cfg = self.config.get("error_handling", {})
            if error_cfg.get("move_failed_videos", False) and not self.dry_run:
                try:
                    review_folder = self.config["organization"].get("special_folders", {}).get(
                        "review_needed", "_ESPECIALES/Revisar_Manual"
                    )
                    review_path = os.path.join(self.config["paths"]["output"], review_folder)
                    os.makedirs(review_path, exist_ok=True)
                    if os.path.exists(video_path):
                        shutil.move(video_path, os.path.join(review_path, filename))
                        self.log.info("Moved failed video to: %s", review_folder)
                except Exception:
                    pass

            # Mark in DB if we have a video_id
            if video_id is not None:
                try:
                    self.db.update_video(video_id, {
                        "error_message": str(e)[:500],
                        "needs_review": True,
                        "processing_duration_seconds": round(time.time() - video_start, 2),
                    })
                except Exception:
                    pass

    def _run_face_clustering(self, batch_id: str) -> None:
        """Run face clustering across all detected faces.

        Args:
            batch_id: Current batch ID for updating video records.
        """
        try:
            face_data = self.db.get_all_face_embeddings()
            if not face_data:
                self.log.info("No face embeddings to cluster")
                return

            assignments = self.face_clusterer.cluster_faces(face_data)
            if not assignments:
                return

            # Create person cluster records and build face->db_cluster_id mapping
            cluster_stats = self.face_clusterer.get_cluster_stats(assignments)
            final_assignments: Dict[int, int] = {}

            for stat in cluster_stats:
                db_cluster_id = self.db.insert_person_cluster({
                    "face_count": stat["face_count"],
                    "label": f"Person_{stat['cluster_id']}",
                })
                for face_id in stat["face_ids"]:
                    final_assignments[face_id] = db_cluster_id

            # Assign noise points (DBSCAN label -1) as cluster_id = -1
            for face_id, label in assignments.items():
                if label == -1:
                    final_assignments[face_id] = -1

            # Single batch update with DB-generated cluster IDs
            self.db.update_face_clusters(final_assignments)

            # Update unique_persons count on each video in this batch
            with self.db._connect() as conn:
                conn.execute(
                    """UPDATE videos SET unique_persons = (
                        SELECT COUNT(DISTINCT f.cluster_id)
                        FROM faces f WHERE f.video_id = videos.id
                        AND f.cluster_id IS NOT NULL AND f.cluster_id >= 0
                    ) WHERE batch_id = ?""",
                    (batch_id,),
                )

            self.log.info(
                "Face clustering: %d clusters from %d faces",
                len(cluster_stats),
                len(face_data),
            )

        except Exception as e:
            self.log.error("Face clustering failed: %s", e, exc_info=True)

    def _unload_models(self) -> None:
        """Unload all AI models to free memory."""
        self.clip_analyzer.unload_model()
        self.face_detector.unload_model()
        self.whisper_transcriber.unload_model()
        self.ocr_processor.unload_model()

    def _generate_report(self, batch_id: str, elapsed: float) -> str:
        """Generate a text report for the batch.

        Args:
            batch_id: Batch ID to report on.
            elapsed: Total processing time in seconds.

        Returns:
            Path to the generated report file.
        """
        reports_dir = os.path.join(self.analysis_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d")
        report_path = os.path.join(reports_dir, f"lote_{batch_id}_{timestamp}.txt")

        stats = self.db.get_batch_stats(batch_id)

        lines = [
            "=" * 60,
            f"  VIDEO INTELLIGENCE - Batch Report",
            f"  Batch ID: {batch_id}",
            f"  Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
            "=" * 60,
            "",
            "SUMMARY",
            "-" * 40,
            f"  Total videos found:    {self.stats['total']}",
            f"  Successfully processed:{self.stats['processed']}",
            f"  Skipped (duplicates):  {self.stats['skipped']}",
            f"  Duplicates moved:      {self.stats['duplicates_moved']}",
            f"  Errors:                {self.stats['errors']}",
            f"  Needs manual review:   {stats.get('needs_review', 0)}",
            f"  Processing time:       {format_duration(elapsed)}",
            "",
            "THEME DISTRIBUTION",
            "-" * 40,
        ]

        for theme, count in stats.get("theme_distribution", {}).items():
            lines.append(f"  {theme:<25} {count:>5}")

        lines.extend([
            "",
            "DATE DISTRIBUTION",
            "-" * 40,
        ])

        for entry in stats.get("date_distribution", []):
            year = entry["year"] or "Sin Fecha"
            month_name = entry.get("month_name") or "?"
            lines.append(f"  {year}/{month_name:<15} {entry['count']:>5}")

        lines.extend([
            "",
            "ANALYSIS DETAILS",
            "-" * 40,
            f"  Total faces detected:  {stats.get('total_faces_detected', 0)}",
            f"  Unique persons found:  {stats.get('unique_persons', 0)}",
            f"  Videos with speech:    {stats.get('videos_with_speech', 0)}",
            f"  Total in database:     {self.db.get_total_video_count()}",
        ])

        if self.stats["error_details"]:
            lines.extend([
                "",
                "ERRORS",
                "-" * 40,
            ])
            for err in self.stats["error_details"]:
                lines.append(f"  {err['filename']}: {err['error'][:80]}")

        lines.extend(["", "=" * 60])

        report_text = "\n".join(lines)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        self.log.info("Report saved: %s", report_path)
        return report_path

    def _export_csv(self, batch_id: str) -> str:
        """Export batch data as CSV.

        Args:
            batch_id: Batch ID to export.

        Returns:
            Path to the generated CSV file.
        """
        exports_dir = os.path.join(self.analysis_dir, "exports")
        os.makedirs(exports_dir, exist_ok=True)

        csv_path = os.path.join(exports_dir, f"batch_{batch_id}.csv")
        videos = self.db.get_all_videos_for_export(batch_id)

        if videos:
            fieldnames = videos[0].keys()
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(videos)

            self.log.info("CSV exported: %s (%d rows)", csv_path, len(videos))
        else:
            self.log.info("No data to export for batch %s", batch_id)

        return csv_path

    def _ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        dirs = [
            self.config["paths"]["input"],
            self.config["paths"]["output"],
            os.path.join(self.analysis_dir, "database"),
            os.path.join(self.analysis_dir, "thumbnails", "videos"),
            os.path.join(self.analysis_dir, "thumbnails", "faces"),
            os.path.join(self.analysis_dir, "reports"),
            os.path.join(self.analysis_dir, "exports"),
            os.path.join(self.analysis_dir, "logs"),
        ]

        # Create special folders
        special = self.config["organization"].get("special_folders", {})
        for folder in special.values():
            dirs.append(os.path.join(self.config["paths"]["output"], folder))

        # Temp dir
        temp_dir = self.config["paths"].get("temp")
        if temp_dir:
            dirs.append(temp_dir)

        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def _print_summary(self, elapsed: float) -> None:
        """Print final processing summary to console."""
        self.log.info("")
        self.log.info("=" * 60)
        self.log.info("PROCESSING COMPLETE")
        self.log.info("=" * 60)
        self.log.info("  Processed: %d", self.stats["processed"])
        self.log.info("  Skipped:   %d", self.stats["skipped"])
        self.log.info("  Errors:    %d", self.stats["errors"])
        self.log.info("  Time:      %s", format_duration(elapsed))
        self.log.info("=" * 60)


def main() -> None:
    """Entry point for the Video Intelligence System."""
    parser = argparse.ArgumentParser(
        description="Video Intelligence System - AI-powered video analysis & organization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Process all videos in input folder
  python main.py --config custom.yaml         # Use custom config
  python main.py --dry-run                    # Preview without moving files
  python main.py --report-only abc12345       # Regenerate report for batch ID
        """,
    )

    parser.add_argument(
        "--config",
        default=os.path.join(PROJECT_ROOT, "config.yaml"),
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview organization without moving files",
    )
    parser.add_argument(
        "--report-only",
        type=str,
        metavar="BATCH_ID",
        help="Regenerate report for an existing batch (no processing)",
    )

    args = parser.parse_args()

    system = VideoIntelligence(config_path=args.config, dry_run=args.dry_run)

    if args.report_only:
        report = system._generate_report(args.report_only, 0)
        csv_out = system._export_csv(args.report_only)
        print(f"Report: {report}")
        print(f"CSV:    {csv_out}")
    else:
        system.run()


if __name__ == "__main__":
    main()

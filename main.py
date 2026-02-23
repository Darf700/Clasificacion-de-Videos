#!/usr/bin/env python3
"""Video Intelligence System - Main Orchestrator.

Analyzes and organizes video files using AI models for face detection,
visual categorization, audio transcription, and OCR text extraction.

Usage:
    python main.py                    # Process all videos in input folder
    python main.py --input /path/to   # Custom input path
    python main.py --config custom.yaml
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

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
from utils.video_utils import classify_format, scan_video_files


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Configuration dictionary.
    """
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = PROJECT_ROOT / config_file

    if not config_file.exists():
        print(f"ERROR: Config file not found: {config_file}")
        sys.exit(1)

    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def print_banner() -> None:
    """Print the application banner."""
    print("\n" + "=" * 55)
    print("  Video Intelligence System")
    print("  AI-powered video analysis & organization")
    print("=" * 55)


def print_summary(
    batch_id: str,
    total: int,
    processed: int,
    skipped: int,
    failed: int,
    elapsed: float,
    theme_dist: dict,
    year_dist: dict,
    n_clusters: int,
    report_path: Optional[str] = None,
) -> None:
    """Print batch processing summary."""
    print("\n" + "=" * 55)
    print("  Processing Complete")
    print("=" * 55)
    print(f"  Batch ID:     {batch_id}")
    print(f"  Total videos: {total}")
    print(f"  Processed:    {processed}")
    print(f"  Skipped:      {skipped}")
    print(f"  Failed:       {failed}")

    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"  Time:         {minutes}min {seconds}sec")

    if theme_dist:
        print("\n  Theme distribution:")
        for theme, count in sorted(theme_dist.items(), key=lambda x: x[1], reverse=True):
            print(f"    {theme}: {count} videos")

    if year_dist:
        print("\n  Year distribution:")
        for year, count in sorted(year_dist.items()):
            print(f"    {year}: {count} videos")

    if n_clusters > 0:
        print(f"\n  Face clusters: {n_clusters}")

    if report_path:
        print(f"\n  Report: {report_path}")
    print("=" * 55 + "\n")


class VideoIntelligenceSystem:
    """Main orchestrator for the video analysis pipeline."""

    def __init__(self, config: dict):
        """Initialize the system with configuration.

        Args:
            config: Configuration dictionary loaded from YAML.
        """
        self.config = config
        self.paths = config["paths"]
        self.batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.errors: list[dict] = []

        # Resolve paths
        analysis_dir = Path(self.paths["analysis"])

        # Setup logging
        log_cfg = config.get("logging", {})
        log_dir = log_cfg.get("log_dir", "_ANALYSIS/logs")
        if not Path(log_dir).is_absolute():
            log_dir = str(analysis_dir / log_dir.replace("_ANALYSIS/", ""))

        self.logger = setup_logging(
            log_dir=log_dir,
            level=log_cfg.get("level", "INFO"),
            console_level=log_cfg.get("console_level", "INFO"),
            file_logging=log_cfg.get("file_logging", True),
            console_logging=log_cfg.get("console_logging", True),
        )

        # Database path resolution
        db_cfg = config.get("database", {})
        db_path = db_cfg.get("path", "_ANALYSIS/database/analysis.db")
        if not Path(db_path).is_absolute():
            db_path = str(analysis_dir / db_path.replace("_ANALYSIS/", ""))
        backup_dir = db_cfg.get("backup_dir", "")
        if backup_dir and not Path(backup_dir).is_absolute():
            backup_dir = str(analysis_dir / backup_dir.replace("_ANALYSIS/", ""))

        # Initialize components
        self.db = DatabaseManager(
            db_path=db_path,
            wal_mode=db_cfg.get("wal_mode", True),
            auto_backup=db_cfg.get("auto_backup", True),
            backup_dir=backup_dir or None,
        )

        proc_cfg = config.get("processing", {})
        model_cfg = config.get("models", {})
        analysis_cfg = config.get("analysis", {})

        self.metadata_extractor = MetadataExtractor()

        self.frame_sampler = FrameSampler(
            frames_per_video=analysis_cfg.get("frames", {}).get("count_per_video", 30),
        )

        self.date_extractor = DateExtractor(
            date_sources=config.get("organization", {}).get("date_sources"),
        )

        clip_cfg = model_cfg.get("clip", {})
        self.clip_analyzer = CLIPAnalyzer(
            model_name=clip_cfg.get("model_name", "ViT-B-32"),
            pretrained=clip_cfg.get("pretrained", "openai"),
            device=clip_cfg.get("device", "cuda"),
        )

        theme_cfg = analysis_cfg.get("themes", {})
        self.theme_classifier = ThemeClassifier(
            confidence_threshold=theme_cfg.get("confidence_threshold", 0.6),
            fallback_theme=theme_cfg.get("fallback_theme", "Otros"),
        )

        face_cfg = model_cfg.get("face", {})
        self.face_detector = FaceDetector(
            detection_threshold=face_cfg.get("detection_threshold", 0.8),
            device=face_cfg.get("device", "cuda"),
        )

        cluster_cfg = analysis_cfg.get("face_clustering", {})
        self.face_clusterer = FaceClusterer(
            algorithm=cluster_cfg.get("algorithm", "dbscan"),
            eps=cluster_cfg.get("eps", 0.6),
            min_samples=cluster_cfg.get("min_samples", 2),
            metric=cluster_cfg.get("metric", "cosine"),
        )

        whisper_cfg = model_cfg.get("whisper", {})
        audio_cfg = analysis_cfg.get("audio", {})
        self.whisper = WhisperTranscriber(
            model_name=whisper_cfg.get("model_name", "medium"),
            device=whisper_cfg.get("device", "cuda"),
            compute_type=whisper_cfg.get("compute_type", "float16"),
            sample_duration=audio_cfg.get("sample_duration", 30),
            speech_threshold=audio_cfg.get("speech_detection_threshold", 0.3),
        )

        ocr_cfg = model_cfg.get("ocr", {})
        text_cfg = analysis_cfg.get("text_extraction", {})
        self.ocr = OCRProcessor(
            languages=ocr_cfg.get("languages", ["es", "en"]),
            gpu=ocr_cfg.get("gpu", True),
            confidence_threshold=ocr_cfg.get("confidence_threshold", 0.5),
            max_frames=text_cfg.get("max_frames", 10),
        )

        org_cfg = config.get("organization", {})
        self.file_organizer = FileOrganizer(
            output_dir=self.paths["output"],
            operation=org_cfg.get("operation", "move"),
            no_date_folder=org_cfg.get("no_date_folder", "Sin_Fecha"),
            month_names=org_cfg.get("month_names"),
        )

        # Theme prompts
        self.theme_prompts = config.get("themes", {}).get("prompts", {})

        # Thumbnail config
        self.thumb_cfg = config.get("thumbnails", {})

        # Error handling config
        self.error_cfg = config.get("error_handling", {})

    def run(self, input_dir: Optional[str] = None) -> None:
        """Run the full processing pipeline.

        Args:
            input_dir: Override input directory from config.
        """
        start_time = time.time()
        input_path = Path(input_dir or self.paths["input"])

        self.logger.info(f"Starting batch {self.batch_id}")
        print(f"\n  Scanning: {input_path}")

        # Initialize database
        self.db.initialize()
        batch_number = self.db.get_next_batch_number()

        # Scan for videos
        videos = scan_video_files(input_path)
        if not videos:
            print("  No video files found in input directory.")
            return

        # Check for duplicates
        new_videos = []
        skipped = 0
        print(f"  Found: {len(videos)} videos")
        print("  Checking for duplicates...")

        for video_path in videos:
            try:
                file_hash = calculate_md5(video_path)
                if self.db.video_exists(file_hash):
                    skipped += 1
                    self.logger.info(f"Skipping duplicate: {video_path.name}")
                else:
                    new_videos.append((video_path, file_hash))
            except Exception as e:
                self.logger.error(f"Hash error for {video_path.name}: {e}")

        print(f"  New: {len(new_videos)} | Already processed: {skipped}")

        if not new_videos:
            print("  Nothing new to process.")
            return

        # Log batch start
        self.db.insert_processing_log({
            "batch_id": self.batch_id,
            "batch_number": batch_number,
            "videos_found": len(videos),
            "videos_new": len(new_videos),
            "videos_processed": 0,
            "videos_failed": 0,
            "videos_skipped": skipped,
            "processing_start": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "running",
        })

        # Process each video
        processed = 0
        failed = 0
        max_errors = self.error_cfg.get("max_errors_per_batch", 10)

        print("\n  Processing videos...")
        progress = tqdm(new_videos, desc="  Processing", unit="video", ncols=70)

        for video_path, file_hash in progress:
            if failed >= max_errors:
                self.logger.error(f"Max errors reached ({max_errors}), stopping batch")
                break

            try:
                progress.set_postfix_str(video_path.name[:25])
                self._process_single_video(video_path, file_hash)
                processed += 1
            except Exception as e:
                failed += 1
                error_msg = f"{video_path.name}: {str(e)}"
                self.errors.append({"file": video_path.name, "error": str(e)})
                self.logger.error(f"Failed to process {error_msg}")

                if self.error_cfg.get("continue_on_error", True):
                    continue
                else:
                    break

        progress.close()

        # Post-processing: face clustering across all videos in batch
        n_clusters = self._run_face_clustering()

        # Unload models
        self._unload_models()

        # Generate reports
        elapsed = time.time() - start_time
        theme_dist = self.db.get_video_count_by_theme(self.batch_id)
        year_dist = self.db.get_video_count_by_year(self.batch_id)

        report_path = self._generate_report(
            batch_number, processed, skipped, failed, elapsed, theme_dist, year_dist, n_clusters
        )

        # Update processing log
        self.db.insert_processing_log({
            "batch_id": self.batch_id,
            "batch_number": batch_number,
            "videos_found": len(videos),
            "videos_new": len(new_videos),
            "videos_processed": processed,
            "videos_failed": failed,
            "videos_skipped": skipped,
            "processing_start": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "processing_end": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(elapsed, 2),
            "themes_distribution": theme_dist,
            "years_distribution": year_dist,
            "errors": self.errors,
            "report_path": report_path,
            "status": "completed" if failed == 0 else "completed",
        })

        self.db.close()

        # Print summary
        print_summary(
            self.batch_id, len(videos), processed, skipped, failed,
            elapsed, theme_dist, year_dist, n_clusters, report_path,
        )

    def _process_single_video(self, video_path: Path, file_hash: str) -> None:
        """Process a single video through the full pipeline.

        Args:
            video_path: Path to the video file.
            file_hash: Pre-computed MD5 hash.
        """
        video_start = time.time()
        logger = get_logger("pipeline")
        logger.info(f"Processing: {video_path.name}")

        # Phase 1: Metadata extraction
        metadata = self.metadata_extractor.extract(video_path)
        metadata["file_hash"] = file_hash
        metadata["batch_id"] = self.batch_id

        # Classify format
        fmt = classify_format(
            metadata.get("width"), metadata.get("height"), metadata.get("duration_seconds")
        )
        metadata.update(fmt)

        # Phase 2: Frame extraction
        duration = metadata.get("duration_seconds", 0)
        frames = self.frame_sampler.extract_frames(video_path, duration=duration)

        # Generate thumbnail
        if self.thumb_cfg.get("generate_video_thumbs", True):
            analysis_dir = Path(self.paths["analysis"])
            thumb_dir = analysis_dir / "thumbnails" / "videos"
            thumb_path = thumb_dir / f"{video_path.stem}.jpg"
            thumb_size = tuple(self.thumb_cfg.get("video_thumb_size", [320, 180]))
            self.frame_sampler.extract_thumbnail(video_path, thumb_path, duration, thumb_size)

        # Phase 3: Date extraction
        exif_date = metadata.pop("exif_creation_date", None)
        date_info = self.date_extractor.extract(video_path, exif_date=exif_date)
        metadata.update(date_info)

        # Phase 4: CLIP visual categorization
        clip_result = {"primary_theme": "Otros", "theme_confidence": 0.0, "all_scores": {}}
        if frames and self.theme_prompts:
            try:
                clip_result = self.clip_analyzer.analyze_frames(frames, self.theme_prompts)
            except Exception as e:
                logger.warning(f"CLIP analysis failed for {video_path.name}: {e}")

        # Phase 5: Theme classification
        theme_result = self.theme_classifier.classify(clip_result)
        metadata["primary_theme"] = theme_result["theme"]
        metadata["theme_confidence"] = theme_result["confidence"]
        metadata["secondary_themes"] = json.dumps(theme_result.get("secondary_themes", []))
        metadata["needs_review"] = theme_result.get("needs_review", False)

        # Insert video record to get ID
        video_id = self.db.insert_video(metadata)

        # Phase 6: Face detection
        if frames:
            try:
                face_results = self.face_detector.detect_faces(frames, video_duration=duration)
                if face_results:
                    for face in face_results:
                        face["video_id"] = video_id
                    self.db.insert_faces(face_results)
                    self.db.update_video(video_id, {"face_count": len(face_results)})
            except Exception as e:
                logger.warning(f"Face detection failed for {video_path.name}: {e}")

        # Phase 7: Audio transcription
        has_dialogue = False
        if metadata.get("has_audio") and self.config.get("analysis", {}).get("audio", {}).get("enable_transcription", True):
            try:
                transcription = self.whisper.transcribe(video_path)
                if transcription:
                    has_dialogue = True
                    transcription["video_id"] = video_id
                    self.db.insert_transcription(transcription)
                    self.db.update_video(video_id, {
                        "has_dialogue": True,
                        "audio_language": transcription.get("language"),
                    })
            except Exception as e:
                logger.warning(f"Transcription failed for {video_path.name}: {e}")

        if not has_dialogue:
            self.db.update_video(video_id, {"has_dialogue": False})

        # Phase 8: OCR text extraction
        if frames and self.config.get("analysis", {}).get("text_extraction", {}).get("enable", True):
            try:
                ocr_results = self.ocr.extract_text(frames, video_duration=duration)
                if ocr_results:
                    for ocr_rec in ocr_results:
                        ocr_rec["video_id"] = video_id
                    self.db.insert_ocr_texts(ocr_results)
                    self.db.update_video(video_id, {"has_text_overlay": True})
                else:
                    self.db.update_video(video_id, {"has_text_overlay": False})
            except Exception as e:
                logger.warning(f"OCR failed for {video_path.name}: {e}")

        # Insert visual tags
        if clip_result.get("all_scores"):
            tags = []
            for tag, score in clip_result["all_scores"].items():
                tags.append({
                    "video_id": video_id,
                    "tag": tag,
                    "confidence": score,
                    "source": "clip_auto",
                })
            self.db.insert_visual_tags(tags)

        # Phase 9: File organization
        final_path = self.file_organizer.organize(
            video_path,
            year=date_info.get("year"),
            month=date_info.get("month"),
            theme=theme_result["theme"],
        )

        # Update final path and processing time
        elapsed = time.time() - video_start
        self.db.update_video(video_id, {
            "final_path": str(final_path),
            "processing_duration_seconds": round(elapsed, 2),
        })

        logger.info(
            f"Done: {video_path.name} -> {theme_result['theme']} "
            f"({theme_result['confidence']:.0%}) [{elapsed:.1f}s]"
        )

    def _run_face_clustering(self) -> int:
        """Run face clustering across all faces in the database.

        Returns:
            Number of clusters found.
        """
        logger = get_logger("pipeline")
        try:
            face_records = self.db.get_all_face_embeddings()
            if not face_records:
                return 0

            face_cluster_map = self.face_clusterer.cluster(face_records)
            if not face_cluster_map:
                return 0

            # Update face cluster assignments
            self.db.update_face_clusters(face_cluster_map)

            # Build and save cluster stats
            stats = self.face_clusterer.build_cluster_stats(face_cluster_map, face_records)
            for stat in stats:
                self.db.upsert_person_cluster(stat)

            # Update video-person relationships
            for face_id, cluster_id in face_cluster_map.items():
                if cluster_id == -1:
                    continue
                record = next((r for r in face_records if r["id"] == face_id), None)
                if record:
                    self.db.upsert_video_person(
                        record["video_id"], cluster_id, 1, 0.0
                    )

            n_clusters = len(set(face_cluster_map.values()) - {-1})
            logger.info(f"Face clustering complete: {n_clusters} clusters")
            return n_clusters

        except Exception as e:
            logger.warning(f"Face clustering failed: {e}")
            return 0

    def _unload_models(self) -> None:
        """Unload all AI models to free GPU memory."""
        self.clip_analyzer.unload_model()
        self.face_detector.unload_model()
        self.whisper.unload_model()
        self.ocr.unload_model()

    def _generate_report(
        self,
        batch_number: int,
        processed: int,
        skipped: int,
        failed: int,
        elapsed: float,
        theme_dist: dict,
        year_dist: dict,
        n_clusters: int,
    ) -> Optional[str]:
        """Generate batch processing report.

        Args:
            batch_number: Sequential batch number.
            processed: Number of videos processed.
            skipped: Number of videos skipped.
            failed: Number of videos that failed.
            elapsed: Total processing time in seconds.
            theme_dist: Theme distribution dictionary.
            year_dist: Year distribution dictionary.
            n_clusters: Number of face clusters.

        Returns:
            Path to the generated report file, or None.
        """
        report_cfg = self.config.get("reporting", {})
        analysis_dir = Path(self.paths["analysis"])

        report_path = None

        if report_cfg.get("generate_text_report", True):
            report_dir = analysis_dir / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            date_str = datetime.now().strftime("%Y-%m-%d")
            report_file = report_dir / f"lote_{batch_number:03d}_{date_str}.txt"

            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)

            lines = [
                "=" * 55,
                "  Video Intelligence System - Batch Report",
                "=" * 55,
                f"  Batch:     #{batch_number}",
                f"  Batch ID:  {self.batch_id}",
                f"  Date:      {date_str}",
                f"  Duration:  {minutes}min {seconds}sec",
                "",
                f"  Videos processed: {processed}",
                f"  Videos skipped:   {skipped}",
                f"  Videos failed:    {failed}",
                "",
            ]

            if theme_dist:
                lines.append("  Theme Distribution:")
                for theme, count in sorted(theme_dist.items(), key=lambda x: x[1], reverse=True):
                    lines.append(f"    {theme}: {count}")
                lines.append("")

            if year_dist:
                lines.append("  Year Distribution:")
                for year, count in sorted(year_dist.items()):
                    lines.append(f"    {year}: {count}")
                lines.append("")

            lines.append(f"  Face clusters: {n_clusters}")

            if self.errors:
                lines.append("")
                lines.append("  Errors:")
                for err in self.errors:
                    lines.append(f"    - {err['file']}: {err['error']}")

            lines.append("")
            lines.append("=" * 55)

            with open(report_file, "w") as f:
                f.write("\n".join(lines))

            report_path = str(report_file)
            self.logger.info(f"Report saved: {report_file}")

        if report_cfg.get("generate_csv_export", True):
            try:
                import pandas as pd

                export_dir = analysis_dir / "exports"
                export_dir.mkdir(parents=True, exist_ok=True)
                csv_file = export_dir / f"batch_{batch_number:03d}.csv"

                batch_videos = self.db.get_batch_videos(self.batch_id)
                if batch_videos:
                    df = pd.DataFrame(batch_videos)
                    # Drop binary columns
                    for col in df.columns:
                        if df[col].dtype == object:
                            try:
                                df[col].astype(str)
                            except Exception:
                                df = df.drop(columns=[col])
                    df.to_csv(csv_file, index=False)
                    self.logger.info(f"CSV export saved: {csv_file}")
            except Exception as e:
                self.logger.warning(f"CSV export failed: {e}")

        return report_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Video Intelligence System - AI-powered video analysis & organization"
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--input", "-i",
        default=None,
        help="Override input directory",
    )
    args = parser.parse_args()

    print_banner()

    config = load_config(args.config)
    system = VideoIntelligenceSystem(config)
    system.run(input_dir=args.input)


if __name__ == "__main__":
    main()

"""Audio transcription using OpenAI Whisper."""

import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np

from utils.gpu_utils import clear_gpu_memory, get_device
from utils.logging_utils import get_logger

logger = get_logger("whisper")


class WhisperTranscriber:
    """Transcribes audio from video files using Whisper."""

    def __init__(
        self,
        model_name: str = "medium",
        device: str = "cuda",
        compute_type: str = "float16",
        sample_duration: int = 30,
        speech_threshold: float = 0.3,
    ):
        """Initialize Whisper transcriber.

        Args:
            model_name: Whisper model size.
            device: Device to use.
            compute_type: Compute type for inference.
            sample_duration: Seconds to sample for speech detection.
            speech_threshold: Energy threshold for speech detection.
        """
        self.model_name = model_name
        self.device_name = device
        self.compute_type = compute_type
        self.sample_duration = sample_duration
        self.speech_threshold = speech_threshold
        self.model = None

    def load_model(self) -> None:
        """Load the Whisper model."""
        import whisper

        device = get_device(self.device_name)
        logger.info(f"Loading Whisper model: {self.model_name}")
        self.model = whisper.load_model(self.model_name, device=device)
        logger.info("Whisper model loaded")

    def transcribe(self, video_path: str | Path) -> Optional[dict]:
        """Transcribe audio from a video file.

        Args:
            video_path: Path to the video file.

        Returns:
            Dictionary with full_text, language, confidence, word_count,
            processing_time_seconds, or None if no speech detected.
        """
        video_path = Path(video_path)

        # Extract audio to WAV
        audio_path = self._extract_audio(video_path)
        if audio_path is None:
            return None

        try:
            # Check for speech
            if not self._has_speech(audio_path):
                logger.debug(f"No speech detected in {video_path.name}")
                return None

            if self.model is None:
                self.load_model()

            logger.debug(f"Transcribing {video_path.name}")
            start_time = time.time()

            result = self.model.transcribe(
                str(audio_path),
                language=None,  # Auto-detect
                task="transcribe",
                fp16=(self.compute_type == "float16"),
            )

            elapsed = time.time() - start_time
            text = result.get("text", "").strip()

            if not text:
                return None

            # Estimate confidence from segment probabilities
            segments = result.get("segments", [])
            avg_prob = 0.0
            if segments:
                probs = [s.get("avg_logprob", -1) for s in segments]
                avg_prob = np.exp(np.mean(probs))

            transcription = {
                "full_text": text,
                "language": result.get("language", "unknown"),
                "confidence": round(float(avg_prob), 4),
                "word_count": len(text.split()),
                "processing_time_seconds": round(elapsed, 2),
            }

            logger.debug(
                f"Transcription: {len(text)} chars, lang={transcription['language']}, "
                f"time={elapsed:.1f}s"
            )
            return transcription

        finally:
            # Clean up temp audio file
            if audio_path and Path(audio_path).exists():
                Path(audio_path).unlink()

    def _extract_audio(self, video_path: Path) -> Optional[str]:
        """Extract audio from video to a temporary WAV file.

        Args:
            video_path: Path to the video file.

        Returns:
            Path to temporary WAV file, or None on failure.
        """
        try:
            audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            audio_path = audio_file.name
            audio_file.close()

            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i", str(video_path),
                    "-vn",
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    audio_path,
                ],
                capture_output=True,
                timeout=60,
            )

            if result.returncode != 0:
                Path(audio_path).unlink(missing_ok=True)
                return None

            return audio_path

        except (subprocess.TimeoutExpired, Exception) as e:
            logger.debug(f"Audio extraction failed: {e}")
            return None

    def _has_speech(self, audio_path: str) -> bool:
        """Detect if audio contains speech using energy analysis.

        Args:
            audio_path: Path to WAV audio file.

        Returns:
            True if speech is likely present.
        """
        try:
            import wave

            with wave.open(audio_path, "rb") as wf:
                n_frames = wf.getnframes()
                if n_frames == 0:
                    return False

                # Read a sample of frames
                sample_frames = min(n_frames, 16000 * self.sample_duration)
                raw_data = wf.readframes(sample_frames)

            audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
            audio = audio / 32768.0  # Normalize

            # Calculate RMS energy
            rms = np.sqrt(np.mean(audio ** 2))
            return rms > self.speech_threshold

        except Exception as e:
            logger.debug(f"Speech detection error: {e}")
            return True  # Assume speech if detection fails

    def unload_model(self) -> None:
        """Unload the Whisper model to free memory."""
        self.model = None
        clear_gpu_memory()
        logger.debug("Whisper model unloaded")

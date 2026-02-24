"""Audio transcription using OpenAI Whisper.

Returns transcription records aligned with official schema:
- processing_time_seconds (not duration_seconds)
"""

import os
import subprocess
import tempfile
import time
from typing import Any, Dict, Optional

import numpy as np

from utils.gpu_utils import clear_gpu_memory, get_device
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class WhisperTranscriber:
    """Transcribes speech from video audio using Whisper.

    First detects if the video contains speech, then transcribes
    using the specified Whisper model.

    Args:
        model_name: Whisper model size (tiny, base, small, medium, large).
        device: Compute device ('cuda' or 'cpu'). Auto-detected if None.
        audio_sample_duration: Duration in seconds to check for speech.
    """

    def __init__(
        self,
        model_name: str = "medium",
        device: Optional[str] = None,
        audio_sample_duration: int = 30,
    ) -> None:
        self.model_name = model_name
        self.device = device or get_device()
        self.audio_sample_duration = audio_sample_duration
        self.model = None

    def load_model(self) -> None:
        """Load the Whisper model."""
        import whisper

        self.model = whisper.load_model(self.model_name, device=self.device)
        logger.info("Whisper model loaded: %s on %s", self.model_name, self.device)

    def transcribe(
        self,
        video_path: str,
        video_id: int,
        has_audio: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Transcribe speech from a video file.

        First checks if the video contains speech, then runs full
        transcription if speech is detected.

        Args:
            video_path: Path to the video file.
            video_id: Database ID of the video.
            has_audio: Whether the video has an audio track.

        Returns:
            Transcription dictionary matching schema:
                - video_id: int
                - full_text: str
                - language: str
                - confidence: float
                - word_count: int
                - processing_time_seconds: float
            Or None if no speech detected or no audio.
        """
        if not has_audio:
            logger.debug("Video %d has no audio track, skipping transcription", video_id)
            return None

        # Extract audio to temporary WAV file
        audio_path = self._extract_audio(video_path)
        if audio_path is None:
            return None

        try:
            # Check for speech using energy analysis
            if not self._has_speech(audio_path):
                logger.debug("No speech detected in video %d", video_id)
                return None

            # Run Whisper transcription
            if self.model is None:
                self.load_model()

            start_time = time.time()

            result = self.model.transcribe(
                audio_path,
                language=None,  # Auto-detect
                task="transcribe",
                verbose=False,
            )

            processing_time = time.time() - start_time

            text = result.get("text", "").strip()
            language = result.get("language", "unknown")

            if not text:
                logger.debug("Whisper returned empty transcription for video %d", video_id)
                return None

            # Calculate average confidence from segments
            segments = result.get("segments", [])
            avg_confidence = 0.0
            if segments:
                confidences = [
                    1.0 - seg.get("no_speech_prob", 0.0)
                    for seg in segments
                ]
                avg_confidence = float(np.mean(confidences))

            transcription = {
                "video_id": video_id,
                "full_text": text,
                "language": language,
                "confidence": round(avg_confidence, 4),
                "word_count": len(text.split()),
                "processing_time_seconds": round(processing_time, 2),
            }

            logger.debug(
                "Transcribed video %d: %d words, lang=%s, time=%.1fs",
                video_id,
                transcription["word_count"],
                language,
                processing_time,
            )
            return transcription

        finally:
            # Clean up temporary audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)

    def _extract_audio(self, video_path: str) -> Optional[str]:
        """Extract audio from video to a temporary WAV file.

        Args:
            video_path: Path to the video file.

        Returns:
            Path to temporary WAV file, or None on failure.
        """
        try:
            # mktemp is deprecated due to race conditions; use mkstemp instead
            fd, audio_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)  # Close the file descriptor; ffmpeg will write to the path

            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-vn",                    # No video
                "-acodec", "pcm_s16le",   # PCM 16-bit
                "-ar", "16000",           # 16kHz sample rate (Whisper expects this)
                "-ac", "1",               # Mono
                "-y",                     # Overwrite
                audio_path,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                logger.warning("Audio extraction failed: %s", result.stderr.strip()[:200])
                return None

            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                return None

            return audio_path

        except subprocess.TimeoutExpired:
            logger.warning("Audio extraction timed out for %s", video_path)
            return None
        except Exception as e:
            logger.warning("Audio extraction error: %s", e)
            return None

    def _has_speech(self, audio_path: str) -> bool:
        """Detect if audio contains speech using energy analysis.

        Analyzes the first N seconds of audio to determine if there
        is enough energy variance to indicate speech.

        Args:
            audio_path: Path to WAV audio file.

        Returns:
            True if speech is likely present.
        """
        try:
            import wave

            with wave.open(audio_path, "rb") as wf:
                sample_rate = wf.getframerate()
                n_frames = min(
                    wf.getnframes(),
                    sample_rate * self.audio_sample_duration,
                )
                frames = wf.readframes(n_frames)

            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)

            if len(audio) == 0:
                return False

            # Normalize
            max_val = np.max(np.abs(audio))
            if max_val == 0:
                return False
            audio = audio / max_val

            # Calculate RMS energy in 50ms windows
            window_size = int(sample_rate * 0.05)
            if window_size == 0:
                return False

            n_windows = len(audio) // window_size
            if n_windows == 0:
                return False

            energies = []
            for i in range(n_windows):
                window = audio[i * window_size : (i + 1) * window_size]
                rms = np.sqrt(np.mean(window**2))
                energies.append(rms)

            energies = np.array(energies)

            # Speech detection heuristics:
            # 1. Mean energy above silence threshold
            # 2. Energy variance (speech has dynamic range)
            mean_energy = np.mean(energies)
            energy_std = np.std(energies)
            active_ratio = np.mean(energies > 0.02)

            has_speech = mean_energy > 0.01 and energy_std > 0.005 and active_ratio > 0.1

            logger.debug(
                "Speech detection: mean=%.4f, std=%.4f, active=%.2f -> %s",
                mean_energy,
                energy_std,
                active_ratio,
                "speech" if has_speech else "no speech",
            )
            return has_speech

        except Exception as e:
            logger.warning("Speech detection failed: %s", e)
            # Default to True so we don't skip videos with speech
            return True

    def unload_model(self) -> None:
        """Unload Whisper model to free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
            clear_gpu_memory()
            logger.info("Whisper model unloaded")

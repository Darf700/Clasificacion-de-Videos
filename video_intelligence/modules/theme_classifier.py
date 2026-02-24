"""Theme classification logic based on CLIP analysis results.

Builds visual_tags records using schema field names:
- tag (not theme)
- source (default 'clip_auto')
"""

import json
from typing import Dict, List, Optional, Tuple

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class ThemeClassifier:
    """Assigns themes to videos based on CLIP similarity scores.

    Takes the raw CLIP scores per theme and selects the best matching
    theme based on confidence thresholds.

    Args:
        confidence_threshold: Minimum confidence to assign a theme.
        fallback_theme: Theme name when no category meets the threshold.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        fallback_theme: str = "Otros",
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.fallback_theme = fallback_theme

    def classify(
        self, clip_scores: Dict[str, float]
    ) -> Tuple[str, float, bool]:
        """Classify a video into a theme based on CLIP scores.

        Args:
            clip_scores: Dictionary mapping theme names to confidence scores.

        Returns:
            Tuple of (theme_name, confidence, needs_review).
            - theme_name: Selected theme or fallback.
            - confidence: Score of the selected theme.
            - needs_review: True if assigned to fallback or low confidence.
        """
        if not clip_scores:
            logger.warning("No CLIP scores provided, assigning fallback theme")
            return (self.fallback_theme, 0.0, True)

        # Sort by confidence descending
        sorted_themes = sorted(clip_scores.items(), key=lambda x: x[1], reverse=True)
        best_theme, best_score = sorted_themes[0]

        if best_score >= self.confidence_threshold:
            logger.debug(
                "Theme assigned: %s (%.3f)", best_theme, best_score
            )
            return (best_theme, best_score, False)
        else:
            logger.debug(
                "Below threshold (%.3f < %.3f), best was: %s. Assigning: %s",
                best_score,
                self.confidence_threshold,
                best_theme,
                self.fallback_theme,
            )
            return (self.fallback_theme, best_score, True)

    def get_top_themes(
        self, clip_scores: Dict[str, float], top_n: int = 3
    ) -> List[Tuple[str, float]]:
        """Get the top N themes by confidence.

        Args:
            clip_scores: Dictionary mapping theme names to confidence scores.
            top_n: Number of top themes to return.

        Returns:
            List of (theme_name, confidence) tuples, sorted by confidence.
        """
        sorted_themes = sorted(clip_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_themes[:top_n]

    def get_secondary_themes_json(
        self,
        clip_scores: Dict[str, float],
        primary_theme: str,
        top_n: int = 3,
    ) -> Optional[str]:
        """Get secondary themes as a JSON array string for storage.

        Args:
            clip_scores: All CLIP theme scores.
            primary_theme: The assigned primary theme (excluded).
            top_n: Number of secondary themes to include.

        Returns:
            JSON string of secondary theme names, or None.
        """
        secondary = [
            theme for theme, _ in sorted(
                clip_scores.items(), key=lambda x: x[1], reverse=True
            )
            if theme != primary_theme
        ][:top_n]

        return json.dumps(secondary) if secondary else None

    def build_visual_tags(
        self,
        video_id: int,
        clip_scores: Dict[str, float],
        source: str = "clip_auto",
    ) -> List[Dict]:
        """Build visual_tags records for database insertion.

        Args:
            video_id: Database ID of the video.
            clip_scores: All CLIP theme scores.
            source: Tag source identifier.

        Returns:
            List of dictionaries matching the visual_tags schema.
        """
        tags = []
        for theme, confidence in clip_scores.items():
            tags.append({
                "video_id": video_id,
                "tag": theme,
                "confidence": round(confidence, 4),
                "source": source,
            })
        return tags

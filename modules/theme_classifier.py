"""Theme classification logic combining CLIP scores with rules."""

from typing import Optional

from utils.logging_utils import get_logger

logger = get_logger("theme_classifier")


class ThemeClassifier:
    """Assigns themes to videos based on CLIP analysis results."""

    def __init__(self, confidence_threshold: float = 0.6, fallback_theme: str = "Otros"):
        """Initialize theme classifier.

        Args:
            confidence_threshold: Minimum confidence to assign a theme.
            fallback_theme: Theme to use when confidence is too low.
        """
        self.confidence_threshold = confidence_threshold
        self.fallback_theme = fallback_theme

    def classify(self, clip_result: dict) -> dict:
        """Classify a video into a theme based on CLIP scores.

        Args:
            clip_result: Result dict from CLIPAnalyzer.analyze_frames().

        Returns:
            Dictionary with:
                theme: Assigned theme name.
                confidence: Confidence score.
                needs_review: Whether manual review is recommended.
                secondary_themes: List of other likely themes.
        """
        primary = clip_result.get("primary_theme", self.fallback_theme)
        confidence = clip_result.get("theme_confidence", 0.0)
        all_scores = clip_result.get("all_scores", {})

        needs_review = False

        if confidence < self.confidence_threshold:
            logger.info(
                f"Low confidence ({confidence:.2%}) for {primary}, "
                f"assigning to {self.fallback_theme}"
            )
            primary = self.fallback_theme
            needs_review = True

        # Get secondary themes (above half the threshold)
        secondary = []
        half_threshold = self.confidence_threshold / 2
        for theme, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
            if theme != primary and score >= half_threshold:
                secondary.append(theme)
            if len(secondary) >= 2:
                break

        result = {
            "theme": primary,
            "confidence": round(confidence, 4),
            "needs_review": needs_review,
            "secondary_themes": secondary,
        }

        logger.debug(f"Classification: {primary} ({confidence:.2%})")
        return result

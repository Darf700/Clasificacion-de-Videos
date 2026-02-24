"""Tests for modules.theme_classifier - ThemeClassifier."""

import json

import pytest

from modules.theme_classifier import ThemeClassifier


class TestThemeClassifier:
    @pytest.fixture
    def classifier(self):
        return ThemeClassifier(confidence_threshold=0.6, fallback_theme="Otros")

    # --- classify ---

    def test_classify_above_threshold(self, classifier):
        scores = {"Comedia": 0.85, "Drama": 0.45, "Musica": 0.30}
        theme, confidence, needs_review = classifier.classify(scores)
        assert theme == "Comedia"
        assert confidence == 0.85
        assert needs_review is False

    def test_classify_below_threshold(self, classifier):
        scores = {"Comedia": 0.50, "Drama": 0.45}
        theme, confidence, needs_review = classifier.classify(scores)
        assert theme == "Otros"
        assert confidence == 0.50
        assert needs_review is True

    def test_classify_empty_scores(self, classifier):
        theme, confidence, needs_review = classifier.classify({})
        assert theme == "Otros"
        assert confidence == 0.0
        assert needs_review is True

    def test_classify_exact_threshold(self):
        c = ThemeClassifier(confidence_threshold=0.7)
        theme, conf, review = c.classify({"A": 0.7, "B": 0.3})
        assert theme == "A"
        assert review is False

    def test_classify_single_theme(self, classifier):
        theme, conf, review = classifier.classify({"Solo": 0.9})
        assert theme == "Solo"
        assert review is False

    # --- get_top_themes ---

    def test_top_themes(self, classifier):
        scores = {"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3}
        top = classifier.get_top_themes(scores, top_n=2)
        assert len(top) == 2
        assert top[0] == ("A", 0.9)
        assert top[1] == ("B", 0.7)

    def test_top_themes_less_than_n(self, classifier):
        scores = {"A": 0.9}
        top = classifier.get_top_themes(scores, top_n=5)
        assert len(top) == 1

    # --- get_secondary_themes_json ---

    def test_secondary_themes_json(self, classifier):
        scores = {"Comedia": 0.9, "Drama": 0.7, "Accion": 0.5, "Terror": 0.3}
        result = classifier.get_secondary_themes_json(scores, "Comedia", top_n=2)
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert "Comedia" not in parsed
        assert parsed[0] == "Drama"

    def test_secondary_themes_json_no_others(self, classifier):
        result = classifier.get_secondary_themes_json({"Solo": 0.9}, "Solo")
        assert result is None

    # --- build_visual_tags ---

    def test_build_visual_tags(self, classifier):
        scores = {"Comedia": 0.85, "Drama": 0.45}
        tags = classifier.build_visual_tags(1, scores, source="clip_auto")
        assert len(tags) == 2
        for tag in tags:
            assert "video_id" in tag
            assert "tag" in tag
            assert "confidence" in tag
            assert "source" in tag
            assert tag["video_id"] == 1
            assert tag["source"] == "clip_auto"

    def test_build_visual_tags_empty(self, classifier):
        tags = classifier.build_visual_tags(1, {})
        assert tags == []

    # --- Custom fallback ---

    def test_custom_fallback_theme(self):
        c = ThemeClassifier(fallback_theme="Sin_Tema")
        theme, _, _ = c.classify({"A": 0.1})
        assert theme == "Sin_Tema"

"""Tests for modules.file_organizer - FileOrganizer."""

import os

import pytest

from modules.file_organizer import FileOrganizer


class TestFileOrganizer:
    @pytest.fixture
    def organizer(self, tmp_dir):
        return FileOrganizer(output_base=tmp_dir, operation="copy")

    @pytest.fixture
    def source_video(self, tmp_dir):
        """Create a fake source video file."""
        src = os.path.join(tmp_dir, "source", "video.mp4")
        os.makedirs(os.path.dirname(src), exist_ok=True)
        with open(src, "wb") as f:
            f.write(b"\x00" * 1024)
        return src

    # --- __init__ validation ---

    def test_invalid_operation_raises(self, tmp_dir):
        with pytest.raises(ValueError, match="Invalid operation"):
            FileOrganizer(output_base=tmp_dir, operation="delete")

    def test_valid_operations(self, tmp_dir):
        FileOrganizer(output_base=tmp_dir, operation="move")
        FileOrganizer(output_base=tmp_dir, operation="copy")

    # --- _build_target_dir ---

    def test_build_target_dir_full(self, organizer, tmp_dir):
        d = organizer._build_target_dir(2023, 4, "Comedia")
        expected = os.path.join(tmp_dir, "2023", "Abril", "Comedia")
        assert d == expected

    def test_build_target_dir_no_theme(self, organizer, tmp_dir):
        d = organizer._build_target_dir(2023, 4, None)
        expected = os.path.join(tmp_dir, "2023", "Abril", "Otros")
        assert d == expected

    def test_build_target_dir_no_date(self, organizer, tmp_dir):
        d = organizer._build_target_dir(None, None, "Comedia")
        expected = os.path.join(tmp_dir, "Sin_Fecha", "Comedia")
        assert d == expected

    def test_build_target_dir_no_date_no_theme(self, organizer, tmp_dir):
        d = organizer._build_target_dir(None, None, None)
        expected = os.path.join(tmp_dir, "Sin_Fecha", "Otros")
        assert d == expected

    def test_build_target_dir_year_only(self, organizer, tmp_dir):
        # year without month -> treated as no date
        d = organizer._build_target_dir(2023, None, "X")
        expected = os.path.join(tmp_dir, "Sin_Fecha", "X")
        assert d == expected

    def test_all_months_have_names(self, organizer):
        for m in range(1, 13):
            d = organizer._build_target_dir(2023, m, "T")
            assert "Mes_" not in d  # All should use Spanish names

    # --- organize (copy mode) ---

    def test_organize_copy(self, organizer, source_video):
        result = organizer.organize(source_video, 2023, 6, "Comedia")
        assert os.path.exists(result)
        assert "2023" in result
        assert "Junio" in result
        assert "Comedia" in result
        # Source still exists (copy mode)
        assert os.path.exists(source_video)

    def test_organize_move(self, tmp_dir, source_video):
        org = FileOrganizer(output_base=tmp_dir, operation="move")
        result = org.organize(source_video, 2023, 1, "Drama")
        assert os.path.exists(result)
        assert not os.path.exists(source_video)

    def test_organize_nonexistent_source(self, organizer):
        with pytest.raises(FileNotFoundError):
            organizer.organize("/nonexistent/video.mp4", 2023, 1, "X")

    # --- Collision resolution ---

    def test_collision_resolution(self, organizer, tmp_dir, source_video):
        # First copy
        result1 = organizer.organize(source_video, 2023, 3, "Test")
        assert os.path.exists(result1)

        # Recreate the source
        with open(source_video, "wb") as f:
            f.write(b"\x00" * 1024)

        # Second copy should get _1 suffix
        result2 = organizer.organize(source_video, 2023, 3, "Test")
        assert os.path.exists(result2)
        assert result1 != result2
        assert "_1" in os.path.basename(result2)

    def test_resolve_collision_no_conflict(self, organizer, tmp_dir):
        path = os.path.join(tmp_dir, "unique_file.mp4")
        assert organizer._resolve_collision(path) == path

    # --- Preview ---

    def test_get_target_preview(self, organizer):
        preview = organizer.get_target_preview("test.mp4", 2023, 4, "Comedia")
        assert "2023" in preview
        assert "Abril" in preview
        assert "Comedia" in preview
        assert "test.mp4" in preview

    # --- Relative path ---

    def test_relative_path(self, organizer, tmp_dir):
        full = os.path.join(tmp_dir, "2023", "Enero", "video.mp4")
        rel = organizer._relative_path(full)
        assert rel == os.path.join("2023", "Enero", "video.mp4")

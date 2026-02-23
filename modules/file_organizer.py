"""File organization - moves videos to Year/Month/Theme directory structure."""

import shutil
from pathlib import Path
from typing import Optional

from utils.logging_utils import get_logger

logger = get_logger("file_organizer")

MONTH_NAMES = {
    1: "Enero",
    2: "Febrero",
    3: "Marzo",
    4: "Abril",
    5: "Mayo",
    6: "Junio",
    7: "Julio",
    8: "Agosto",
    9: "Septiembre",
    10: "Octubre",
    11: "Noviembre",
    12: "Diciembre",
}


class FileOrganizer:
    """Organizes video files into Year/Month/Theme directory structure."""

    def __init__(
        self,
        output_dir: str,
        operation: str = "move",
        no_date_folder: str = "Sin_Fecha",
        month_names: Optional[dict] = None,
    ):
        """Initialize file organizer.

        Args:
            output_dir: Base output directory for organized videos.
            operation: 'move' or 'copy'.
            no_date_folder: Folder name for videos without dates.
            month_names: Custom month name mapping (number -> name).
        """
        self.output_dir = Path(output_dir)
        self.operation = operation
        self.no_date_folder = no_date_folder
        self.month_names = month_names or MONTH_NAMES

    def organize(
        self,
        video_path: str | Path,
        year: Optional[int],
        month: Optional[int],
        theme: str,
    ) -> Path:
        """Move or copy a video to its organized location.

        Args:
            video_path: Current path to the video file.
            year: Year for organization (None = no date).
            month: Month for organization (None = no date).
            theme: Theme category name.

        Returns:
            New path of the organized file.
        """
        video_path = Path(video_path)
        target_dir = self._build_target_dir(year, month, theme)
        target_dir.mkdir(parents=True, exist_ok=True)

        target_path = target_dir / video_path.name

        # Handle filename conflicts
        if target_path.exists():
            target_path = self._resolve_conflict(target_path)

        if self.operation == "move":
            shutil.move(str(video_path), str(target_path))
            logger.info(f"Moved: {video_path.name} -> {target_path.relative_to(self.output_dir)}")
        else:
            shutil.copy2(str(video_path), str(target_path))
            logger.info(f"Copied: {video_path.name} -> {target_path.relative_to(self.output_dir)}")

        return target_path

    def _build_target_dir(
        self,
        year: Optional[int],
        month: Optional[int],
        theme: str,
    ) -> Path:
        """Build the target directory path.

        Args:
            year: Year (or None).
            month: Month (or None).
            theme: Theme name.

        Returns:
            Target directory path.
        """
        if year and month:
            month_name = self.month_names.get(month, f"Mes_{month:02d}")
            return self.output_dir / str(year) / month_name / theme
        elif year:
            return self.output_dir / str(year) / self.no_date_folder / theme
        else:
            return self.output_dir / self.no_date_folder / theme

    def _resolve_conflict(self, path: Path) -> Path:
        """Resolve filename conflicts by appending a counter.

        Args:
            path: Conflicting file path.

        Returns:
            New unique path.
        """
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        counter = 1

        while path.exists():
            path = parent / f"{stem}_{counter}{suffix}"
            counter += 1

        return path

    def move_to_special(
        self,
        video_path: str | Path,
        special_folder: str,
        base_dir: str | Path = None,
    ) -> Path:
        """Move a video to a special folder (duplicates, review, etc.).

        Args:
            video_path: Current path to the video.
            special_folder: Name of the special folder.
            base_dir: Base directory for special folders (defaults to output_dir parent).

        Returns:
            New path of the file.
        """
        video_path = Path(video_path)
        if base_dir is None:
            target_dir = self.output_dir.parent / special_folder
        else:
            target_dir = Path(base_dir) / special_folder

        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / video_path.name

        if target_path.exists():
            target_path = self._resolve_conflict(target_path)

        shutil.move(str(video_path), str(target_path))
        logger.info(f"Moved to special: {video_path.name} -> {special_folder}")
        return target_path

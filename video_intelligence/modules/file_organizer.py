"""File organization - moves videos to Year/Month/Theme structure."""

import os
import shutil
from pathlib import Path
from typing import Dict, Optional

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class FileOrganizer:
    """Organizes video files into Year/Month/Theme directory structure.

    Moves (or copies) video files from the input directory to an organized
    structure under the output directory.

    Args:
        output_base: Base output directory (e.g., /mnt/video_hub/01_PROCESADOS).
        operation: File operation mode - 'move' or 'copy'.
        no_date_folder: Folder name for videos without a date.
        no_theme_folder: Folder name for videos without a theme.
        month_names: Dictionary mapping month number to Spanish name.
    """

    def __init__(
        self,
        output_base: str,
        operation: str = "move",
        no_date_folder: str = "Sin_Fecha",
        no_theme_folder: str = "Otros",
        month_names: Optional[Dict[int, str]] = None,
    ) -> None:
        if operation not in ("move", "copy"):
            raise ValueError(f"Invalid operation '{operation}', must be 'move' or 'copy'")
        self.output_base = output_base
        self.operation = operation
        self.no_date_folder = no_date_folder
        self.no_theme_folder = no_theme_folder
        self.month_names = month_names or {
            1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
            5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
            9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
        }

    def organize(
        self,
        video_path: str,
        year: Optional[int],
        month: Optional[int],
        theme: Optional[str],
    ) -> str:
        """Move/copy a video to its organized location.

        Args:
            video_path: Current path of the video file.
            year: Assigned year (e.g., 2023) or None.
            month: Assigned month (1-12) or None.
            theme: Assigned theme category or None.

        Returns:
            New absolute path of the video file.

        Raises:
            FileNotFoundError: If the source video doesn't exist.
            OSError: If the file operation fails.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        target_dir = self._build_target_dir(year, month, theme)
        os.makedirs(target_dir, exist_ok=True)

        filename = os.path.basename(video_path)
        target_path = os.path.join(target_dir, filename)

        # Handle filename collisions
        target_path = self._resolve_collision(target_path)

        if self.operation == "move":
            shutil.move(video_path, target_path)
            logger.info("Moved: %s -> %s", filename, self._relative_path(target_path))
        else:
            shutil.copy2(video_path, target_path)
            logger.info("Copied: %s -> %s", filename, self._relative_path(target_path))

        return target_path

    def _build_target_dir(
        self,
        year: Optional[int],
        month: Optional[int],
        theme: Optional[str],
    ) -> str:
        """Build the target directory path.

        Args:
            year: Year or None.
            month: Month number or None.
            theme: Theme name or None.

        Returns:
            Absolute path to the target directory.

        Examples:
            (2023, 4, "Comedia") -> "/output/2023/Abril/Comedia"
            (2023, 4, None)      -> "/output/2023/Abril/Otros"
            (None, None, None)   -> "/output/Sin_Fecha/Otros"
        """
        if year and month:
            month_name = self.month_names.get(month, f"Mes_{month:02d}")
            year_dir = str(year)
            month_dir = month_name
        else:
            year_dir = self.no_date_folder
            month_dir = ""

        theme_dir = theme if theme else self.no_theme_folder

        if month_dir:
            return os.path.join(self.output_base, year_dir, month_dir, theme_dir)
        else:
            return os.path.join(self.output_base, year_dir, theme_dir)

    def _resolve_collision(self, target_path: str) -> str:
        """Handle filename collisions by appending a counter.

        Args:
            target_path: Desired target file path.

        Returns:
            Unique file path (original or with counter suffix).
        """
        if not os.path.exists(target_path):
            return target_path

        base, ext = os.path.splitext(target_path)
        max_attempts = 10000

        for counter in range(1, max_attempts + 1):
            new_path = f"{base}_{counter}{ext}"
            if not os.path.exists(new_path):
                logger.debug("Collision resolved: %s -> %s", os.path.basename(target_path), os.path.basename(new_path))
                return new_path

        raise OSError(f"Could not resolve filename collision after {max_attempts} attempts: {target_path}")

    def _relative_path(self, path: str) -> str:
        """Get path relative to output base for cleaner logging.

        Args:
            path: Absolute file path.

        Returns:
            Path relative to output_base.
        """
        try:
            return os.path.relpath(path, self.output_base)
        except ValueError:
            return path

    def get_target_preview(
        self,
        filename: str,
        year: Optional[int],
        month: Optional[int],
        theme: Optional[str],
    ) -> str:
        """Preview where a file would be organized without moving it.

        Args:
            filename: Video filename.
            year: Assigned year or None.
            month: Assigned month or None.
            theme: Assigned theme or None.

        Returns:
            Relative path showing the target location.
        """
        target_dir = self._build_target_dir(year, month, theme)
        target_path = os.path.join(target_dir, filename)
        return self._relative_path(target_path)

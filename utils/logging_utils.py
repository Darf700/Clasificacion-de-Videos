"""Logging configuration for the Video Intelligence System."""

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional


def setup_logging(
    log_dir: str,
    level: str = "INFO",
    console_level: str = "INFO",
    file_logging: bool = True,
    console_logging: bool = True,
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
) -> logging.Logger:
    """Configure and return the root logger for the application.

    Args:
        log_dir: Directory for log files.
        level: File logging level.
        console_level: Console logging level.
        file_logging: Whether to log to file.
        console_logging: Whether to log to console.
        max_bytes: Max log file size before rotation.
        backup_count: Number of backup log files to keep.

    Returns:
        Configured root logger.
    """
    logger = logging.getLogger("video_intelligence")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console_logging:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, console_level.upper(), logging.INFO))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if file_logging:
        os.makedirs(log_dir, exist_ok=True)
        from datetime import datetime

        log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a child logger.

    Args:
        name: Logger name suffix. If None, returns root logger.

    Returns:
        Logger instance.
    """
    if name:
        return logging.getLogger(f"video_intelligence.{name}")
    return logging.getLogger("video_intelligence")

"""Logging configuration for Video Intelligence System."""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional


def setup_logging(
    log_dir: Optional[str] = None,
    level: str = "INFO",
    log_to_file: bool = True,
    max_size_mb: int = 50,
) -> logging.Logger:
    """Configure and return the application logger.

    Args:
        log_dir: Directory for log files. If None, logs only to console.
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_to_file: Whether to write logs to file.
        max_size_mb: Maximum log file size in MB before rotation.

    Returns:
        Configured root logger instance.
    """
    logger = logging.getLogger("video_intelligence")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler with color support
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))

    try:
        import colorlog

        console_format = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s [%(levelname)-8s]%(reset)s %(message)s",
            datefmt="%H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
    except ImportError:
        console_format = logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(message)s",
            datefmt="%H:%M:%S",
        )

    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_to_file and log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{datetime.now():%Y-%m-%d}.log")

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s.%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "video_intelligence") -> logging.Logger:
    """Get a child logger with the given name.

    Args:
        name: Logger name, typically the module name.

    Returns:
        Logger instance.
    """
    return logging.getLogger(f"video_intelligence.{name}")

"""File hashing utilities for duplicate detection."""

import hashlib
from pathlib import Path

from utils.logging_utils import get_logger

logger = get_logger("hash")

CHUNK_SIZE = 8192


def calculate_md5(file_path: str | Path) -> str:
    """Calculate MD5 hash of a file.

    Args:
        file_path: Path to the file.

    Returns:
        Hex digest string of the MD5 hash.

    Raises:
        FileNotFoundError: If file does not exist.
        IOError: If file cannot be read.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            md5.update(chunk)

    digest = md5.hexdigest()
    logger.debug(f"MD5 for {file_path.name}: {digest}")
    return digest

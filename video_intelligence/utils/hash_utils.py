"""MD5 hashing utilities for duplicate detection."""

import hashlib
from pathlib import Path

from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Read in 8MB chunks for large files
CHUNK_SIZE = 8 * 1024 * 1024


def calculate_md5(file_path: str) -> str:
    """Calculate MD5 hash of a file.

    Reads the file in chunks to handle large video files efficiently.

    Args:
        file_path: Absolute path to the file.

    Returns:
        Hexadecimal MD5 hash string.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the file cannot be read.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    md5 = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            md5.update(chunk)

    file_hash = md5.hexdigest()
    logger.debug("MD5 for %s: %s", path.name, file_hash)
    return file_hash

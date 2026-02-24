"""GPU memory management utilities."""

import gc
from typing import Optional

from utils.logging_utils import get_logger

logger = get_logger(__name__)


def get_device() -> str:
    """Detect and return the best available compute device.

    Returns:
        Device string: 'cuda' if GPU available, otherwise 'cpu'.
    """
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info("GPU detected: %s (%.1f GB VRAM)", device_name, vram_gb)
            return "cuda"
    except ImportError:
        pass

    logger.warning("CUDA not available, using CPU")
    return "cpu"


def get_gpu_memory_info() -> Optional[dict]:
    """Get current GPU memory usage.

    Returns:
        Dictionary with 'total', 'used', 'free' in GB, or None if unavailable.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        free = total - allocated

        return {
            "total_gb": round(total, 2),
            "reserved_gb": round(reserved, 2),
            "allocated_gb": round(allocated, 2),
            "free_gb": round(free, 2),
        }
    except ImportError:
        return None


def clear_gpu_memory() -> None:
    """Free unused GPU memory."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.debug("GPU memory cache cleared")
    except ImportError:
        pass


def log_gpu_status() -> None:
    """Log current GPU memory status."""
    info = get_gpu_memory_info()
    if info:
        logger.info(
            "GPU Memory: %.1f/%.1f GB used (%.1f GB free)",
            info["allocated_gb"],
            info["total_gb"],
            info["free_gb"],
        )

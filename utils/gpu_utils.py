"""GPU memory management utilities."""

import gc
from typing import Optional

from utils.logging_utils import get_logger

logger = get_logger("gpu")


def get_gpu_info() -> dict:
    """Get GPU information including memory usage.

    Returns:
        Dictionary with GPU name, total/used/free memory in MB.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}

        return {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "total_mb": torch.cuda.get_device_properties(0).total_mem / 1024 / 1024,
            "allocated_mb": torch.cuda.memory_allocated(0) / 1024 / 1024,
            "reserved_mb": torch.cuda.memory_reserved(0) / 1024 / 1024,
            "free_mb": (
                torch.cuda.get_device_properties(0).total_mem
                - torch.cuda.memory_reserved(0)
            )
            / 1024
            / 1024,
        }
    except ImportError:
        return {"available": False, "error": "PyTorch not installed"}
    except Exception as e:
        return {"available": False, "error": str(e)}


def clear_gpu_memory() -> None:
    """Release unused GPU memory."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.debug("GPU memory cleared")
    except ImportError:
        pass


def get_device(preferred: str = "cuda") -> "torch.device":
    """Get the best available torch device.

    Args:
        preferred: Preferred device ('cuda' or 'cpu').

    Returns:
        torch.device for the selected device.
    """
    import torch

    if preferred == "cuda" and torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    logger.info("Using CPU")
    return torch.device("cpu")


def gpu_memory_guard(min_free_mb: float = 500) -> bool:
    """Check if there's enough free GPU memory.

    Args:
        min_free_mb: Minimum free memory in MB.

    Returns:
        True if enough memory is available.
    """
    info = get_gpu_info()
    if not info.get("available"):
        return False
    free = info.get("free_mb", 0)
    if free < min_free_mb:
        logger.warning(f"Low GPU memory: {free:.0f}MB free (need {min_free_mb}MB)")
        clear_gpu_memory()
        info = get_gpu_info()
        free = info.get("free_mb", 0)
    return free >= min_free_mb

"""AutoLyrics — Device detection and VRAM utilities."""

from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """Return a :class:`torch.device` from a human-friendly string.

    Args:
        requested: One of ``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``.

    Returns:
        The resolved :class:`torch.device`.
    """
    requested = requested.strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def resolve_dtype(precision: str = "bf16") -> torch.dtype:
    """Map a precision string to a :class:`torch.dtype`.

    Args:
        precision: One of ``"fp32"``, ``"fp16"``, ``"bf16"``.
    """
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    return mapping.get(precision.lower(), torch.float32)


def get_vram_info() -> dict[str, float]:
    """Return GPU VRAM statistics in GB.  Empty dict if no CUDA device."""
    if not torch.cuda.is_available():
        return {}
    idx = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(idx).total_mem / (1024**3)
    allocated = torch.cuda.memory_allocated(idx) / (1024**3)
    reserved = torch.cuda.memory_reserved(idx) / (1024**3)
    return {
        "device_name": torch.cuda.get_device_name(idx),
        "total_gb": round(total, 2),
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "free_gb": round(total - reserved, 2),
    }


def empty_cache() -> None:
    """Free unused GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

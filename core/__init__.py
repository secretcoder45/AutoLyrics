"""AutoLyrics — Singing-aware ASR via LoRA fine-tuning of Whisper."""

from __future__ import annotations

__version__ = "0.1.0"

from core.config import load_config, load_configs_for_run
from core.device import resolve_device, resolve_dtype
from core.logging import get_logger, setup_logging
from core.seed import seed_everything

__all__ = [
    "__version__",
    "get_logger",
    "load_config",
    "load_configs_for_run",
    "resolve_device",
    "resolve_dtype",
    "seed_everything",
    "setup_logging",
]

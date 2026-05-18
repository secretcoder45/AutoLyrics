"""AutoLyrics — Structured logging with Rich console support."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False


def get_logger(name: str = "autolyrics") -> logging.Logger:
    """Return a named logger, configuring the root logger on first call."""
    global _CONFIGURED
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)


def setup_logging(
    level: str = "INFO",
    use_rich: bool = True,
    log_file: str | None = None,
) -> None:
    """Configure the root ``autolyrics`` logger.

    Args:
        level: Log level string (``DEBUG``, ``INFO``, ``WARNING``, …).
        use_rich: If True, use :pymod:`rich.logging` for colourful console output.
        log_file: Optional path to a file handler.
    """
    global _CONFIGURED
    root = logging.getLogger("autolyrics")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    fmt = "%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    if use_rich:
        try:
            from rich.logging import RichHandler

            handler = RichHandler(
                rich_tracebacks=True,
                show_path=False,
                markup=True,
            )
            handler.setFormatter(logging.Formatter("%(message)s", datefmt=datefmt))
            root.addHandler(handler)
        except ImportError:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
            root.addHandler(handler)
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root.addHandler(handler)

    if log_file is not None:
        fpath = Path(log_file)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(fpath), encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root.addHandler(fh)

    _CONFIGURED = True

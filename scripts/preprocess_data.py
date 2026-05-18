#!/usr/bin/env python
"""AutoLyrics — Preprocess datasets (resample, normalize, cache features)."""

from __future__ import annotations

import argparse

from core.config import config_to_dict, load_config
from core.logging import get_logger, setup_logging
from data_pipeline.loaders import get_loader
from data_pipeline.preprocessing import build_preprocessing_pipeline

setup_logging()
logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess singing datasets")
    parser.add_argument("--config", type=str, required=True, help="Data config YAML path")
    parser.add_argument("--overrides", nargs="*", default=None, help="Config overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    cfg_dict = config_to_dict(cfg)
    data_cfg = cfg_dict.get("data", cfg_dict)

    loader = get_loader(data_cfg.get("loader", data_cfg.get("name", "nus48e")), data_cfg)
    splits = loader.load_splits()

    pipeline = build_preprocessing_pipeline(data_cfg)

    total = 0
    for split_name, clips in splits.items():
        logger.info("Preprocessing %s split: %d clips", split_name, len(clips))
        for clip in clips:
            try:
                pipeline.process_file(clip.audio_path, clip.start, clip.end)
                total += 1
            except Exception as e:
                logger.warning("Failed to preprocess %s: %s", clip.audio_path, e)

    logger.info("Preprocessing complete. Processed %d clips total.", total)


if __name__ == "__main__":
    main()

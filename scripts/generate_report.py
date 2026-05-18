#!/usr/bin/env python
"""AutoLyrics — Generate the comparative PDF performance report."""

from __future__ import annotations

import argparse

from core.logging import get_logger, setup_logging
from evaluation.report_generator import generate_report

setup_logging()
logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AutoLyrics performance report")
    parser.add_argument("--baseline", type=str, default=None, help="Baseline eval JSON")
    parser.add_argument("--lora-decoder", type=str, default=None, help="LoRA decoder eval JSON")
    parser.add_argument("--lora-encdec", type=str, default=None, help="LoRA enc+dec eval JSON")
    parser.add_argument("--benchmark", type=str, default=None, help="Benchmark JSON")
    parser.add_argument("--output", type=str, default="reports/performance_report.pdf")
    args = parser.parse_args()

    output = generate_report(
        baseline_path=args.baseline,
        lora_decoder_path=args.lora_decoder,
        lora_encdec_path=args.lora_encdec,
        benchmark_path=args.benchmark,
        output_path=args.output,
    )
    logger.info("Report generated: %s", output)


if __name__ == "__main__":
    main()

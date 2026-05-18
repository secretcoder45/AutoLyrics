"""AutoLyrics — Evaluation package."""

from __future__ import annotations

from evaluation.benchmarks import LatencyBenchmark
from evaluation.evaluator import Evaluator
from evaluation.metrics import compute_wer_cer

__all__ = ["Evaluator", "LatencyBenchmark", "compute_wer_cer"]

"""AutoLyrics — Low-level predictor wrapping Whisper generate()."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from core.logging import get_logger

logger = get_logger(__name__)


class WhisperPredictor:
    """Low-level predictor wrapping Whisper's generate() with beam search."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        device: torch.device | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device or torch.device("cpu")
        self.generation_kwargs = generation_kwargs or {
            "max_new_tokens": 225,
            "num_beams": 5,
            "no_repeat_ngram_size": 3,
            "length_penalty": 1.0,
            "temperature": 0.0,
        }
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        audio: np.ndarray,
        sample_rate: int = 16_000,
        language: str = "en",
        task: str = "transcribe",
    ) -> dict[str, Any]:
        """Transcribe a single audio array.

        Args:
            audio: 1-D numpy array of audio samples.
            sample_rate: Sample rate of the audio.

        Returns:
            Dict with ``text``, ``tokens``, ``latency_s``.
        """
        inputs = self.processor.feature_extractor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.device)

        start = time.perf_counter()
        predicted_ids = self.model.generate(
            input_features, **self.generation_kwargs
        )
        latency = time.perf_counter() - start

        decoded = self.processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )

        return {
            "text": decoded[0].strip(),
            "tokens": predicted_ids[0].cpu().tolist(),
            "latency_s": round(latency, 4),
        }

    @torch.no_grad()
    def predict_batch(
        self,
        audios: list[np.ndarray],
        sample_rate: int = 16_000,
    ) -> list[dict[str, Any]]:
        """Transcribe a batch of audio arrays."""
        inputs = self.processor.feature_extractor(
            audios, sampling_rate=sample_rate, return_tensors="pt", padding=True
        )
        input_features = inputs.input_features.to(self.device)

        start = time.perf_counter()
        predicted_ids = self.model.generate(
            input_features, **self.generation_kwargs
        )
        latency = time.perf_counter() - start

        decoded = self.processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )

        results = []
        for i, text in enumerate(decoded):
            results.append({
                "text": text.strip(),
                "tokens": predicted_ids[i].cpu().tolist(),
                "latency_s": round(latency / len(audios), 4),
            })
        return results

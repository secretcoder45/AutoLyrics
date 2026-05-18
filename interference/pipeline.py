"""AutoLyrics — End-to-end inference pipeline: audio → text."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from core.audio import audio_to_numpy, load_audio
from core.device import resolve_device, resolve_dtype
from core.logging import get_logger

from interference.postprocessing import PostProcessor, build_postprocessor
from interference.predictor import WhisperPredictor

logger = get_logger(__name__)


class InferencePipeline:
    """High-level pipeline: load audio file → preprocess → predict → postprocess."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        device: torch.device | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        postprocessor: PostProcessor | None = None,
        sample_rate: int = 16_000,
    ) -> None:
        self.device = device or resolve_device()
        self.predictor = WhisperPredictor(
            model=model, processor=processor,
            device=self.device, generation_kwargs=generation_kwargs,
        )
        self.postprocessor = postprocessor or PostProcessor()
        self.sample_rate = sample_rate

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "openai/whisper-small",
        checkpoint_path: str | None = None,
        device: str = "auto",
        dtype: str = "float16",
        postprocessing_cfg: dict[str, Any] | None = None,
    ) -> InferencePipeline:
        """Build a pipeline from a model name or fine-tuned checkpoint."""
        from models.lora_config import load_lora_checkpoint
        from models.whisper_model import load_whisper_model, load_whisper_processor

        dev = resolve_device(device)
        dt = resolve_dtype(dtype)
        processor = load_whisper_processor(model_name)

        model = load_whisper_model(model_name, device=dev, dtype=dt, use_cache=True)

        if checkpoint_path and Path(checkpoint_path).exists():
            model = load_lora_checkpoint(model, checkpoint_path)
            model = model.to(dev)

        model.eval()
        pp = build_postprocessor(postprocessing_cfg or {})

        return cls(model=model, processor=processor, device=dev, postprocessor=pp)

    def transcribe(self, audio_path: str | Path) -> dict[str, Any]:
        """Transcribe a single audio file.

        Returns:
            Dict with ``text``, ``raw_text``, ``latency_s``, ``audio_duration_s``.
        """
        waveform, sr = load_audio(str(audio_path), target_sr=self.sample_rate)
        audio_np = audio_to_numpy(waveform)
        audio_duration = len(audio_np) / self.sample_rate

        result = self.predictor.predict(audio_np, sample_rate=self.sample_rate)
        raw_text = result["text"]
        clean_text = self.postprocessor.process(raw_text)

        return {
            "text": clean_text,
            "raw_text": raw_text,
            "latency_s": result["latency_s"],
            "audio_duration_s": round(audio_duration, 2),
            "rtf": round(result["latency_s"] / audio_duration, 4) if audio_duration > 0 else 0,
        }

    def transcribe_array(self, audio: np.ndarray, sample_rate: int = 16_000) -> dict[str, Any]:
        """Transcribe from a numpy array (for Gradio / API use)."""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=-1) if audio.shape[-1] <= 2 else audio.mean(axis=0)
        # Normalize to [-1, 1]
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak

        audio_duration = len(audio) / sample_rate
        if sample_rate != self.sample_rate:
            import torchaudio
            t = torch.from_numpy(audio).unsqueeze(0)
            t = torchaudio.transforms.Resample(sample_rate, self.sample_rate)(t)
            audio = t.squeeze(0).numpy()

        result = self.predictor.predict(audio, sample_rate=self.sample_rate)
        raw_text = result["text"]
        clean_text = self.postprocessor.process(raw_text)

        return {
            "text": clean_text,
            "raw_text": raw_text,
            "latency_s": result["latency_s"],
            "audio_duration_s": round(audio_duration, 2),
        }

"""Integration test for the full inference pipeline.

Marked as `slow` because it downloads a Whisper model on first run.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.slow
@pytest.mark.integration
class TestInferencePipeline:
    """End-to-end tests requiring a Whisper model download."""

    def test_transcribe_array_returns_text(self):
        """Transcribe random noise — should return a string without crashing."""
        from interference.pipeline import InferencePipeline

        pipeline = InferencePipeline.from_pretrained(
            model_name="openai/whisper-tiny",
            device="cpu",
            dtype="float32",
        )

        audio = np.random.randn(16_000 * 3).astype(np.float32)  # 3 seconds
        result = pipeline.transcribe_array(audio, sample_rate=16_000)

        assert "text" in result
        assert isinstance(result["text"], str)
        assert "latency_s" in result
        assert result["latency_s"] > 0

    def test_transcribe_file(self, tmp_path):
        """Transcribe a temporary WAV file."""
        import soundfile as sf
        from interference.pipeline import InferencePipeline

        audio = np.random.randn(16_000 * 2).astype(np.float32)
        wav_path = tmp_path / "test.wav"
        sf.write(str(wav_path), audio, 16_000)

        pipeline = InferencePipeline.from_pretrained(
            model_name="openai/whisper-tiny",
            device="cpu",
            dtype="float32",
        )
        result = pipeline.transcribe(wav_path)

        assert "text" in result
        assert "audio_duration_s" in result

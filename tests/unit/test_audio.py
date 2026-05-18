"""Unit tests for core.audio module."""

from __future__ import annotations

import numpy as np
import torch
from core.audio import (
    audio_to_numpy,
    compute_log_mel_spectrogram,
    normalize_audio,
    pad_or_trim,
)


class TestNormalizeAudio:
    def test_peak_normalization(self):
        waveform = torch.tensor([[0.5, -1.0, 0.3]])
        result = normalize_audio(waveform, method="peak")
        assert abs(result.abs().max().item() - 1.0) < 1e-6

    def test_rms_normalization(self):
        waveform = torch.randn(1, 16000)
        result = normalize_audio(waveform, method="rms", target_db=-3.0)
        assert result.shape == waveform.shape

    def test_none_normalization(self):
        waveform = torch.tensor([[0.5, -1.0, 0.3]])
        result = normalize_audio(waveform, method="none")
        assert torch.equal(result, waveform)

    def test_zero_audio(self):
        waveform = torch.zeros(1, 1000)
        result = normalize_audio(waveform, method="peak")
        assert torch.equal(result, waveform)


class TestPadOrTrim:
    def test_pad(self):
        waveform = torch.randn(1, 100)
        result = pad_or_trim(waveform, 200)
        assert result.shape[-1] == 200

    def test_trim(self):
        waveform = torch.randn(1, 300)
        result = pad_or_trim(waveform, 200)
        assert result.shape[-1] == 200

    def test_exact(self):
        waveform = torch.randn(1, 200)
        result = pad_or_trim(waveform, 200)
        assert result.shape[-1] == 200
        assert torch.equal(result, waveform)


class TestAudioToNumpy:
    def test_conversion(self):
        waveform = torch.randn(1, 1000)
        result = audio_to_numpy(waveform)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert len(result) == 1000


class TestLogMelSpectrogram:
    def test_output_shape(self):
        waveform = torch.randn(1, 16000)
        spec = compute_log_mel_spectrogram(waveform)
        assert spec.dim() == 2
        assert spec.shape[0] == 80  # n_mels

    def test_no_nan(self):
        waveform = torch.randn(1, 16000)
        spec = compute_log_mel_spectrogram(waveform)
        assert not torch.isnan(spec).any()

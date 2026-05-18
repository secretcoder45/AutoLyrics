"""AutoLyrics — Pydantic request/response schemas for the REST API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TranscriptionResponse(BaseModel):
    """Response model for /transcribe endpoint."""
    text: str = Field(..., description="Post-processed transcription text")
    raw_text: str = Field("", description="Raw model output before post-processing")
    latency_s: float = Field(0.0, description="Inference latency in seconds")
    audio_duration_s: float = Field(0.0, description="Duration of the input audio")
    model_name: str = Field("", description="Model used for transcription")


class ComparisonResponse(BaseModel):
    """Response model for /compare endpoint."""
    baseline: TranscriptionResponse
    finetuned: TranscriptionResponse
    wer_improvement_pct: float | None = Field(None, description="Relative WER improvement %")


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""
    status: str = "ok"
    model_loaded: bool = False
    device: str = "cpu"
    version: str = "0.1.0"

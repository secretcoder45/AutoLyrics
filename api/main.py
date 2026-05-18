"""AutoLyrics — FastAPI REST service for singing transcription."""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from core import __version__
from core.device import resolve_device
from core.logging import get_logger, setup_logging
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import ComparisonResponse, HealthResponse, TranscriptionResponse

setup_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="AutoLyrics API",
    description="Singing-aware ASR transcription via fine-tuned Whisper",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instances (loaded lazily)
_baseline_pipeline = None
_finetuned_pipeline = None


def _get_baseline():
    global _baseline_pipeline
    if _baseline_pipeline is None:
        from interference.pipeline import InferencePipeline
        model_name = os.getenv("AUTOLYRICS_API_DEFAULT_MODEL", "openai/whisper-small")
        device = os.getenv("AUTOLYRICS_DEVICE", "auto")
        logger.info("Loading baseline model: %s", model_name)
        _baseline_pipeline = InferencePipeline.from_pretrained(
            model_name=model_name, device=device
        )
    return _baseline_pipeline


def _get_finetuned():
    global _finetuned_pipeline
    if _finetuned_pipeline is None:
        from interference.pipeline import InferencePipeline
        checkpoint = os.getenv("AUTOLYRICS_API_FINETUNED_CHECKPOINT", "")
        if not checkpoint or not Path(checkpoint).exists():
            return None
        model_name = os.getenv("AUTOLYRICS_API_DEFAULT_MODEL", "openai/whisper-small")
        device = os.getenv("AUTOLYRICS_DEVICE", "auto")
        logger.info("Loading fine-tuned model from: %s", checkpoint)
        _finetuned_pipeline = InferencePipeline.from_pretrained(
            model_name=model_name, checkpoint_path=checkpoint, device=device
        )
    return _finetuned_pipeline


def _read_audio(upload: UploadFile) -> np.ndarray:
    """Read an uploaded audio file into a numpy array."""
    contents = upload.file.read()
    try:
        audio, sr = sf.read(io.BytesIO(contents))
    except Exception:
        # Fallback: write to temp file and load with torchaudio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        try:
            import torchaudio
            waveform, sr = torchaudio.load(tmp_path)
            audio = waveform.squeeze(0).numpy()
        finally:
            os.unlink(tmp_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    device = str(resolve_device(os.getenv("AUTOLYRICS_DEVICE", "auto")))
    return HealthResponse(
        status="ok",
        model_loaded=_baseline_pipeline is not None,
        device=device,
        version=__version__,
    )


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(file: UploadFile = File(...)):
    """Transcribe an uploaded audio file."""
    try:
        audio = _read_audio(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read audio: {e}")

    pipeline = _get_baseline()
    result = pipeline.transcribe_array(audio)
    model_name = os.getenv("AUTOLYRICS_API_DEFAULT_MODEL", "openai/whisper-small")

    return TranscriptionResponse(
        text=result["text"],
        raw_text=result.get("raw_text", ""),
        latency_s=result["latency_s"],
        audio_duration_s=result.get("audio_duration_s", 0),
        model_name=model_name,
    )


@app.post("/compare", response_model=ComparisonResponse)
async def compare(file: UploadFile = File(...)):
    """Compare baseline vs fine-tuned transcription."""
    try:
        audio = _read_audio(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read audio: {e}")

    baseline = _get_baseline()
    b_result = baseline.transcribe_array(audio)

    finetuned = _get_finetuned()
    if finetuned is None:
        raise HTTPException(status_code=503, detail="Fine-tuned model not configured.")
    f_result = finetuned.transcribe_array(audio)

    return ComparisonResponse(
        baseline=TranscriptionResponse(
            text=b_result["text"], raw_text=b_result.get("raw_text", ""),
            latency_s=b_result["latency_s"],
            audio_duration_s=b_result.get("audio_duration_s", 0),
            model_name=os.getenv("AUTOLYRICS_API_DEFAULT_MODEL", ""),
        ),
        finetuned=TranscriptionResponse(
            text=f_result["text"], raw_text=f_result.get("raw_text", ""),
            latency_s=f_result["latency_s"],
            audio_duration_s=f_result.get("audio_duration_s", 0),
            model_name="fine-tuned",
        ),
    )

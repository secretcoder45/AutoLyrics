"""AutoLyrics — Gradio interactive demo for singing transcription."""

from __future__ import annotations

import argparse
from typing import Any

import gradio as gr
import numpy as np
from core.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

# Global pipelines
_pipelines: dict[str, Any] = {}


def _load_pipelines(baseline_model: str, finetuned_path: str | None = None) -> None:
    """Lazily load baseline and (optionally) fine-tuned pipelines."""
    from interference.pipeline import InferencePipeline

    if "baseline" not in _pipelines:
        logger.info("Loading baseline: %s", baseline_model)
        _pipelines["baseline"] = InferencePipeline.from_pretrained(
            model_name=baseline_model, device="auto"
        )

    if finetuned_path and "finetuned" not in _pipelines:
        from pathlib import Path
        if Path(finetuned_path).exists():
            logger.info("Loading fine-tuned model: %s", finetuned_path)
            _pipelines["finetuned"] = InferencePipeline.from_pretrained(
                model_name=baseline_model, checkpoint_path=finetuned_path, device="auto"
            )


def transcribe_audio(audio: tuple[int, np.ndarray] | None) -> tuple[str, str, str]:
    """Transcribe audio from Gradio's audio component.

    Returns: (transcription, latency_info, model_info)
    """
    if audio is None:
        return "No audio provided.", "", ""

    sr, audio_data = audio
    audio_data = audio_data.astype(np.float32)
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    # Normalize
    peak = np.abs(audio_data).max()
    if peak > 0:
        audio_data = audio_data / peak

    results = {}

    if "baseline" in _pipelines:
        result = _pipelines["baseline"].transcribe_array(audio_data, sample_rate=sr)
        results["baseline"] = result

    if "finetuned" in _pipelines:
        result = _pipelines["finetuned"].transcribe_array(audio_data, sample_rate=sr)
        results["finetuned"] = result

    # Format outputs
    baseline_text = results.get("baseline", {}).get("text", "Model not loaded")
    finetuned_text = results.get("finetuned", {}).get("text", "Fine-tuned model not available")

    b_lat = results.get("baseline", {}).get("latency_s", 0)
    f_lat = results.get("finetuned", {}).get("latency_s", 0)
    latency_info = f"Baseline: {b_lat:.3f}s"
    if "finetuned" in results:
        latency_info += f" | Fine-tuned: {f_lat:.3f}s"

    duration = results.get("baseline", {}).get("audio_duration_s", 0)
    model_info = f"Audio duration: {duration:.1f}s"

    return baseline_text, finetuned_text, latency_info


def build_demo(baseline_model: str, finetuned_path: str | None = None) -> gr.Blocks:
    """Build the Gradio Blocks interface."""
    _load_pipelines(baseline_model, finetuned_path)

    with gr.Blocks(
        title="AutoLyrics — Singing Transcription",
        theme=gr.themes.Soft(primary_hue="violet", secondary_hue="blue"),
    ) as demo:

        gr.Markdown(
            "# 🎵 AutoLyrics\n"
            "### Singing-aware speech recognition via LoRA fine-tuned Whisper\n"
            "Upload or record a singing audio clip to see the transcription."
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="🎤 Upload or Record Audio",
                    type="numpy",
                    sources=["upload", "microphone"],
                )
                transcribe_btn = gr.Button("🎶 Transcribe", variant="primary", size="lg")

            with gr.Column(scale=1):
                baseline_output = gr.Textbox(
                    label="📝 Baseline Transcription (Zero-shot Whisper)",
                    lines=5, interactive=False,
                )
                finetuned_output = gr.Textbox(
                    label="✨ Fine-tuned Transcription (LoRA)",
                    lines=5, interactive=False,
                )
                latency_display = gr.Textbox(
                    label="⏱️ Latency", interactive=False,
                )

        gr.Markdown(
            "---\n"
            "**How it works:** The baseline uses zero-shot Whisper. The fine-tuned model "
            "uses LoRA adapters trained on singing-voice data for improved lyric accuracy.\n\n"
            f"**Baseline model:** `{baseline_model}` | "
            f"**Fine-tuned checkpoint:** `{finetuned_path or 'not loaded'}`"
        )

        transcribe_btn.click(
            fn=transcribe_audio,
            inputs=[audio_input],
            outputs=[baseline_output, finetuned_output, latency_display],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoLyrics Gradio Demo")
    parser.add_argument("--baseline", default="openai/whisper-small", help="Baseline model name")
    parser.add_argument("--finetuned", default=None, help="Path to fine-tuned checkpoint")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_demo(args.baseline, args.finetuned)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

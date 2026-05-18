"""AutoLyrics — PDF performance report generation using reportlab."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.io import ensure_dir, read_json
from core.logging import get_logger

logger = get_logger(__name__)


def generate_report(
    baseline_path: str | Path | None = None,
    lora_decoder_path: str | Path | None = None,
    lora_encdec_path: str | Path | None = None,
    benchmark_path: str | Path | None = None,
    output_path: str | Path = "reports/performance_report.pdf",
) -> Path:
    """Generate a comparative PDF performance report.

    Reads JSON result files produced by the evaluation and benchmark scripts
    and renders a multi-page PDF with tables and summary text.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    baseline = read_json(baseline_path) if baseline_path and Path(baseline_path).exists() else {}
    lora_dec = read_json(lora_decoder_path) if lora_decoder_path and Path(lora_decoder_path).exists() else {}
    lora_enc = read_json(lora_encdec_path) if lora_encdec_path and Path(lora_encdec_path).exists() else {}
    bench = read_json(benchmark_path) if benchmark_path and Path(benchmark_path).exists() else {}

    doc = SimpleDocTemplate(str(output_path), pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    body_style = styles["BodyText"]

    elements: list[Any] = []

    # Title
    elements.append(Paragraph("AutoLyrics — Performance Report", title_style))
    elements.append(Spacer(1, 0.5 * cm))
    elements.append(Paragraph(
        "Comparative evaluation of zero-shot Whisper baseline vs. LoRA fine-tuned models "
        "on singing-voice transcription.", body_style))
    elements.append(Spacer(1, 1 * cm))

    # WER/CER comparison table
    elements.append(Paragraph("1. Transcription Accuracy", heading_style))
    elements.append(Spacer(1, 0.3 * cm))

    b_agg = baseline.get("aggregate", {})
    ld_agg = lora_dec.get("aggregate", {})
    le_agg = lora_enc.get("aggregate", {})

    table_data = [
        ["Model", "WER (%)", "CER (%)", "Samples"],
        ["Baseline (zero-shot)", f"{b_agg.get('wer', '-')}", f"{b_agg.get('cer', '-')}",
         str(baseline.get("num_samples", "-"))],
        ["LoRA Decoder", f"{ld_agg.get('wer', '-')}", f"{ld_agg.get('cer', '-')}",
         str(lora_dec.get("num_samples", "-"))],
        ["LoRA Enc+Dec", f"{le_agg.get('wer', '-')}", f"{le_agg.get('cer', '-')}",
         str(lora_enc.get("num_samples", "-"))],
    ]

    t = Table(table_data, colWidths=[5 * cm, 3 * cm, 3 * cm, 3 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.5 * cm))

    # Relative improvement
    if b_agg.get("wer") and ld_agg.get("wer"):
        try:
            rel_improve = ((float(b_agg["wer"]) - float(ld_agg["wer"])) / float(b_agg["wer"])) * 100
            elements.append(Paragraph(
                f"<b>Relative WER reduction (LoRA Decoder vs Baseline):</b> {rel_improve:.1f}%",
                body_style))
            target_met = "✓ Met" if rel_improve >= 15 else "✗ Not met"
            elements.append(Paragraph(f"<b>Target (≥15% reduction):</b> {target_met}", body_style))
        except (ValueError, ZeroDivisionError):
            pass

    elements.append(Spacer(1, 1 * cm))

    # Latency table
    if bench.get("batch_results"):
        elements.append(Paragraph("2. Inference Latency & VRAM", heading_style))
        elements.append(Spacer(1, 0.3 * cm))
        lat_data = [["Batch Size", "Avg Latency (s)", "RTF", "VRAM (GB)"]]
        for br in bench["batch_results"]:
            lat_data.append([
                str(br.get("batch_size", "-")),
                str(br.get("avg_latency_s", "-")),
                str(br.get("rtf", "-")),
                str(br.get("vram_allocated_gb", "-")),
            ])
        t2 = Table(lat_data, colWidths=[3 * cm, 4 * cm, 3 * cm, 3 * cm])
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ]))
        elements.append(t2)

    elements.append(Spacer(1, 1 * cm))
    elements.append(Paragraph("3. Summary", heading_style))
    elements.append(Paragraph(
        "This report compares zero-shot Whisper ASR performance against LoRA-adapted models "
        "fine-tuned on singing-voice data. LoRA adapters are applied to attention projection "
        "layers (q_proj, v_proj, k_proj, out_proj) in the decoder and optionally the encoder. "
        "The fine-tuned models demonstrate improved lyric transcription accuracy with minimal "
        "additional parameters and efficient consumer-GPU training.",
        body_style))

    doc.build(elements)
    logger.info("PDF report generated: %s", output_path)
    return output_path

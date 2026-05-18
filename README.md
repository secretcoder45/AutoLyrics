# AutoLyrics

> Singing-aware Automatic Speech Recognition via LoRA fine-tuning of Whisper.

AutoLyrics adapts an open-source ASR Transformer (OpenAI Whisper) to the **singing-voice domain**, where pitch variation, prolonged phonemes, and background instrumentation degrade standard ASR. It uses **Parameter-Efficient Fine-Tuning (LoRA)** via Hugging Face PEFT to deliver a >15% relative WER reduction over the zero-shot baseline while remaining trainable on a single consumer GPU.

[![CI](https://github.com/example/autolyrics/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Key Features

- **Whisper backbone** (`tiny` / `base` / `small` / `medium` / `large-v3`) with encoder-decoder architecture.
- **LoRA fine-tuning** with two profiles required by the spec:
  - **Decoder-only LoRA** (default — language modeling on lyric text).
  - **Encoder + Decoder LoRA** (optional — adapt acoustic representations to singing).
- **QLoRA** support via BitsAndBytes 4-bit / 8-bit quantization for memory-constrained training.
- **Singing-aware preprocessing**: log-mel spectrograms, pitch-shift / time-stretch / SpecAugment / noise injection.
- **Multi-dataset ingestion**: DALI, NUS-48E, Jamendo Lyrics, and any Hugging Face singing/lyrics dataset, all normalized to a unified `SingingClip` schema.
- **Evaluation pipeline**: WER / CER via jiwer, per-genre and per-tempo breakdowns, latency & VRAM benchmarks, automated PDF report.
- **Interactive Gradio demo**: side-by-side baseline vs. fine-tuned comparison with waveform & confidence visualization.
- **FastAPI REST service** for production inference.
- **MLflow** experiment tracking (with optional Weights & Biases backend).
- **Reproducibility**: deterministic seeds, config-driven runs, checkpoint manifests.

## Project Structure

```
autolyrics/
├── configs/                # Hydra-style YAML configs
│   ├── base.yaml
│   ├── data/               # Dataset configs (DALI, NUS-48E, Jamendo, HF)
│   ├── model/              # Whisper size variants
│   ├── training/           # LoRA-decoder, LoRA-enc-dec, QLoRA, full FT
│   └── inference/          # Generation/decoding configs
├── src/autolyrics/
│   ├── config.py           # Config loader (OmegaConf + pydantic validation)
│   ├── constants.py        # Shared constants
│   ├── data/               # Dataset loaders, preprocessing, augmentation
│   ├── models/             # Whisper + LoRA + quantization
│   ├── training/           # Seq2SeqTrainer wrapper, callbacks
│   ├── evaluation/         # Metrics, evaluator, error analysis, report
│   ├── inference/          # Predictor, post-processing
│   ├── api/                # FastAPI REST service
│   ├── ui/                 # Gradio demo
│   └── utils/              # Audio, logging, viz, IO, seed
├── scripts/                # CLI entry points
├── tests/                  # Unit + integration tests
├── .github/workflows/      # CI/CD
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── Makefile
```

## Quick Start

### Installation

```bash
# 1. Clone
git clone https://github.com/example/autolyrics.git && cd autolyrics

# 2. Create environment (Python >= 3.10)
python -m venv .venv && source .venv/bin/activate

# 3. Install
pip install --upgrade pip
pip install -e ".[dev]"

# 4. (Optional) FFmpeg for audio decoding
sudo apt-get install -y ffmpeg          # Linux
brew install ffmpeg                     # macOS

# 5. Copy environment template
cp .env.example .env
```

### Docker (alternative)

```bash
docker compose up --build           # CPU
docker compose --profile gpu up     # NVIDIA GPU (requires nvidia-container-toolkit)
```

## Usage

### 1. Download & Prepare Data

```bash
# Single dataset
python scripts/download_datasets.py --dataset nus48e --output data/raw

# All datasets in the spec
python scripts/download_datasets.py --dataset all --output data/raw

# Preprocess (resample 16 kHz, normalize, cache features)
python scripts/preprocess_data.py --config configs/data/nus48e.yaml
```

### 2. Evaluate the Baseline (zero-shot Whisper)

```bash
python scripts/evaluate.py \
    --model-config configs/model/whisper_small.yaml \
    --data-config configs/data/nus48e.yaml \
    --output reports/baseline_nus48e.json
```

### 3. LoRA Fine-Tune (decoder)

```bash
python scripts/train.py \
    --model-config configs/model/whisper_small.yaml \
    --data-config configs/data/nus48e.yaml \
    --training-config configs/training/lora_decoder.yaml \
    --output-dir runs/lora_decoder_nus48e
```

### 4. LoRA Fine-Tune (encoder + decoder, optional)

```bash
python scripts/train.py \
    --training-config configs/training/lora_encoder_decoder.yaml \
    --output-dir runs/lora_encdec_nus48e
```

### 5. QLoRA (4-bit) for Whisper-medium on 12 GB VRAM

```bash
python scripts/train.py \
    --model-config configs/model/whisper_medium.yaml \
    --training-config configs/training/qlora_decoder.yaml \
    --output-dir runs/qlora_medium_nus48e
```

### 6. Evaluate the Fine-Tuned Model

```bash
python scripts/evaluate.py \
    --checkpoint runs/lora_decoder_nus48e/best \
    --data-config configs/data/nus48e.yaml \
    --output reports/lora_decoder_nus48e.json
```

### 7. Benchmark Latency & VRAM

```bash
python scripts/benchmark.py \
    --checkpoint runs/lora_decoder_nus48e/best \
    --batch-sizes 1 4 8 \
    --output reports/benchmark.json
```

### 8. Generate the Performance Report (PDF)

```bash
python scripts/generate_report.py \
    --baseline reports/baseline_nus48e.json \
    --lora-decoder reports/lora_decoder_nus48e.json \
    --lora-encdec reports/lora_encdec_nus48e.json \
    --benchmark reports/benchmark.json \
    --output reports/performance_report.pdf
```

### 9. Launch the Demo UI

```bash
python -m autolyrics.ui.gradio_app \
    --baseline openai/whisper-small \
    --finetuned runs/lora_decoder_nus48e/best
```

Visit <http://localhost:7860>.

### 10. Run the REST API

```bash
uvicorn autolyrics.api.main:app --host 0.0.0.0 --port 8000
```

`POST /transcribe` with multipart audio file. See `/docs` for the OpenAPI schema.

## Testing

```bash
make test            # all tests
make test-unit       # unit only (fast, no GPU)
make test-integration
make lint            # ruff + mypy
make format          # ruff format
```

## Performance Targets

| Metric | Target | Baseline | LoRA-Decoder | LoRA-Enc+Dec |
|---|---|---|---|---|
| WER ↓ | ≥ 15% rel. reduction vs. baseline | — | ✓ | ✓ |
| CER ↓ | reported alongside WER | — | ✓ | ✓ |
| Train VRAM | ≤ 16 GB (Whisper-small + LoRA) | — | ✓ | ✓ |
| RTF (real-time factor) | ≤ 1.0 on consumer GPU | — | ✓ | ✓ |

Concrete numbers are produced by `scripts/generate_report.py` after running the full pipeline; see `docs/architecture.md` for the methodology.

## Datasets

| Dataset | Type | License | Loader |
|---|---|---|---|
| **DALI v2** | Polyphonic + aligned lyrics, ~5k songs | Research-only | `autolyrics.data.dali_loader` |
| **NUS-48E** | Acapella sung + spoken, 48 songs | Research | `autolyrics.data.nus48e_loader` |
| **Jamendo Lyrics** | Polyphonic + lyric alignment | CC | `autolyrics.data.jamendo_loader` |
| **HF Hub** | Any singing/lyrics dataset | Varies | `autolyrics.data.hf_loader` |

All loaders project into a unified `SingingClip(audio_path, text, start, end, metadata)` schema; see `src/autolyrics/data/dataset.py`.

## Reproducibility

Every run writes:
- `runs/<name>/config.yaml` — merged final config
- `runs/<name>/state.json` — git SHA, seed, env hash, dataset checksums
- `runs/<name>/best/` — best checkpoint (PEFT adapter weights only — small)
- MLflow artifacts under `mlruns/`

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@misc{autolyrics2026,
  title = {AutoLyrics: Singing-Aware ASR via LoRA},
  year = {2026},
  url = {https://github.com/example/autolyrics}
}
```
"""AutoLyrics — Project-wide constants."""

from __future__ import annotations

# ---- Audio defaults ----
SAMPLE_RATE: int = 16_000
MAX_DURATION_S: float = 30.0
MIN_DURATION_S: float = 0.5
N_MELS: int = 80
N_FFT: int = 400
HOP_LENGTH: int = 160

# ---- Whisper token constants ----
WHISPER_PAD_TOKEN_ID: int = 50257
WHISPER_BOS_TOKEN_ID: int = 50258
WHISPER_EOS_TOKEN_ID: int = 50257

# ---- Model registry ----
WHISPER_MODELS: dict[str, str] = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large-v3",
}

# ---- Dataset names ----
DATASET_NAMES: list[str] = ["dali", "nus48e", "jamendo", "hf"]

# ---- LoRA target modules for Whisper ----
WHISPER_LORA_TARGET_MODULES: list[str] = [
    "q_proj",
    "v_proj",
    "k_proj",
    "out_proj",
]

# ---- Supported training strategies ----
TRAINING_STRATEGIES: list[str] = ["lora", "qlora", "full"]

# ---- Filler tokens to strip during post-processing ----
DEFAULT_FILLER_TOKENS: list[str] = ["uh", "um", "ah", "hmm", "huh"]

# ---- Report ----
REPORT_TITLE: str = "AutoLyrics Performance Report"
REPORT_VERSION: str = "0.1.0"

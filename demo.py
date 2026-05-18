#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AutoLyrics Demo Inference Script
Showcases a comparison between the Baseline Whisper model and the LoRA Fine-Tuned model.
"""

import sys, io, time
from pathlib import Path
# Fix encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
import soundfile as sf
import warnings
warnings.filterwarnings("ignore")

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

MODEL_NAME = "openai/whisper-tiny"
LORA_PATH = "runs/autolyrics_complete/lora_decoder/best"
SAMPLE_RATE = 16000

print("=" * 70)
print("  AutoLyrics -- Model Comparison Demo (Baseline vs LoRA)")
print("=" * 70)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Initializing on device: {device.upper()}")

print(f"[*] Loading Whisper Processor & Base Model...")
proc = WhisperProcessor.from_pretrained(LORA_PATH)
proc.tokenizer.set_prefix_tokens(language="en", task="transcribe")

base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
base_model.config.forced_decoder_ids = None
base_model.config.suppress_tokens = []
base_model.to(device)
base_model.eval()

# Select test clips
clips = list(Path("data/mock_singing/test").glob("*.wav"))[:3]
print(f"[*] Discovered {len(clips)} singing clips for testing.")

def transcribe(model, audio_path):
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1: audio = audio.mean(1)
    if sr != SAMPLE_RATE:
        import torchaudio
        audio = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(
            torch.from_numpy(audio).unsqueeze(0)).squeeze(0).numpy()
            
    inputs = proc.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    
    t0 = time.perf_counter()
    with torch.no_grad():
        predicted_ids = model.generate(input_features=input_features, max_new_tokens=225)
    lat = time.perf_counter() - t0
    text = proc.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    return text, lat

print("\n" + "-" * 70)
print("  PHASE 1: Zero-Shot Baseline Inference")
print("-" * 70)
baseline_results = []
for clip in clips:
    text, lat = transcribe(base_model, clip)
    print(f"  Clip: {clip.name} | Latency: {lat:.2f}s")
    print(f"  Transcription: \"{text}\"\n")
    baseline_results.append(text)

print("-" * 70)
print("  PHASE 2: Injecting LoRA Weights...")
print("-" * 70)
lora_model = PeftModel.from_pretrained(base_model, LORA_PATH)
lora_model.eval()
print("[*] LoRA adapters successfully merged.")

print("\n" + "-" * 70)
print("  PHASE 3: LoRA Fine-Tuned Inference")
print("-" * 70)
lora_results = []
for clip in clips:
    # Use base_model.model to bypass PEFT wrapper issues
    text, lat = transcribe(lora_model.base_model.model, clip)
    print(f"  Clip: {clip.name} | Latency: {lat:.2f}s")
    print(f"  Transcription: \"{text}\"\n")
    lora_results.append(text)

print("=" * 70)
print("  DEMONSTRATION COMPLETE")
print("=" * 70)

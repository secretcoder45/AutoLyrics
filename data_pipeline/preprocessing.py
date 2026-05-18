"""AutoLyrics — Preprocessing pipeline: Whisper feature extraction and caching."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

from core.audio import load_audio, load_audio_segment, normalize_audio, trim_silence
from core.logging import get_logger

logger = get_logger(__name__)


class PreprocessingPipeline:
    """End-to-end audio preprocessing: load → resample → normalize → trim → cache."""

    def __init__(
        self,
        target_sr: int = 16_000,
        normalize_method: str = "peak",
        do_trim_silence: bool = False,
        trim_top_db: float = 25.0,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.target_sr = target_sr
        self.normalize_method = normalize_method
        self.do_trim_silence = do_trim_silence
        self.trim_top_db = trim_top_db
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def process_file(self, audio_path: str, start: float = 0.0, end: float = 0.0) -> Any:
        """Process a single audio file or segment, using cache when available."""
        cache_key = self._cache_key(audio_path, start, end)
        if self.cache_dir:
            cached = self._load_cache(cache_key)
            if cached is not None:
                return cached

        if start > 0 or end > 0:
            waveform = load_audio_segment(audio_path, start, end, self.target_sr)
        else:
            waveform, _ = load_audio(audio_path, self.target_sr)

        waveform = normalize_audio(waveform, method=self.normalize_method)

        if self.do_trim_silence:
            waveform = trim_silence(waveform, top_db=self.trim_top_db)

        if self.cache_dir:
            self._save_cache(cache_key, waveform)

        return waveform

    def _cache_key(self, path: str, start: float, end: float) -> str:
        raw = f"{path}:{start}:{end}:{self.target_sr}:{self.normalize_method}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _load_cache(self, key: str) -> Any:
        if self.cache_dir is None:
            return None
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        return None

    def _save_cache(self, key: str, data: Any) -> None:
        if self.cache_dir is None:
            return
        cache_file = self.cache_dir / f"{key}.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)


def build_preprocessing_pipeline(cfg: dict[str, Any]) -> PreprocessingPipeline:
    """Build a pipeline from a data config's ``preprocessing`` section."""
    prep = cfg.get("preprocessing", {})
    return PreprocessingPipeline(
        target_sr=prep.get("target_sr", 16_000),
        normalize_method=prep.get("normalize", "peak"),
        do_trim_silence=prep.get("trim_silence", False),
        trim_top_db=prep.get("trim_top_db", 25.0),
        cache_dir=cfg.get("cache_dir"),
    )

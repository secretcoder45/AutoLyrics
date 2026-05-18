"""AutoLyrics — Text post-processing for transcription output."""

from __future__ import annotations

import re
from typing import Any

from core.constants import DEFAULT_FILLER_TOKENS
from core.logging import get_logger

logger = get_logger(__name__)


class PostProcessor:
    """Clean up raw Whisper transcription output for readability."""

    def __init__(
        self,
        collapse_repetitions: bool = True,
        strip_filler_tokens: bool = True,
        filler_tokens: list[str] | None = None,
        remove_punctuation: bool = False,
        lowercase: bool = False,
    ) -> None:
        self.collapse_repetitions = collapse_repetitions
        self.strip_filler_tokens = strip_filler_tokens
        self.filler_tokens = set(filler_tokens or DEFAULT_FILLER_TOKENS)
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase

    def process(self, text: str) -> str:
        """Apply all post-processing steps to a transcription string."""
        if not text:
            return text

        if self.lowercase:
            text = text.lower()

        if self.strip_filler_tokens:
            text = self._remove_fillers(text)

        if self.collapse_repetitions:
            text = self._collapse_reps(text)

        if self.remove_punctuation:
            text = re.sub(r"[^\w\s]", "", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _remove_fillers(self, text: str) -> str:
        """Remove filler words (uh, um, ah, etc.)."""
        words = text.split()
        filtered = [w for w in words if w.lower().strip(".,!?") not in self.filler_tokens]
        return " ".join(filtered)

    def _collapse_reps(self, text: str) -> str:
        """Collapse repeated words/phrases caused by decoding artifacts."""
        # Collapse immediate word repetitions: "the the the" → "the"
        words = text.split()
        if not words:
            return text
        result = [words[0]]
        repeat_count = 0
        for i in range(1, len(words)):
            if words[i].lower() == words[i - 1].lower():
                repeat_count += 1
                if repeat_count >= 2:
                    continue
            else:
                repeat_count = 0
            result.append(words[i])
        return " ".join(result)


def build_postprocessor(cfg: dict[str, Any]) -> PostProcessor:
    """Build a PostProcessor from a config dict."""
    pp = cfg.get("postprocessing", {})
    return PostProcessor(
        collapse_repetitions=pp.get("collapse_repetitions", True),
        strip_filler_tokens=pp.get("strip_filler_tokens", True),
        filler_tokens=pp.get("filler_tokens"),
        remove_punctuation=pp.get("remove_punctuation", False),
        lowercase=pp.get("lowercase", False),
    )

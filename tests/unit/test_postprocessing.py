"""Unit tests for interference.postprocessing module."""

from __future__ import annotations

from interference.postprocessing import PostProcessor


class TestPostProcessor:
    def setup_method(self):
        self.pp = PostProcessor()

    def test_empty_string(self):
        assert self.pp.process("") == ""

    def test_filler_removal(self):
        result = self.pp.process("hello uh world um yeah")
        assert "uh" not in result
        assert "um" not in result
        assert "hello" in result

    def test_repetition_collapse(self):
        result = self.pp.process("the the the cat")
        assert result.count("the") <= 2

    def test_whitespace_normalization(self):
        result = self.pp.process("  hello   world  ")
        assert result == "hello world"

    def test_lowercase(self):
        pp = PostProcessor(lowercase=True)
        result = pp.process("Hello WORLD")
        assert result == "hello world"

    def test_punctuation_removal(self):
        pp = PostProcessor(remove_punctuation=True)
        result = pp.process("hello, world!")
        assert "," not in result
        assert "!" not in result

    def test_combined(self):
        pp = PostProcessor(
            collapse_repetitions=True,
            strip_filler_tokens=True,
            lowercase=True,
        )
        result = pp.process("Uh Hello Hello hello World Um")
        assert "uh" not in result.lower()
        assert "um" not in result.lower()

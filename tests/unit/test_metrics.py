"""Unit tests for evaluation.metrics module."""

from __future__ import annotations

from evaluation.metrics import compute_per_sample_metrics, compute_wer_cer, normalize_text


class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("Hello World") == "hello world"

    def test_punctuation_removal(self):
        assert normalize_text("hello, world!") == "hello world"

    def test_whitespace_collapse(self):
        assert normalize_text("hello   world") == "hello world"

    def test_combined(self):
        assert normalize_text("  Hello,  World!  ") == "hello world"


class TestComputeWerCer:
    def test_perfect_match(self):
        preds = ["hello world"]
        refs = ["hello world"]
        result = compute_wer_cer(preds, refs)
        assert result["wer"] == 0.0
        assert result["cer"] == 0.0

    def test_total_mismatch(self):
        preds = ["foo bar"]
        refs = ["hello world"]
        result = compute_wer_cer(preds, refs)
        assert result["wer"] == 100.0

    def test_partial_match(self):
        preds = ["hello earth"]
        refs = ["hello world"]
        result = compute_wer_cer(preds, refs)
        assert 0 < result["wer"] < 100

    def test_empty_reference_filtered(self):
        preds = ["hello", "world"]
        refs = ["hello", ""]
        result = compute_wer_cer(preds, refs)
        assert result["wer"] == 0.0

    def test_multiple_samples(self):
        preds = ["the cat sat", "on the mat"]
        refs = ["the cat sat", "on a mat"]
        result = compute_wer_cer(preds, refs)
        assert result["wer"] > 0


class TestPerSampleMetrics:
    def test_returns_list(self):
        preds = ["hello world", "foo bar"]
        refs = ["hello world", "foo baz"]
        results = compute_per_sample_metrics(preds, refs)
        assert len(results) == 2
        assert results[0]["wer"] == 0.0
        assert results[1]["wer"] > 0

import pytest

from transcript_task.eval.results import BenchmarkResult, ClipResult


def _clip(**overrides) -> ClipResult:
    defaults = dict(clip_id="clip1", duration=5.0, wer_raw=0.2, wer_refined=0.1)
    defaults.update(overrides)
    return ClipResult(**defaults)


def _result(clips=None, **overrides) -> BenchmarkResult:
    defaults = dict(
        tier="quick", seed=42, tag="test-run", asr_model="whisper-x",
        llm_model="qwen-x", refine_prompt_id="refine-v1",
        summarize_prompt_id="summarize-v1", summary_language="pt",
    )
    defaults.update(overrides)
    return BenchmarkResult(clips=clips or [], **defaults)


class TestClipResultRoundtrip:
    def test_to_dict_from_dict_roundtrip(self):
        clip = _clip(error="asr failed")
        restored = ClipResult.from_dict(clip.to_dict())
        assert restored == clip

    def test_from_dict_ignores_unknown_keys(self):
        data = _clip().to_dict()
        data["some_future_field"] = "x"
        ClipResult.from_dict(data)  # must not raise


class TestBenchmarkResultAggregates:
    def test_aggregate_mean_and_median(self):
        result = _result([_clip(wer_refined=0.1), _clip(wer_refined=0.3)])
        agg = result.aggregate("wer_refined")
        assert agg["mean"] == pytest.approx(0.2)
        assert agg["median"] == pytest.approx(0.2)

    def test_aggregate_with_no_values_is_all_none(self):
        result = _result([_clip(wer_refined=None)])
        agg = result.aggregate("wer_refined")
        assert agg == {"mean": None, "median": None, "p50": None, "p95": None}

    def test_aggregate_skips_none_but_uses_present_values(self):
        result = _result([_clip(wer_refined=0.2), _clip(wer_refined=None)])
        agg = result.aggregate("wer_refined")
        assert agg["mean"] == pytest.approx(0.2)

    def test_n_clips_and_n_errors(self):
        result = _result([_clip(), _clip(error="boom"), _clip(error="boom2")])
        assert result.n_clips == 3
        assert result.n_errors == 2

    def test_schema_valid_rate(self):
        result = _result([
            _clip(schema_valid=True), _clip(schema_valid=True), _clip(schema_valid=False)
        ])
        assert result.schema_valid_rate() == pytest.approx(2 / 3)

    def test_schema_valid_rate_none_when_unmeasured(self):
        result = _result([_clip(schema_valid=None)])
        assert result.schema_valid_rate() is None

    def test_slug_uniqueness(self):
        result = _result([
            _clip(docx_filename="a.docx"), _clip(docx_filename="a.docx"), _clip(docx_filename="b.docx"),
        ])
        assert result.slug_uniqueness() == pytest.approx(2 / 3)


class TestBenchmarkResultRoundtrip:
    def test_to_dict_from_dict_roundtrip(self):
        result = _result([_clip(), _clip(wer_refined=0.5, error="x")])
        restored = BenchmarkResult.from_dict(result.to_dict())
        assert restored.tier == result.tier
        assert restored.tag == result.tag
        assert len(restored.clips) == 2
        assert restored.clips[1].error == "x"

    def test_flat_metrics_has_no_none_values(self):
        result = _result([_clip(), _clip(wer_refined=None, schema_valid=None)])
        flat = result.flat_metrics()
        assert all(v is not None for v in flat.values())
        assert "wer_refined_mean" in flat


class TestMarkdownTable:
    def test_renders_without_raising_with_mixed_none_values(self):
        result = _result([
            _clip(),
            _clip(clip_id="clip2", wer_raw=None, wer_refined=0.9, error="asr failed"),
        ])
        markdown = result.markdown_table()
        assert "clip1" in markdown
        assert "clip2" in markdown
        assert "asr failed" in markdown

    def test_renders_with_zero_clips(self):
        result = _result([])
        markdown = result.markdown_table()
        assert "n=0" in markdown

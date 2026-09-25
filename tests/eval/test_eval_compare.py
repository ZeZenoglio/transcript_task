from transcript_task.eval.compare import compare_runs
from transcript_task.eval.results import BenchmarkResult, ClipResult


def _clip(wer_refined: float | None) -> ClipResult:
    return ClipResult(clip_id="c", duration=5.0, wer_refined=wer_refined)


def _run(tag: str, wer_values: list[float]) -> BenchmarkResult:
    return BenchmarkResult(
        tier="quick",
        seed=1,
        tag=tag,
        asr_model="m",
        llm_model="l",
        refine_prompt_id="r",
        summarize_prompt_id="s",
        summary_language="pt",
        clips=[_clip(w) for w in wer_values],
    )


class TestCompareRuns:
    def test_identical_runs_do_not_regress(self):
        baseline = _run("baseline", [0.1, 0.2])
        candidate = _run("candidate", [0.1, 0.2])
        result = compare_runs(baseline, candidate)
        assert result.regressed is False

    def test_improvement_does_not_regress(self):
        baseline = _run("baseline", [0.3, 0.3])
        candidate = _run("candidate", [0.1, 0.1])
        result = compare_runs(baseline, candidate)
        assert result.regressed is False

    def test_worsening_past_threshold_regresses(self):
        baseline = _run("baseline", [0.1, 0.1])
        candidate = _run("candidate", [0.2, 0.2])
        result = compare_runs(baseline, candidate, threshold=0.02)
        assert result.regressed is True

    def test_small_worsening_under_threshold_does_not_regress(self):
        baseline = _run("baseline", [0.10, 0.10])
        candidate = _run("candidate", [0.11, 0.11])
        result = compare_runs(baseline, candidate, threshold=0.02)
        assert result.regressed is False

    def test_missing_metric_in_either_run_is_skipped_not_flagged(self):
        baseline = _run("baseline", [])
        candidate = _run("candidate", [0.5, 0.5])
        result = compare_runs(baseline, candidate)
        assert result.regressed is False
        assert result.diffs == []

    def test_markdown_table_renders(self):
        baseline = _run("baseline", [0.1])
        candidate = _run("candidate", [0.3])
        result = compare_runs(baseline, candidate, threshold=0.02)
        markdown = result.markdown_table()
        assert "wer_refined" in markdown
        assert "YES" in markdown

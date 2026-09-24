import json

import mlflow
import pytest

from fakes import FakeChatModel
from transcript_task.eval.compare import compare_runs
from transcript_task.eval.mlflow_sink import (
    load_local_result,
    log_comparison,
    log_run,
    write_local_artifacts,
)
from transcript_task.eval.results import BenchmarkResult, ClipResult


def _result(tag: str = "test-run") -> BenchmarkResult:
    return BenchmarkResult(
        tier="smoke", seed=None, tag=tag, asr_model="whisper-x", llm_model="qwen-x",
        refine_prompt_id="refine-v1", summarize_prompt_id="summarize-v1", summary_language="pt",
        settings_snapshot={"llm_temperature": 0.2},
        clips=[
            ClipResult(clip_id="c1", duration=4.0, wer_raw=0.3, wer_refined=0.1, schema_valid=True),
            ClipResult(clip_id="c2", duration=6.0, wer_raw=0.2, wer_refined=0.15, error="asr failed"),
        ],
    )


class TestWriteLocalArtifacts:
    def test_writes_results_json_and_table(self, tmp_path):
        result = _result()
        run_dir = write_local_artifacts(result, tmp_path)
        assert (run_dir / "results.json").exists()
        assert (run_dir / "table.md").exists()
        assert not (run_dir / "interpretation.md").exists()

    def test_writes_interpretation_when_given(self, tmp_path):
        run_dir = write_local_artifacts(_result(), tmp_path, interpretation="texto")
        assert (run_dir / "interpretation.md").read_text() == "texto"

    def test_results_json_roundtrips_via_load_local_result(self, tmp_path):
        result = _result()
        write_local_artifacts(result, tmp_path)
        restored = load_local_result(tmp_path, result.tag)
        assert restored.tag == result.tag
        assert len(restored.clips) == 2

    def test_load_missing_tag_raises_with_a_helpful_message(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="nonexistent"):
            load_local_result(tmp_path, "nonexistent")


class TestLogRun:
    def test_logs_a_run_and_returns_a_run_id(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        tracking_dir = tmp_path / "mlruns"
        artifacts_dir = tmp_path / "benchmarks"
        monkeypatch.setattr(
            "transcript_task.eval.mlflow_sink.DEFAULT_ARTIFACTS_DIR", str(artifacts_dir)
        )
        run_id = log_run(_result(), tracking_dir=str(tracking_dir))
        assert run_id

        client = mlflow.tracking.MlflowClient(tracking_uri=f"file:{tracking_dir}")
        run = client.get_run(run_id)
        assert run.data.params["tier"] == "smoke"
        assert run.data.params["asr_model"] == "whisper-x"
        assert "wer_refined_mean" in run.data.metrics

    def test_logs_interpretation_artifact_when_model_given(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        tracking_dir = tmp_path / "mlruns"
        artifacts_dir = tmp_path / "benchmarks"
        monkeypatch.setattr(
            "transcript_task.eval.mlflow_sink.DEFAULT_ARTIFACTS_DIR", str(artifacts_dir)
        )
        model = FakeChatModel(["Interpretação de teste."])
        run_id = log_run(_result("with-interp"), tracking_dir=str(tracking_dir), interpretation_model=model)

        client = mlflow.tracking.MlflowClient(tracking_uri=f"file:{tracking_dir}")
        artifacts = {a.path for a in client.list_artifacts(run_id)}
        assert "interpretation.md" in artifacts


class TestLogComparison:
    def test_logs_comparison_metrics(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        tracking_dir = tmp_path / "mlruns"
        baseline = _result("baseline")
        candidate = _result("candidate")
        comparison = compare_runs(baseline, candidate, threshold=0.02)

        run_id = log_comparison(comparison, tracking_dir=str(tracking_dir))
        client = mlflow.tracking.MlflowClient(tracking_uri=f"file:{tracking_dir}")
        run = client.get_run(run_id)
        assert run.data.params["baseline_tag"] == "baseline"
        assert "regressed" in run.data.metrics

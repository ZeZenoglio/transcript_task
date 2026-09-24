"""MLflow logging -- a sink, not a dependency of metric computation.

Every function here takes an already-computed `BenchmarkResult` and either
writes it to local artifact files or logs it to MLflow. `compute_benchmark`
(in runner.py) never imports this module; `scripts/benchmark.py` is the only
caller, so ripping MLflow out entirely would touch one file.

Tracking is local file-backed (`mlruns/` in the project root, per PLAN.md
decision #5) -- no server, `mlflow ui` reads the same directory.

MLflow 3.x put the plain filesystem backend into maintenance mode and now
raises on `file:./mlruns` unless `MLFLOW_ALLOW_FILE_STORE` is set, nudging
new projects toward a SQLite/DB-backed store instead. Decision #5 predates
that, and a single-user local tool has no need for the DB backend's extra
moving part, so this sets the opt-out flag rather than revisiting the
decision -- `mlflow ui` still reads `mlruns/` exactly as decided.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

from .compare import ComparisonResult
from .interpretation import write_interpretation
from .results import BenchmarkResult

DEFAULT_TRACKING_DIR = "mlruns"
DEFAULT_ARTIFACTS_DIR = "benchmarks"


def write_local_artifacts(result: BenchmarkResult, out_dir: Path, interpretation: str | None = None) -> Path:
    """Write the fallback/companion local artifacts: results.json (the full
    round-trippable object, used by `compare`), table.md (the same thing a
    human or MLflow would see), and interpretation.md if provided.

    This exists independently of whether MLflow logging succeeds -- per
    PLAN.md's explicit fallback design, a benchmark run's results are never
    *only* inside mlruns/.
    """
    run_dir = out_dir / result.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "table.md").write_text(result.markdown_table(), encoding="utf-8")
    if interpretation is not None:
        (run_dir / "interpretation.md").write_text(interpretation, encoding="utf-8")
    return run_dir


def load_local_result(out_dir: Path, tag: str) -> BenchmarkResult:
    path = Path(out_dir) / tag / "results.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No local benchmark result for tag {tag!r} at {path}. "
            f"Run `scripts/benchmark.py run --tag {tag}` first."
        )
    return BenchmarkResult.from_dict(json.loads(path.read_text(encoding="utf-8")))


def log_run(
    result: BenchmarkResult,
    *,
    tracking_dir: str = DEFAULT_TRACKING_DIR,
    experiment_name: str = "transcript_task",
    interpretation_model=None,
) -> str:
    """Log one benchmark run to local MLflow tracking. Returns the run id.

    Params logged: model ids, prompt ids, tier, seed, and a settings
    snapshot. Metrics: everything from `BenchmarkResult.flat_metrics()`.
    Artifacts: the per-clip results table and, if an interpretation model is
    given, the LLM-written interpretation (see interpretation.py).
    """
    import mlflow

    mlflow.set_tracking_uri(f"file:{tracking_dir}")
    mlflow.set_experiment(experiment_name)

    interpretation_text = (
        write_interpretation(result, interpretation_model)
        if interpretation_model is not None else None
    )
    local_dir = write_local_artifacts(
        result, Path(DEFAULT_ARTIFACTS_DIR), interpretation=interpretation_text
    )

    with mlflow.start_run(run_name=result.tag) as run:
        mlflow.log_params({
            "dataset": result.dataset,
            "tier": result.tier,
            "seed": result.seed,
            "asr_model": result.asr_model,
            "llm_model": result.llm_model,
            "refine_prompt_id": result.refine_prompt_id,
            "summarize_prompt_id": result.summarize_prompt_id,
            "summary_language": result.summary_language,
            "n_clips": result.n_clips,
            **{f"settings.{k}": v for k, v in result.settings_snapshot.items()},
        })
        mlflow.log_metrics(result.flat_metrics())
        mlflow.log_artifact(str(local_dir / "results.json"))
        mlflow.log_artifact(str(local_dir / "table.md"))
        if interpretation_text is not None:
            mlflow.log_artifact(str(local_dir / "interpretation.md"))
        return run.info.run_id


def log_comparison(
    comparison: ComparisonResult,
    *,
    tracking_dir: str = DEFAULT_TRACKING_DIR,
    experiment_name: str = "transcript_task",
) -> str:
    import mlflow

    mlflow.set_tracking_uri(f"file:{tracking_dir}")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"compare-{comparison.baseline_tag}-{comparison.candidate_tag}") as run:
        mlflow.log_param("baseline_tag", comparison.baseline_tag)
        mlflow.log_param("candidate_tag", comparison.candidate_tag)
        mlflow.log_param("threshold", comparison.threshold)
        mlflow.log_metric("regressed", float(comparison.regressed))
        for d in comparison.diffs:
            mlflow.log_metric(f"delta_{d.metric}", d.delta)
        return run.info.run_id

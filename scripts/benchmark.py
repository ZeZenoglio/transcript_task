"""Evaluation harness CLI (Phase 6): score the pipeline against FLEURS
ground truth, log the run, and gate CI on regressions.

Usage:
    # smoke tier: the 4 clips already committed under tests/fixtures/,
    # no network, no full FLEURS download needed.
    uv run python scripts/benchmark.py run --tier smoke --tag smoke-baseline

    # quick tier: stratified n=30 sample of the full FLEURS split.
    # Run scripts/fetch_dataset.py first if data/fleurs_pt/ doesn't exist.
    uv run python scripts/benchmark.py run --tier quick --tag qwen9b-baseline

    # full tier: the whole 919-clip split. Slow; for release checks.
    uv run python scripts/benchmark.py run --tier full --tag release-1.0

    # Swap models via the same env vars Settings always honors:
    TRANSCRIPT_LLM_MODEL=qwen3.5:4b uv run python scripts/benchmark.py run \\
        --tier quick --tag qwen4b-candidate

    # Compare two tagged runs; exits non-zero if a metric regressed past
    # --threshold. This exit code is what CI consumes.
    uv run python scripts/benchmark.py compare qwen9b-baseline qwen4b-candidate

Every run's results are written to benchmarks/<tag>/ (results.json, table.md,
interpretation.md) independent of whether MLflow logging succeeds -- MLflow
is a sink for the same object, not the source of truth. `compare` reads from
benchmarks/, not from MLflow, so CI never needs a tracking server reachable.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from transcript_task.asr import MlxWhisperTranscriber
from transcript_task.eval.compare import compare_runs
from transcript_task.eval.embeddings import SentenceTransformerEmbedder
from transcript_task.eval.mlflow_sink import (
    DEFAULT_ARTIFACTS_DIR,
    load_local_result,
    log_comparison,
    log_run,
    write_local_artifacts,
)
from transcript_task.eval.results import BenchmarkResult
from transcript_task.eval.runner import evaluate_clip, peak_rss_mb
from transcript_task.eval.tiers import DEFAULT_QUICK_N, DEFAULT_SEED, select_tier
from transcript_task.prompts import REFINE_PROMPT_TEMPLATE, get_summarize_template
from transcript_task.refine import OllamaChatModel
from transcript_task.settings import PROJECT_ROOT, Settings

FULL_MANIFEST = PROJECT_ROOT / "data" / "fleurs_pt" / "manifest.jsonl"
FULL_AUDIO_ROOT = PROJECT_ROOT / "data" / "fleurs_pt"
SMOKE_MANIFEST = PROJECT_ROOT / "tests" / "fixtures" / "manifest.jsonl"
SMOKE_AUDIO_ROOT = PROJECT_ROOT / "tests" / "fixtures"


def _load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _log(msg: str) -> None:
    print(f"[benchmark] {msg}", flush=True)


def run_benchmark(args: argparse.Namespace) -> BenchmarkResult:
    settings = Settings()

    full_manifest = _load_manifest(FULL_MANIFEST)
    smoke_manifest = _load_manifest(SMOKE_MANIFEST)
    if args.tier != "smoke" and not full_manifest:
        sys.exit(
            f"{FULL_MANIFEST} not found. Run scripts/fetch_dataset.py first, "
            f"or use --tier smoke to run against the committed fixtures."
        )

    clips = select_tier(full_manifest, smoke_manifest, args.tier, n=args.n, seed=args.seed)
    audio_root = SMOKE_AUDIO_ROOT if args.tier == "smoke" else FULL_AUDIO_ROOT
    _log(f"tier={args.tier} n={len(clips)} asr_model={settings.asr_model} llm_model={settings.llm_model}")

    transcriber = MlxWhisperTranscriber(settings.asr_model)
    chat_model = OllamaChatModel(settings.llm_model)
    embedder = None if args.no_semdist else SentenceTransformerEmbedder()

    clip_results = []
    with tempfile.TemporaryDirectory(prefix="benchmark_normalized_") as tmp:
        tmp_dir = Path(tmp)
        for i, clip in enumerate(clips, start=1):
            started = time.time()
            result = evaluate_clip(
                clip, audio_root, settings, transcriber, chat_model, embedder, tmp_dir,
                skip_refine=args.skip_refine,
            )
            clip_results.append(result)
            status = result.error or (
                f"wer_raw={result.wer_raw:.3f} wer_refined={result.wer_refined:.3f}"
                if result.wer_refined is not None else f"wer_raw={result.wer_raw:.3f}"
            )
            _log(f"[{i}/{len(clips)}] {result.clip_id} ({time.time() - started:.1f}s): {status}")

    model_load_seconds = {}
    if clip_results:
        if clip_results[0].asr_seconds is not None:
            model_load_seconds["asr_first_call"] = clip_results[0].asr_seconds
        if clip_results[0].refine_seconds is not None:
            model_load_seconds["llm_first_call"] = clip_results[0].refine_seconds

    benchmark_result = BenchmarkResult(
        tier=args.tier,
        seed=args.seed if args.tier == "quick" else None,
        tag=args.tag,
        asr_model=settings.asr_model,
        llm_model=settings.llm_model,
        refine_prompt_id=REFINE_PROMPT_TEMPLATE.id,
        summarize_prompt_id=get_summarize_template(settings.summary_language).id,
        summary_language=settings.summary_language,
        clips=clip_results,
        settings_snapshot={
            "llm_temperature": settings.llm_temperature,
            "llm_num_ctx": settings.llm_num_ctx,
            "anonymize_metadata": settings.anonymize_metadata,
            "target_codec": settings.target_codec,
            "skip_refine": args.skip_refine,
        },
        peak_rss_mb=peak_rss_mb(),
        model_load_seconds=model_load_seconds,
    )

    interpretation_model = None if args.no_interpretation else chat_model
    if args.no_mlflow:
        from transcript_task.eval.interpretation import write_interpretation

        interpretation_text = (
            write_interpretation(benchmark_result, interpretation_model)
            if interpretation_model is not None else None
        )
        write_local_artifacts(
            benchmark_result, Path(DEFAULT_ARTIFACTS_DIR), interpretation=interpretation_text
        )
    else:
        run_id = log_run(benchmark_result, interpretation_model=interpretation_model)
        _log(f"logged to mlflow: run_id={run_id} (mlflow ui --backend-store-uri file:mlruns)")

    _log(f"done: {benchmark_result.n_clips} clips, {benchmark_result.n_errors} errors, "
         f"results in {DEFAULT_ARTIFACTS_DIR}/{args.tag}/")
    return benchmark_result


def run_compare(args: argparse.Namespace) -> int:
    baseline = load_local_result(Path(DEFAULT_ARTIFACTS_DIR), args.baseline_tag)
    candidate = load_local_result(Path(DEFAULT_ARTIFACTS_DIR), args.candidate_tag)
    comparison = compare_runs(baseline, candidate, threshold=args.threshold)

    print(comparison.markdown_table())
    if not args.no_mlflow:
        run_id = log_comparison(comparison)
        _log(f"logged comparison to mlflow: run_id={run_id}")

    if comparison.regressed:
        _log("REGRESSION detected past threshold.")
        return 1
    _log("no regression past threshold.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    run_ap = sub.add_parser("run", help="run a benchmark and log the results")
    run_ap.add_argument("--tier", choices=["smoke", "quick", "full"], default="quick")
    run_ap.add_argument("--tag", required=True, help="a name for this run, e.g. qwen9b-baseline")
    run_ap.add_argument("--n", type=int, default=DEFAULT_QUICK_N, help="quick tier sample size")
    run_ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="quick tier sampling seed")
    run_ap.add_argument("--skip-refine", action="store_true", help="score ASR only, no LLM cleanup")
    run_ap.add_argument("--no-semdist", action="store_true", help="skip SemDist (no embedding model load)")
    run_ap.add_argument("--no-interpretation", action="store_true", help="skip the LLM interpretation report")
    run_ap.add_argument("--no-mlflow", action="store_true", help="write local artifacts only, skip MLflow")

    compare_ap = sub.add_parser("compare", help="diff two tagged runs; exit 1 on regression")
    compare_ap.add_argument("baseline_tag")
    compare_ap.add_argument("candidate_tag")
    compare_ap.add_argument("--threshold", type=float, default=0.02,
                            help="absolute WER/CER/SemDist increase that counts as a regression")
    compare_ap.add_argument("--no-mlflow", action="store_true")

    args = ap.parse_args()
    if args.command == "run":
        run_benchmark(args)
    elif args.command == "compare":
        sys.exit(run_compare(args))


if __name__ == "__main__":
    main()

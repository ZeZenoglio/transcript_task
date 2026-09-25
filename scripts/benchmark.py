"""Evaluation harness CLI (Phase 6): score the pipeline against ground truth
from FLEURS (clean read speech) or Common Voice (noisier, more varied real
recordings -- see fetch_common_voice.py's docstring for why), log the run,
and gate CI on regressions.

Usage:
    # smoke tier: the 4 clips already committed under tests/fixtures/,
    # no network, no full dataset download needed.
    uv run python scripts/benchmark.py run --tier smoke --tag smoke-baseline

    # quick tier: stratified n=30 sample of the full FLEURS split.
    # Run scripts/fetch_dataset.py first if data/fleurs_pt/ doesn't exist.
    uv run python scripts/benchmark.py run --tier quick --tag qwen9b-baseline

    # the same, but against the noisier Common Voice tier instead --
    # run scripts/fetch_common_voice.py first.
    uv run python scripts/benchmark.py run --dataset common_voice --tier quick --tag qwen9b-noisy

    # full tier: the whole split. Slow; for release checks.
    uv run python scripts/benchmark.py run --tier full --tag release-1.0

    # Swap models via the same env vars Settings always honors:
    TRANSCRIPT_LLM_MODEL=qwen3.5:4b uv run python scripts/benchmark.py run \\
        --tier quick --tag qwen4b-candidate

    # Compare two tagged runs; exits non-zero if a metric regressed past
    # --threshold. This exit code is what CI consumes. Only compare runs
    # from the same dataset -- see the module docstring's compare_runs note.
    uv run python scripts/benchmark.py compare qwen9b-baseline qwen4b-candidate

Every run's results are written to benchmarks/<tag>/ (results.json, table.md,
interpretation.md) independent of whether MLflow logging succeeds -- MLflow
is a sink for the same object, not the source of truth. `compare` reads from
benchmarks/, not from MLflow, so CI never needs a tracking server reachable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transcript_task.eval.compare import compare_runs
from transcript_task.eval.mlflow_sink import DEFAULT_ARTIFACTS_DIR, load_local_result, log_comparison
from transcript_task.eval.orchestrator import DATASETS, DatasetNotFetched, run_benchmark_tier
from transcript_task.eval.results import BenchmarkResult
from transcript_task.eval.tiers import DEFAULT_QUICK_N, DEFAULT_SEED
from transcript_task.settings import Settings


def _log(msg: str) -> None:
    print(f"[benchmark] {msg}", flush=True)


def run_benchmark(args: argparse.Namespace) -> BenchmarkResult:
    settings = Settings()
    _log(f"dataset={args.dataset} tier={args.tier} asr_model={settings.asr_model} "
         f"llm_model={settings.llm_model} refine_prompt_id={settings.refine_prompt_id}")

    def on_clip(i: int, total: int, result) -> None:
        status = result.error or (
            f"wer_raw={result.wer_raw:.3f} wer_refined={result.wer_refined:.3f}"
            if result.wer_refined is not None else f"wer_raw={result.wer_raw:.3f}"
        )
        _log(f"[{i}/{total}] {result.clip_id}: {status}")

    try:
        benchmark_result = run_benchmark_tier(
            settings, dataset=args.dataset, tier=args.tier, tag=args.tag,
            n=args.n, seed=args.seed, skip_refine=args.skip_refine,
            no_semdist=args.no_semdist, no_interpretation=args.no_interpretation,
            use_mlflow=not args.no_mlflow, on_clip=on_clip,
        )
    except DatasetNotFetched as exc:
        sys.exit(str(exc))

    if not args.no_mlflow:
        _log("logged to mlflow (mlflow ui --backend-store-uri file:mlruns)")
    _log(f"done: {benchmark_result.n_clips} clips, {benchmark_result.n_errors} errors, "
         f"results in {DEFAULT_ARTIFACTS_DIR}/{args.tag}/")
    return benchmark_result


def run_compare(args: argparse.Namespace) -> int:
    baseline = load_local_result(Path(DEFAULT_ARTIFACTS_DIR), args.baseline_tag)
    candidate = load_local_result(Path(DEFAULT_ARTIFACTS_DIR), args.candidate_tag)
    if baseline.dataset != candidate.dataset:
        sys.exit(
            f"Refusing to compare runs from different datasets "
            f"({baseline.dataset!r} vs {candidate.dataset!r}) -- their WER/CER/SemDist "
            f"numbers aren't on the same scale (see the noisy-tier numbers in the README)."
        )
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
    run_ap.add_argument("--dataset", choices=list(DATASETS), default="fleurs")
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

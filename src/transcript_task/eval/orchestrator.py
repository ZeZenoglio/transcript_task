"""Shared "run a benchmark tier end to end" orchestration.

One implementation of what `scripts/benchmark.py run` does, used by both
that CLI and the API's `POST /v1/benchmark` (Phase 7) -- extracted here
specifically so the API doesn't duplicate this logic in `api/app.py`.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

from ..asr import MlxWhisperTranscriber
from ..prompts import get_summarize_template
from ..refine import OllamaChatModel
from ..settings import PROJECT_ROOT, Settings
from .embeddings import SentenceTransformerEmbedder
from .mlflow_sink import DEFAULT_ARTIFACTS_DIR, log_run, write_local_artifacts
from .results import BenchmarkResult, ClipResult
from .runner import evaluate_clip, peak_rss_mb
from .tiers import DEFAULT_QUICK_N, DEFAULT_SEED, select_tier


# Each entry is self-contained: full split (for quick/full tiers, fetched
# separately) and the small committed smoke set (for CI/no-download runs).
# expected_language_variant feeds evaluate_clip's summary-conformance check
# (see summarize.TranscriptSummary.language_variant) -- None means "don't
# assert," for a dataset whose clips don't share one known variant.
class DatasetConfig(TypedDict):
    full_manifest: Path
    full_audio_root: Path
    smoke_manifest: Path
    smoke_audio_root: Path
    expected_language_variant: str | None
    fetch_hint: str


DATASETS: dict[str, DatasetConfig] = {
    "fleurs": {
        "full_manifest": PROJECT_ROOT / "data" / "fleurs_pt" / "manifest.jsonl",
        "full_audio_root": PROJECT_ROOT / "data" / "fleurs_pt",
        "smoke_manifest": PROJECT_ROOT / "tests" / "fixtures" / "manifest.jsonl",
        "smoke_audio_root": PROJECT_ROOT / "tests" / "fixtures",
        "expected_language_variant": "pt-BR",
        "fetch_hint": "scripts/fetch_dataset.py",
    },
    "common_voice": {
        "full_manifest": PROJECT_ROOT / "data" / "common_voice_pt" / "manifest.jsonl",
        "full_audio_root": PROJECT_ROOT / "data" / "common_voice_pt",
        "smoke_manifest": PROJECT_ROOT / "tests" / "fixtures_noisy" / "manifest.jsonl",
        "smoke_audio_root": PROJECT_ROOT / "tests" / "fixtures_noisy",
        # Common Voice pt mixes pt-BR/pt-PT/unlabeled contributors -- no
        # single expected variant to assert per clip.
        "expected_language_variant": None,
        "fetch_hint": "scripts/fetch_common_voice.py",
    },
}


class DatasetNotFetched(Exception):
    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        cfg = DATASETS[dataset]
        super().__init__(
            f"{cfg['full_manifest']} not found. Run {cfg['fetch_hint']} first, "
            f"or use tier='smoke' to run against the committed fixtures."
        )


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def run_benchmark_tier(
    settings: Settings,
    *,
    dataset: str,
    tier: str,
    tag: str,
    n: int = DEFAULT_QUICK_N,
    seed: int = DEFAULT_SEED,
    skip_refine: bool = False,
    no_semdist: bool = False,
    no_interpretation: bool = False,
    use_mlflow: bool = True,
    on_clip: Callable[[int, int, ClipResult], None] | None = None,
    transcriber=None,
    chat_model=None,
) -> BenchmarkResult:
    """Run one benchmark tier end to end and return the (already-persisted,
    per PLAN.md's fallback design) `BenchmarkResult`. Raises
    `DatasetNotFetched` if a non-smoke tier is requested but the dataset was
    never downloaded -- callers decide how to surface that (CLI: exit with
    a message; API: turn it into an HTTP error).

    `transcriber`/`chat_model` are optional, purely for tests -- same
    Protocol-injection pattern `evaluate_clip` itself already uses; real
    callers leave them None and get the real `MlxWhisperTranscriber`/
    `OllamaChatModel`. Needed so the API's `/v1/benchmark` endpoint can be
    tested without triggering a real model run from a background thread.
    """
    dataset_cfg = DATASETS[dataset]

    full_manifest = load_manifest(dataset_cfg["full_manifest"])
    smoke_manifest = load_manifest(dataset_cfg["smoke_manifest"])
    if tier != "smoke" and not full_manifest:
        raise DatasetNotFetched(dataset)

    clips = select_tier(full_manifest, smoke_manifest, tier, n=n, seed=seed)
    audio_root = (
        dataset_cfg["smoke_audio_root"] if tier == "smoke" else dataset_cfg["full_audio_root"]
    )

    transcriber = transcriber or MlxWhisperTranscriber(settings.asr_model)
    chat_model = chat_model or OllamaChatModel(settings.llm_model)
    embedder = None if no_semdist else SentenceTransformerEmbedder()

    clip_results = []
    with tempfile.TemporaryDirectory(prefix="benchmark_normalized_") as tmp:
        tmp_dir = Path(tmp)
        for i, clip in enumerate(clips, start=1):
            result = evaluate_clip(
                clip,
                audio_root,
                settings,
                transcriber,
                chat_model,
                embedder,
                tmp_dir,
                expected_language_variant=dataset_cfg["expected_language_variant"],
                skip_refine=skip_refine,
            )
            clip_results.append(result)
            if on_clip is not None:
                on_clip(i, len(clips), result)

    model_load_seconds = {}
    if clip_results:
        if clip_results[0].asr_seconds is not None:
            model_load_seconds["asr_first_call"] = clip_results[0].asr_seconds
        if clip_results[0].refine_seconds is not None:
            model_load_seconds["llm_first_call"] = clip_results[0].refine_seconds

    benchmark_result = BenchmarkResult(
        tier=tier,
        seed=seed if tier == "quick" else None,
        tag=tag,
        asr_model=settings.asr_model,
        llm_model=settings.llm_model,
        refine_prompt_id=settings.refine_prompt_id,
        summarize_prompt_id=get_summarize_template(settings.summary_language).id,
        summary_language=settings.summary_language,
        dataset=dataset,
        clips=clip_results,
        settings_snapshot={
            "llm_temperature": settings.llm_temperature,
            "llm_num_ctx": settings.llm_num_ctx,
            "anonymize_metadata": settings.anonymize_metadata,
            "target_codec": settings.target_codec,
            "skip_refine": skip_refine,
        },
        peak_rss_mb=peak_rss_mb(),
        model_load_seconds=model_load_seconds,
    )

    interpretation_model = None if no_interpretation else chat_model
    if not use_mlflow:
        from .interpretation import write_interpretation

        interpretation_text = (
            write_interpretation(benchmark_result, interpretation_model)
            if interpretation_model is not None
            else None
        )
        write_local_artifacts(
            benchmark_result, Path(DEFAULT_ARTIFACTS_DIR), interpretation=interpretation_text
        )
    else:
        log_run(benchmark_result, interpretation_model=interpretation_model)

    return benchmark_result

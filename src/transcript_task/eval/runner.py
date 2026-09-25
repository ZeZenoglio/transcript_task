"""Ties the eval package's pure metric functions to the real pipeline
stages, one clip at a time. This is the only module in the eval package
that imports `transcript_task`'s ASR/LLM/audio code directly.

Deliberately not built on top of `pipeline.py`'s stage_* functions: those
are batch-oriented around `output/transcripts.json` checkpointing and write
real .docx files to disk, neither of which the benchmark wants. This reuses
the same underlying building blocks (`Transcriber`, `ChatModel`,
`convert_to_target`, `refine_transcript`, `summarize_transcript`,
`docx_filename`) directly instead.
"""

from __future__ import annotations

import platform
import resource
import time
from pathlib import Path

from ..audio import AudioError, convert_to_target, is_already_target_format, probe
from ..pipeline import new_transcript_id
from ..prompts import get_refine_template
from ..refine import ChatModel, refine_transcript
from ..settings import Settings
from ..summarize import TranscriptSummary, docx_filename, summarize_transcript
from .embeddings import Embedder
from .metrics import (
    character_error_rate,
    content_recall,
    evaluate_summary,
    length_ratio,
    semantic_distance,
    word_error_rate,
)
from .results import ClipResult


class _CountingChatModel:
    """Wraps a ChatModel to count calls and forward usage metadata, so the
    caller can tell "answered on the first try" from "needed the retry"
    without summarize.py needing to expose that itself. Implements the same
    ChatModel protocol, so it's a transparent substitution."""

    def __init__(self, inner: ChatModel) -> None:
        self._inner = inner
        self.call_count = 0
        self.usages: list[dict | None] = []

    def chat(self, *, system: str, user: str, options: dict, format: dict | None = None) -> str:
        self.call_count += 1
        result = self._inner.chat(system=system, user=user, options=options, format=format)
        self.usages.append(getattr(self._inner, "last_usage", None))
        return result


def _completion_tokens(chat_model: ChatModel) -> int | None:
    """Read the completion-token count off `last_usage`, a side channel
    OllamaChatModel populates after every call (see refine.py) but which
    isn't part of the ChatModel protocol proper -- so this degrades to None
    for any ChatModel implementation that doesn't set it, rather than
    raising."""
    usage = getattr(chat_model, "last_usage", None)
    return usage.get("completion_tokens") if usage else None


def peak_rss_mb() -> float:
    """Peak resident set size of this process so far, in MB.

    `ru_maxrss` is bytes on macOS/BSD and kilobytes on Linux -- one of the
    stdlib's odder platform inconsistencies (see `man getrusage`)."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if platform.system() == "Darwin" else 1024
    return raw / divisor


def evaluate_clip(
    clip: dict,
    audio_root: Path,
    settings: Settings,
    transcriber,
    chat_model: ChatModel,
    embedder: Embedder | None,
    tmp_dir: Path,
    *,
    expected_language_variant: str | None = "pt-BR",
    skip_refine: bool = False,
) -> ClipResult:
    """Run one FLEURS clip through normalize -> transcribe -> refine ->
    summarize, scoring each stage against `clip["ground_truth"]`.

    Never raises: any stage failure is recorded on `ClipResult.error` and
    downstream metrics for that clip are left as None, exactly like
    pipeline.py's own per-file error handling (a bad clip doesn't abort the
    whole benchmark run).
    """
    clip_id = Path(clip["audio_path"]).stem
    ground_truth = clip["ground_truth"]
    result = ClipResult(clip_id=clip_id, duration=clip["duration"])

    src = audio_root / clip["audio_path"]
    tmp_dir.mkdir(parents=True, exist_ok=True)
    wav = tmp_dir / f"{clip_id}.wav"

    try:
        info = probe(src)
        if is_already_target_format(src, info, settings):
            wav.write_bytes(src.read_bytes())
        else:
            convert_to_target(src, wav, settings)
    except AudioError as exc:
        result.error = f"normalize failed: {exc}"
        return result

    try:
        started = time.time()
        transcription = transcriber.transcribe(wav, language=settings.asr_language)
        result.asr_seconds = time.time() - started
        result.realtime_factor = (
            clip["duration"] / result.asr_seconds if result.asr_seconds > 0 else None
        )
    except Exception as exc:  # noqa: BLE001 - one bad clip must not abort the run
        result.error = f"asr failed: {exc}"
        return result

    raw_transcript = transcription.text
    result.wer_raw = word_error_rate(ground_truth, raw_transcript)
    result.cer_raw = character_error_rate(ground_truth, raw_transcript)
    if embedder is not None:
        result.semdist_raw = semantic_distance(ground_truth, raw_transcript, embedder)

    refined_transcript = raw_transcript
    if not skip_refine:
        try:
            started = time.time()
            refined_transcript = refine_transcript(
                raw_transcript,
                chat_model,
                get_refine_template(settings.refine_prompt_id),
                settings.llm_options,
            )
            result.refine_seconds = time.time() - started
            result.refine_completion_tokens = _completion_tokens(chat_model)
            if result.refine_completion_tokens is not None and result.refine_seconds > 0:
                result.refine_tokens_per_second = (
                    result.refine_completion_tokens / result.refine_seconds
                )
        except Exception as exc:  # noqa: BLE001
            result.error = f"refine failed: {exc}"
            refined_transcript = None

    if refined_transcript is not None:
        result.wer_refined = word_error_rate(ground_truth, refined_transcript)
        result.cer_refined = character_error_rate(ground_truth, refined_transcript)
        result.content_recall = content_recall(raw_transcript, refined_transcript)
        result.length_ratio = length_ratio(raw_transcript, refined_transcript)
        if embedder is not None:
            result.semdist_refined = semantic_distance(ground_truth, refined_transcript, embedder)

    counting_model = _CountingChatModel(chat_model)
    summary: TranscriptSummary | None = None
    try:
        started = time.time()
        summary = summarize_transcript(
            raw_transcript,
            refined_transcript or raw_transcript,
            counting_model,
            language=settings.summary_language,
            options=settings.llm_options,
        )
        result.summarize_seconds = time.time() - started
        total_completion = sum(
            u["completion_tokens"]
            for u in counting_model.usages
            if u and u.get("completion_tokens") is not None
        )
        if total_completion:
            result.summarize_completion_tokens = total_completion
            if result.summarize_seconds > 0:
                result.summarize_tokens_per_second = total_completion / result.summarize_seconds
    except Exception as exc:  # noqa: BLE001
        result.error = (result.error + "; " if result.error else "") + f"summarize failed: {exc}"

    used_fallback = summary is not None and summary == TranscriptSummary.fallback(
        settings.summary_language
    )
    used_retry = counting_model.call_count >= 2

    summary_metrics = evaluate_summary(
        summary,
        used_retry=used_retry,
        used_fallback=used_fallback,
        expected_language_variant=expected_language_variant,
        transcript=refined_transcript or raw_transcript,
        embedder=embedder,
    )
    result.schema_valid = summary_metrics.schema_valid
    result.used_retry = summary_metrics.used_retry
    result.used_fallback = summary_metrics.used_fallback
    result.title_len_ok = summary_metrics.title_len_ok
    result.description_len_ok = summary_metrics.description_len_ok
    result.topic_count_ok = summary_metrics.topic_count_ok
    result.description_transcript_semdist = summary_metrics.description_transcript_semdist

    if summary is not None:
        result.docx_filename = docx_filename(
            new_transcript_id(),
            summary,
            language=settings.summary_language,
            anonymize=settings.anonymize_metadata,
        )

    return result

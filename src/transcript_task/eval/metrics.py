"""Pure metric functions -- no MLflow, no pipeline, no I/O.

Kept this way deliberately (decision in PLAN.md Phase 6): MLflow is a sink,
not a dependency of metric computation, so these functions are trivial to
unit-test and would survive MLflow being ripped out entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jiwer

from ..summarize import TranscriptSummary
from .embeddings import Embedder
from .normalizer import normalize_pt


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER after pt-aware normalization. 0.0 for two empty strings (nothing
    to get wrong); 1.0 if the reference is non-empty and the hypothesis is
    empty (every reference word is a deletion)."""
    ref, hyp = normalize_pt(reference), normalize_pt(hypothesis)
    if not ref and not hyp:
        return 0.0
    if not ref:
        # jiwer raises on an empty reference; an empty reference with any
        # hypothesis text is all insertions, which isn't a rate jiwer defines
        # (division by zero reference words) -- treat as "as wrong as
        # possible" rather than propagating a ZeroDivisionError.
        return 1.0 if hyp else 0.0
    return jiwer.wer(ref, hyp)


def character_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize_pt(reference), normalize_pt(hypothesis)
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0 if hyp else 0.0
    return jiwer.cer(ref, hyp)


def semantic_distance(reference: str, hypothesis: str, embedder: Embedder) -> float:
    """1 - cosine similarity between the two texts' embeddings. 0.0 means
    identical meaning as far as the embedding model can tell; embeddings
    are pre-normalized (`Embedder.embed` returns unit vectors), so cosine
    similarity is just the dot product."""
    if not reference and not hypothesis:
        return 0.0
    vectors = embedder.embed([reference, hypothesis])
    similarity = float(vectors[0] @ vectors[1])
    return 1.0 - similarity


@dataclass
class RefineDelta:
    """WER/CER measured against the same ground truth, before and after the
    refine stage -- the trick from PLAN.md that turns refine (which has no
    ground truth of its own) into something measurable. A negative delta
    means refine helped; positive means it hurt (e.g. paraphrasing away from
    what was actually said)."""

    wer_raw: float
    wer_refined: float
    cer_raw: float
    cer_refined: float

    @property
    def wer_delta(self) -> float:
        return self.wer_refined - self.wer_raw

    @property
    def cer_delta(self) -> float:
        return self.cer_refined - self.cer_raw


def refine_delta(ground_truth: str, raw_transcript: str, refined_transcript: str) -> RefineDelta:
    return RefineDelta(
        wer_raw=word_error_rate(ground_truth, raw_transcript),
        wer_refined=word_error_rate(ground_truth, refined_transcript),
        cer_raw=character_error_rate(ground_truth, raw_transcript),
        cer_refined=character_error_rate(ground_truth, refined_transcript),
    )


def content_recall(raw_transcript: str, refined_transcript: str) -> float:
    """Fraction of the raw transcript's normalized word *set* still present
    (as a substring token) in the refined transcript. A cheap guard against
    refine silently dropping content -- not a substitute for refine_delta,
    which is what actually catches paraphrasing against ground truth; this
    catches outright omission even when there's no ground truth at all
    (i.e. in production, not just in the benchmark)."""
    raw_words = set(normalize_pt(raw_transcript).split())
    if not raw_words:
        return 1.0
    refined_words = set(normalize_pt(refined_transcript).split())
    return len(raw_words & refined_words) / len(raw_words)


def length_ratio(raw_transcript: str, refined_transcript: str) -> float:
    """refined/raw character-length ratio. A guard against refine silently
    truncating (ratio << 1) or rambling/hallucinating (ratio >> 1)."""
    raw_len = len(raw_transcript)
    if raw_len == 0:
        return 1.0
    return len(refined_transcript) / raw_len


# ---------------------------------------------------------------------------
# Summary-stage metrics -- deterministic only, per PLAN.md decision #6.
# ---------------------------------------------------------------------------

@dataclass
class SummaryMetrics:
    schema_valid: bool
    used_retry: bool
    used_fallback: bool
    title_len_ok: bool
    description_len_ok: bool
    topic_count_ok: bool
    language_variant_matches: bool | None  # None when the dataset has no known variant
    description_transcript_semdist: float | None  # None when no embedder was given


def evaluate_summary(
    summary: TranscriptSummary | None,
    *,
    used_retry: bool,
    used_fallback: bool,
    expected_language_variant: str | None,
    transcript: str | None = None,
    embedder: Embedder | None = None,
) -> SummaryMetrics:
    """Deterministic conformance checks against `summarize.TranscriptSummary`.

    Takes an already-validated `TranscriptSummary` (or `None` if summarization
    failed outright before pydantic could even see a value): `schema_valid`
    tracks *that* case, not field-level rules, since pydantic already
    enforces field-level constraints (min/max length, enum membership) at
    construction time -- a `TranscriptSummary` that exists at all already
    satisfies them structurally. The remaining checks are the softer,
    non-enforced conformance signals the plan calls out: are fields being
    used sensibly, not just validly.
    """
    if summary is None:
        return SummaryMetrics(
            schema_valid=False,
            used_retry=used_retry,
            used_fallback=used_fallback,
            title_len_ok=False,
            description_len_ok=False,
            topic_count_ok=False,
            language_variant_matches=None,
            description_transcript_semdist=None,
        )

    title_len_ok = 1 <= len(summary.title) <= 80
    description_len_ok = 1 <= len(summary.description) <= 1200
    # The prompt asks for 3-8 topics (see prompts.py); the pydantic model
    # only enforces 1-12 as a non-failing floor/ceiling (see summarize.py's
    # comment on why), so "did it follow the prompt's actual guidance" is a
    # separate, softer signal worth tracking here.
    topic_count_ok = 3 <= len(summary.topics) <= 8

    language_variant_matches = (
        summary.language_variant == expected_language_variant
        if expected_language_variant is not None
        else None
    )

    semdist = None
    if transcript is not None and embedder is not None:
        semdist = semantic_distance(transcript, summary.description, embedder)

    return SummaryMetrics(
        schema_valid=True,
        used_retry=used_retry,
        used_fallback=used_fallback,
        title_len_ok=title_len_ok,
        description_len_ok=description_len_ok,
        topic_count_ok=topic_count_ok,
        language_variant_matches=language_variant_matches,
        description_transcript_semdist=semdist,
    )


def slug_uniqueness(filenames: list[str]) -> float:
    """Fraction of docx filenames that are unique. 1.0 means no collisions
    (transcript_id makes true collisions impossible in production -- see
    docx_filename's docstring -- but a benchmark run may not always include
    the id in what it's checking, so this stays a general-purpose check)."""
    if not filenames:
        return 1.0
    return len(set(filenames)) / len(filenames)


# ---------------------------------------------------------------------------
# Performance metrics -- every stage.
# ---------------------------------------------------------------------------

@dataclass
class StageTiming:
    seconds: float
    audio_seconds: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def realtime_factor(self) -> float | None:
        """audio-seconds / wall-seconds; >1 means faster than realtime.
        Only meaningful for stages measured against an audio duration
        (ASR) -- None for text-only stages."""
        if self.audio_seconds is None or self.seconds <= 0:
            return None
        return self.audio_seconds / self.seconds

    @property
    def tokens_per_second(self) -> float | None:
        if self.completion_tokens is None or self.seconds <= 0:
            return None
        return self.completion_tokens / self.seconds


@dataclass
class PerformanceSummary:
    """Aggregated performance numbers for one benchmark run. Percentiles are
    computed by the caller (results.py) across all clips' StageTimings, kept
    here only as a typed container so scripts/benchmark.py has one place to
    look."""

    latency_p50: dict[str, float] = field(default_factory=dict)
    latency_p95: dict[str, float] = field(default_factory=dict)
    mean_realtime_factor: dict[str, float] = field(default_factory=dict)
    mean_tokens_per_second: dict[str, float] = field(default_factory=dict)
    peak_rss_mb: float | None = None
    model_load_seconds: dict[str, float] = field(default_factory=dict)


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile, no numpy/scipy dependency needed for
    something this small. `pct` in [0, 100]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    frac = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * frac

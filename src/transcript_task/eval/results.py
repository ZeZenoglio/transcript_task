"""Result containers: one `ClipResult` per benchmarked clip, aggregated into
a `BenchmarkResult` for one run. Both are plain dataclasses with `to_dict`/
`from_dict`, so a run can round-trip through JSON without depending on
MLflow (see mlflow_sink.py) or on the pipeline having been re-run.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .metrics import percentile


@dataclass
class ClipResult:
    clip_id: str
    duration: float
    wer_raw: float | None = None
    wer_refined: float | None = None
    cer_raw: float | None = None
    cer_refined: float | None = None
    semdist_raw: float | None = None
    semdist_refined: float | None = None
    content_recall: float | None = None
    length_ratio: float | None = None
    asr_seconds: float | None = None
    refine_seconds: float | None = None
    summarize_seconds: float | None = None
    realtime_factor: float | None = None
    refine_completion_tokens: int | None = None
    summarize_completion_tokens: int | None = None
    refine_tokens_per_second: float | None = None
    summarize_tokens_per_second: float | None = None
    schema_valid: bool | None = None
    used_retry: bool | None = None
    used_fallback: bool | None = None
    title_len_ok: bool | None = None
    description_len_ok: bool | None = None
    topic_count_ok: bool | None = None
    description_transcript_semdist: float | None = None
    docx_filename: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ClipResult":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _numeric(clips: list[ClipResult], attr: str) -> list[float]:
    return [v for c in clips if (v := getattr(c, attr)) is not None]


@dataclass
class BenchmarkResult:
    tier: str
    seed: int | None
    tag: str
    asr_model: str
    llm_model: str
    refine_prompt_id: str
    summarize_prompt_id: str
    summary_language: str
    dataset: str = "fleurs"
    clips: list[ClipResult] = field(default_factory=list)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    settings_snapshot: dict[str, Any] = field(default_factory=dict)
    peak_rss_mb: float | None = None
    model_load_seconds: dict[str, float] = field(default_factory=dict)

    # --- aggregates ------------------------------------------------------

    @property
    def n_clips(self) -> int:
        return len(self.clips)

    @property
    def n_errors(self) -> int:
        return sum(1 for c in self.clips if c.error)

    def aggregate(self, attr: str) -> dict[str, float | None]:
        values = _numeric(self.clips, attr)
        if not values:
            return {"mean": None, "median": None, "p50": None, "p95": None}
        return {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "p50": percentile(values, 50),
            "p95": percentile(values, 95),
        }

    AGGREGATE_METRICS = (
        "wer_raw", "wer_refined", "cer_raw", "cer_refined",
        "semdist_raw", "semdist_refined", "content_recall", "length_ratio",
        "asr_seconds", "refine_seconds", "summarize_seconds", "realtime_factor",
        "description_transcript_semdist",
        "refine_tokens_per_second", "summarize_tokens_per_second",
    )

    def aggregates(self) -> dict[str, dict[str, float | None]]:
        return {attr: self.aggregate(attr) for attr in self.AGGREGATE_METRICS}

    def schema_valid_rate(self) -> float | None:
        flags = [c.schema_valid for c in self.clips if c.schema_valid is not None]
        return (sum(flags) / len(flags)) if flags else None

    def retry_rate(self) -> float | None:
        flags = [c.used_retry for c in self.clips if c.used_retry is not None]
        return (sum(flags) / len(flags)) if flags else None

    def fallback_rate(self) -> float | None:
        flags = [c.used_fallback for c in self.clips if c.used_fallback is not None]
        return (sum(flags) / len(flags)) if flags else None

    def slug_uniqueness(self) -> float | None:
        names = [c.docx_filename for c in self.clips if c.docx_filename]
        if not names:
            return None
        return len(set(names)) / len(names)

    def summary_metrics(self) -> dict[str, float | None]:
        return {
            "schema_valid_rate": self.schema_valid_rate(),
            "retry_rate": self.retry_rate(),
            "fallback_rate": self.fallback_rate(),
            "slug_uniqueness": self.slug_uniqueness(),
            "title_len_ok_rate": _rate(self.clips, "title_len_ok"),
            "description_len_ok_rate": _rate(self.clips, "description_len_ok"),
            "topic_count_ok_rate": _rate(self.clips, "topic_count_ok"),
        }

    def flat_metrics(self) -> dict[str, float]:
        """Flatten aggregates + summary metrics into MLflow's flat
        metric-name -> float shape (e.g. `wer_refined_mean`)."""
        out: dict[str, float] = {}
        for name, stats in self.aggregates().items():
            for stat_name, value in stats.items():
                if value is not None:
                    out[f"{name}_{stat_name}"] = value
        for name, value in self.summary_metrics().items():
            if value is not None:
                out[name] = value
        out["n_clips"] = float(self.n_clips)
        out["n_errors"] = float(self.n_errors)
        if self.peak_rss_mb is not None:
            out["peak_rss_mb"] = self.peak_rss_mb
        for stage, seconds in self.model_load_seconds.items():
            out[f"model_load_seconds_{stage}"] = seconds
        return out

    # --- (de)serialization -------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "tier": self.tier,
            "seed": self.seed,
            "tag": self.tag,
            "asr_model": self.asr_model,
            "llm_model": self.llm_model,
            "refine_prompt_id": self.refine_prompt_id,
            "summarize_prompt_id": self.summarize_prompt_id,
            "summary_language": self.summary_language,
            "dataset": self.dataset,
            "generated_at": self.generated_at,
            "settings_snapshot": self.settings_snapshot,
            "peak_rss_mb": self.peak_rss_mb,
            "model_load_seconds": self.model_load_seconds,
            "clips": [c.to_dict() for c in self.clips],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkResult":
        clips = [ClipResult.from_dict(c) for c in data.get("clips", [])]
        return cls(
            tier=data["tier"],
            seed=data.get("seed"),
            tag=data.get("tag", ""),
            asr_model=data["asr_model"],
            llm_model=data["llm_model"],
            refine_prompt_id=data.get("refine_prompt_id", ""),
            summarize_prompt_id=data.get("summarize_prompt_id", ""),
            summary_language=data.get("summary_language", "pt"),
            dataset=data.get("dataset", "fleurs"),
            clips=clips,
            generated_at=data.get("generated_at", ""),
            settings_snapshot=data.get("settings_snapshot", {}),
            peak_rss_mb=data.get("peak_rss_mb"),
            model_load_seconds=data.get("model_load_seconds", {}),
        )

    # --- reporting -----------------------------------------------------

    def markdown_table(self) -> str:
        lines = [
            f"# Benchmark: {self.tag} ({self.dataset}/{self.tier}, n={self.n_clips}, seed={self.seed})",
            "",
            f"- ASR model: `{self.asr_model}`",
            f"- LLM model: `{self.llm_model}`",
            f"- Refine prompt: `{self.refine_prompt_id}` · Summarize prompt: `{self.summarize_prompt_id}`",
            f"- Generated: {self.generated_at}",
            f"- Errors: {self.n_errors}/{self.n_clips}",
            "",
            "## Aggregate metrics",
            "",
            "| metric | mean | median | p95 |",
            "|---|---|---|---|",
        ]
        for name, stats in self.aggregates().items():
            mean = f"{stats['mean']:.4f}" if stats["mean"] is not None else "—"
            median = f"{stats['median']:.4f}" if stats["median"] is not None else "—"
            p95 = f"{stats['p95']:.4f}" if stats["p95"] is not None else "—"
            lines.append(f"| {name} | {mean} | {median} | {p95} |")

        lines += ["", "## Summary-stage metrics", "", "| metric | value |", "|---|---|"]
        for name, value in self.summary_metrics().items():
            lines.append(f"| {name} | {value:.4f} |" if value is not None else f"| {name} | — |")

        lines += [
            "", "## Worst clips by WER (refined)", "",
            "| clip | duration | wer_refined | wer_raw | error |",
            "|---|---|---|---|---|",
        ]
        worst = sorted(
            (c for c in self.clips if c.wer_refined is not None),
            key=lambda c: c.wer_refined, reverse=True,
        )[:10]
        for c in worst:
            wer_raw = f"{c.wer_raw:.4f}" if c.wer_raw is not None else "—"
            lines.append(
                f"| {c.clip_id} | {c.duration:.1f}s | {c.wer_refined:.4f} | "
                f"{wer_raw} | {c.error or ''} |"
            )
        return "\n".join(lines) + "\n"


def _rate(clips: list[ClipResult], attr: str) -> float | None:
    flags = [getattr(c, attr) for c in clips if getattr(c, attr) is not None]
    return (sum(flags) / len(flags)) if flags else None

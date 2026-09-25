"""Regression gate: diff two BenchmarkResults, decide pass/fail.

This is what `scripts/benchmark.py compare` turns into a process exit code
for CI to consume. Kept independent of MLflow -- it works on two
`BenchmarkResult` objects however they were loaded (local JSON or an MLflow
artifact download), so CI doesn't need a tracking server reachable to gate a
PR.
"""

from __future__ import annotations

from dataclasses import dataclass

from .results import BenchmarkResult

# Metrics where an *increase* from baseline to candidate is a regression.
# WER/CER: lower is better. semdist: lower (closer meaning) is better.
_LOWER_IS_BETTER = (
    "wer_raw",
    "wer_refined",
    "cer_raw",
    "cer_refined",
    "semdist_raw",
    "semdist_refined",
)


@dataclass
class MetricDiff:
    metric: str
    baseline: float
    candidate: float
    delta: float
    regressed: bool


@dataclass
class ComparisonResult:
    baseline_tag: str
    candidate_tag: str
    threshold: float
    diffs: list[MetricDiff]

    @property
    def regressed(self) -> bool:
        return any(d.regressed for d in self.diffs)

    def markdown_table(self) -> str:
        lines = [
            f"# Compare: {self.baseline_tag} -> {self.candidate_tag}",
            f"(regression threshold: +{self.threshold:.4f} absolute)",
            "",
            "| metric | baseline | candidate | delta | regressed |",
            "|---|---|---|---|---|",
        ]
        for d in self.diffs:
            flag = "YES" if d.regressed else ""
            lines.append(
                f"| {d.metric} | {d.baseline:.4f} | {d.candidate:.4f} | {d.delta:+.4f} | {flag} |"
            )
        return "\n".join(lines) + "\n"


def compare_runs(
    baseline: BenchmarkResult,
    candidate: BenchmarkResult,
    *,
    threshold: float = 0.02,
) -> ComparisonResult:
    """Compare mean WER/CER/SemDist (raw and refined) between two runs.

    A metric "regresses" if it moves in the worse direction by more than
    `threshold` (absolute, e.g. 0.02 = 2 percentage points of WER) *and*
    both runs actually measured it (a metric missing from either run -- e.g.
    comparing a `smoke` run against a `quick` run where one lacks refine
    data -- is skipped rather than treated as a regression by omission).
    """
    diffs: list[MetricDiff] = []
    baseline_agg, candidate_agg = baseline.aggregates(), candidate.aggregates()

    for metric in _LOWER_IS_BETTER:
        base_mean = baseline_agg.get(metric, {}).get("mean")
        cand_mean = candidate_agg.get(metric, {}).get("mean")
        if base_mean is None or cand_mean is None:
            continue
        delta = cand_mean - base_mean
        diffs.append(
            MetricDiff(
                metric=metric,
                baseline=base_mean,
                candidate=cand_mean,
                delta=delta,
                regressed=delta > threshold,
            )
        )

    return ComparisonResult(
        baseline_tag=baseline.tag,
        candidate_tag=candidate.tag,
        threshold=threshold,
        diffs=diffs,
    )

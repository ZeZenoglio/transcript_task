"""Pure logic for the Streamlit demo (Phase 9), kept separate from app.py
specifically so it's unit-testable without Streamlit's runtime -- see
PLAN.md's Phase 9 testing note: "logic extracted into pure helpers and
unit-tested" (Streamlit UIs themselves resist deep automated testing).
"""

from __future__ import annotations

STAGE_ORDER = (
    "queued", "normalizing", "transcribing", "refining", "summarizing", "writing_docx", "done",
)
TERMINAL_STATUSES = frozenset({"done", "failed", "canceled"})


def is_terminal(status: str) -> bool:
    return status in TERMINAL_STATUSES


def stage_progress(status: str) -> float:
    """0.0-1.0 for a progress bar, based on where `status` falls in the
    pipeline's stage order. `failed`/`canceled` return 1.0 -- the job isn't
    going to move further, so a partially-filled bar would just look stuck."""
    if status in ("failed", "canceled"):
        return 1.0
    try:
        idx = STAGE_ORDER.index(status)
    except ValueError:
        return 0.0
    return (idx + 1) / len(STAGE_ORDER)


def pick_transcript(result: dict, view: str) -> str | None:
    """`view` is "raw" or "refined". Falls back to raw if "refined" was
    requested but unavailable (refine wasn't requested for this job, or the
    refine-quality guard rejected the result -- see `result["refine_rejected"]`)."""
    if view == "refined" and result.get("refined_transcript"):
        return result["refined_transcript"]
    return result.get("raw_transcript")


def refined_available(result: dict) -> bool:
    return bool(result.get("refined_transcript"))


def sensitivity_banner(summary: dict | None) -> str | None:
    """A short warning string when the summary flags high sensitivity, or
    None otherwise -- PLAN.md's Phase 9 note asks specifically for a banner
    "when the summary flags high". `docx_writer.py`'s own banner (Phase 3)
    covers both "medium" and "high"; this demo follows the plan's wording
    literally rather than silently widening scope."""
    if summary and summary.get("sensitivity") == "high":
        return ("⚠ This recording may contain sensitive information (personal, "
                "financial, medical, or legal). Review before sharing.")
    return None


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes}:{secs:02d}"


def docx_download_filename(job: dict) -> str:
    stem = (job.get("filename") or "transcript").rsplit(".", 1)[0]
    return f"{stem}.docx"

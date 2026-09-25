"""Runs one API job through the same pipeline stages the CLI uses.

Deliberately reuses `pipeline.py`'s stage_* functions rather than
duplicating their logic (unlike `eval/runner.py`, which has a real reason to
bypass them -- scoring against ground truth is a different concern). An API
job *is* the same thing as a CLI run on one file; the only difference is
each job gets its own isolated `tmp`/`output` directories (via a per-job
`Settings`) instead of sharing the CLI's, so concurrent jobs never collide
on `extract`/`normalize` paths or `transcripts.json`.

`state["items"][filename]["transcript_id"]` is pre-seeded with the job id
before calling stage_normalize, so the id `pipeline.py` would otherwise
generate randomly is instead the job's own id (see PLAN.md's Phase 7 note:
"jobs.id IS the transcript_id").
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

from ..asr import Transcriber
from ..pipeline import stage_docx, stage_normalize, stage_refine, stage_summarize, stage_transcribe
from ..refine import ChatModel
from ..settings import Settings
from .db import JobStatus


class JobStageError(Exception):
    """Raised when a stage records an error on the item instead of raising
    an exception itself (normalize/transcribe do this to keep a CLI batch
    going; a single-job API run has nothing to keep going for, so this
    turns that into something the worker can catch and record as failed)."""

    def __init__(self, stage: str, detail: str) -> None:
        self.stage = stage
        self.detail = detail
        super().__init__(f"{stage}: {detail}")


def job_settings(base: Settings, job_id: str) -> Settings:
    """A per-job Settings with isolated tmp/output dirs, everything else
    inherited from the server's current configuration (base)."""
    job_dir = base.api_jobs_dir / job_id
    return base.model_copy(update={
        "tmp_dir": job_dir / "tmp",
        "output_dir": job_dir / "output",
    })


def run_job(
    job_id: str,
    audio_path: Path,
    filename: str,
    settings: Settings,
    *,
    refine: bool,
    on_stage: Callable[[JobStatus], None] | None = None,
    transcriber: Transcriber | None = None,
    chat_model: ChatModel | None = None,
) -> dict:
    """Run extract(already done by the caller) -> normalize -> transcribe ->
    (refine) -> summarize -> docx for one file. Returns the final item dict
    (same shape as a transcripts.json entry). Raises JobStageError if a
    stage records a failure rather than completing.

    `transcriber`/`chat_model` are optional, purely for tests -- same
    Protocol-based injection pattern as `stage_transcribe`/`stage_refine`
    themselves (see pipeline.py); real callers leave them as None and let
    those stage functions construct the real ones.
    """
    def notify(status: JobStatus) -> None:
        if on_stage is not None:
            on_stage(status)

    state = {"items": {filename: {"transcript_id": job_id}}}

    notify(JobStatus.normalizing)
    stage_normalize([audio_path], state, settings, force=True)
    item = state["items"][filename]
    if item.get("error"):
        raise JobStageError("normalize", item["error"])

    notify(JobStatus.transcribing)
    stage_transcribe(state, settings, force=True, transcriber=transcriber)
    item = state["items"][filename]
    if item.get("error"):
        raise JobStageError("transcribe", item["error"])

    if refine:
        notify(JobStatus.refining)
        stage_refine(state, settings, force=True, model=chat_model)
        # Refine failing outright (e.g. Ollama unreachable) is not fatal for
        # the job -- same as the CLI, the raw transcript still stands on its
        # own (see item["refine_error"]). Not the same thing as the refine
        # *quality guard* rejecting a result (refine_rejected), which is a
        # normal, successful outcome, not an error.

    notify(JobStatus.summarizing)
    stage_summarize(state, settings, force=True, model=chat_model)

    notify(JobStatus.writing_docx)
    stage_docx(state, settings)

    return state["items"][filename]


def cleanup_job_files(base: Settings, job_id: str) -> None:
    """Remove a job's isolated tmp/output directory tree (used by DELETE)."""
    job_dir = base.api_jobs_dir / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)

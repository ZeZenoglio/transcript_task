"""Persistence (Phase 7): SQLite via SQLModel.

A single-machine local tool has no multi-writer problem to solve, so this
is plain SQLite -- no server, same reasoning as the MLflow file-store
decision in Phase 6. `Job.id` is the same `transcript_id` used everywhere
else in this project (filenames, docx provenance headers,
`transcripts.json`), not a fresh autoincrement, so a job, a file, and a
transcript record for the same recording are always addressable by the
same short id.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import JSON, Column
from sqlmodel import Field, Session, SQLModel, create_engine

from ..settings import Settings


def _now() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, Enum):
    queued = "queued"
    normalizing = "normalizing"
    transcribing = "transcribing"
    refining = "refining"
    summarizing = "summarizing"
    writing_docx = "writing_docx"
    done = "done"
    failed = "failed"
    canceled = "canceled"


TERMINAL_STATUSES = frozenset({JobStatus.done, JobStatus.failed, JobStatus.canceled})


class BenchmarkStatus(str, Enum):
    running = "running"
    done = "done"
    failed = "failed"


class Job(SQLModel, table=True):
    id: str = Field(primary_key=True)  # transcript_id (8-char hex)
    filename: str
    status: JobStatus = JobStatus.queued
    refine_requested: bool = True
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    error: str | None = None

    # The config this job actually ran under (see PLAN.md's Phase 7 note:
    # "a job must record the config it ran under, not the current one" --
    # config can change between when a job is queued and when it runs).
    asr_model: str = ""
    llm_model: str = ""
    refine_prompt_id: str = ""
    summarize_prompt_id: str = ""

    duration_seconds: float | None = None
    source_format: str | None = None

    raw_transcript: str | None = None
    refined_transcript: str | None = None
    refine_rejected: dict | None = Field(default=None, sa_column=Column(JSON))
    summary: dict | None = Field(default=None, sa_column=Column(JSON))
    docx_path: str | None = None


class StageTiming(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    job_id: str = Field(foreign_key="job.id", index=True)
    stage: str
    seconds: float
    recorded_at: datetime = Field(default_factory=_now)


class BenchmarkRun(SQLModel, table=True):
    id: str = Field(primary_key=True)  # the run's tag
    dataset: str
    tier: str
    status: BenchmarkStatus = BenchmarkStatus.running
    created_at: datetime = Field(default_factory=_now)
    finished_at: datetime | None = None
    error: str | None = None
    results_path: str | None = None


class ConfigHistory(SQLModel, table=True):
    """Append-only. `changes` is the human-readable diff for an audit trail;
    `snapshot` is the *full* resulting config, which is what lets the
    server reconstruct its current settings on restart by just reading the
    latest row rather than replaying every change ever made."""
    id: int | None = Field(default=None, primary_key=True)
    changed_at: datetime = Field(default_factory=_now)
    changes: dict = Field(sa_column=Column(JSON))  # {"field": {"from": ..., "to": ...}}
    snapshot: dict = Field(sa_column=Column(JSON))


_engine = None


def get_engine(settings: Settings):
    """Module-level singleton engine, built lazily from Settings so tests
    can point it at a temp DB by constructing a fresh Settings and calling
    init_db() explicitly rather than relying on process-wide state."""
    global _engine
    if _engine is None:
        settings.api_db_path.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(settings.api_db_url, connect_args={"check_same_thread": False})
    return _engine


def init_db(settings: Settings) -> None:
    SQLModel.metadata.create_all(get_engine(settings))


def reset_engine_for_tests() -> None:
    """Tests construct a fresh Settings pointing at a temp DB per test; this
    drops the cached engine so get_engine() rebuilds against the new path
    instead of silently reusing whatever the first test created."""
    global _engine
    _engine = None


def get_session(settings: Settings) -> Session:
    return Session(get_engine(settings))


def latest_config_snapshot(settings: Settings) -> dict | None:
    """The most recently persisted config override set, or None if
    `PATCH /v1/config` has never been called -- callers fall back to
    Settings()'s own defaults/env in that case."""
    from sqlmodel import select

    with get_session(settings) as session:
        row = session.exec(
            select(ConfigHistory).order_by(ConfigHistory.id.desc())  # type: ignore[union-attr]
        ).first()
        return row.snapshot if row else None

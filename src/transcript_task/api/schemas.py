"""Request/response models -- kept separate from the SQLModel tables in
db.py. A DB row and an API response often need to shape the same data
differently (e.g. `docx_path` on disk vs. a boolean `docx_available` plus a
download link), so this project doesn't reuse table models as response
models directly.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from .db import BenchmarkStatus, JobStatus

# The subset of Settings fields exposed for live config mutation. Not every
# field belongs here -- paths and the DB location aren't something a running
# server should let a client repoint at will.
CONFIGURABLE_FIELDS = (
    "asr_model", "llm_model", "refine_prompt_id", "summary_language",
    "llm_temperature", "llm_num_ctx", "llm_num_predict", "anonymize_metadata",
)


class Problem(BaseModel):
    """RFC 7807 "Problem Details for HTTP APIs" (https://www.rfc-editor.org/rfc/rfc7807)
    -- every non-2xx response from this API has this shape, served as
    `application/problem+json`. `type` is left as the RFC's own "about:blank"
    default: the HTTP status code plus `title` already say what kind of
    problem this is, and none of this API's errors need a more specific,
    dereferenceable problem-type URI than that."""

    type: str = Field(default="about:blank", description="A URI identifying the problem type.")
    title: str = Field(description="A short, human-readable summary of the problem type.")
    status: int = Field(description="The HTTP status code, repeated here for a body-only consumer.")
    detail: str = Field(description="A human-readable explanation specific to this occurrence.")
    instance: str | None = Field(default=None, description="The request path that produced this problem.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "about:blank",
                    "title": "Not Found",
                    "status": 404,
                    "detail": "no such job: deadbeef",
                    "instance": "/v1/jobs/deadbeef",
                }
            ]
        }
    }


class JobCreateResponse(BaseModel):
    job_id: str = Field(description="Also the transcript_id used everywhere else for this "
                                     "recording: the docx filename, its provenance header, etc.")
    status: JobStatus

    model_config = {
        "json_schema_extra": {"examples": [{"job_id": "a1b2c3d4", "status": "queued"}]}
    }


class JobSummary(BaseModel):
    job_id: str = Field(validation_alias="id")
    filename: str
    status: JobStatus
    refine_requested: bool
    created_at: datetime
    updated_at: datetime
    error: str | None = None

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "examples": [{
                "job_id": "a1b2c3d4", "filename": "interview.m4a", "status": "transcribing",
                "refine_requested": True,
                "created_at": "2026-09-25T10:00:00Z", "updated_at": "2026-09-25T10:00:03Z",
                "error": None,
            }]
        },
    }


class SummaryOut(BaseModel):
    """A read-only mirror of `summarize.TranscriptSummary` with field
    descriptions added for the API docs -- kept separate from that model
    rather than adding descriptions there directly, since
    `TranscriptSummary.model_json_schema()` is also what's sent to Ollama as
    the structured-output constraint (see summarize.py); this API-facing
    copy can be documented freely without risking an untested change to
    that already-tuned production path."""

    title: str = Field(description="Short, filename-safe title (see docx_filename in summarize.py).")
    description: str = Field(description="3-6 sentences describing what the recording is about.")
    topics: list[str] = Field(description="3-8 keywords or themes.")
    speakers_detected: int = Field(description="Estimated distinct speakers; 0 if not estimable.")
    language_variant: Literal["pt-PT", "pt-BR", "unknown"]
    sensitivity: Literal["low", "medium", "high"] = Field(
        description="'medium'/'high' flags personal, financial, medical, or legal content."
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="The model's own confidence in this summary; 'low' means treat it as a stub."
    )


class RefineRejected(BaseModel):
    """Present only when the refine-quality guard (see PLAN.md's Phase 6
    follow-up) discarded the refine attempt and fell back to the raw
    transcript -- `text` is what was generated but rejected, kept here so a
    caller can inspect it rather than just being told "it failed"."""

    reason: Literal["content_recall too low", "length_ratio out of bounds"]
    content_recall: float = Field(ge=0, le=1)
    length_ratio: float = Field(ge=0)
    text: str = Field(description="The rejected refine output, kept for inspection.")


class JobResult(BaseModel):
    job_id: str = Field(validation_alias="id")
    filename: str
    status: JobStatus
    error: str | None = Field(default=None, description="Set only when status is 'failed'.")
    duration_seconds: float | None = Field(default=None, description="Audio duration, in seconds.")
    source_format: str | None = None
    raw_transcript: str | None = Field(
        default=None, description="Unedited ASR output. Always present once transcription succeeds."
    )
    refined_transcript: str | None = Field(
        default=None,
        description="Present only if `refine=true` was requested and the refine-quality guard "
                    "accepted the result. See `refine_rejected` for why it might be absent.",
    )
    refine_rejected: RefineRejected | None = None
    summary: SummaryOut | None = None
    docx_available: bool = Field(
        default=False, description="Whether GET /v1/jobs/{job_id}/docx will return a document."
    )

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "examples": [{
                "job_id": "a1b2c3d4", "filename": "interview.m4a", "status": "done", "error": None,
                "duration_seconds": 184.2, "source_format": "aac",
                "raw_transcript": "boa tarde eh queria confirmar a reuniao de amanha",
                "refined_transcript": "Boa tarde. Queria confirmar a reunião de amanhã.",
                "refine_rejected": None,
                "summary": {
                    "title": "Confirmação de reunião", "description": "Uma chamada curta confirmando o horário de uma reunião.",
                    "topics": ["reunião", "agenda"], "speakers_detected": 1,
                    "language_variant": "pt-PT", "sensitivity": "low", "confidence": "high",
                },
                "docx_available": True,
            }]
        },
    }


class StageTimingOut(BaseModel):
    stage: Literal["transcribe", "refine", "summarize"]
    seconds: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = Field(
        description="'degraded' means at least one dependency (ffmpeg/ffprobe/Ollama) is unreachable; "
                    "jobs will fail at whichever stage needs the missing one."
    )
    ffmpeg: bool
    ffprobe: bool
    ollama_reachable: bool
    ollama_models: list[str]
    asr_model: str
    llm_model: str


class ConfigResponse(BaseModel):
    asr_model: str
    llm_model: str
    refine_prompt_id: str
    summary_language: Literal["pt", "en"]
    llm_temperature: float
    llm_num_ctx: int
    llm_num_predict: int
    anonymize_metadata: bool
    # Read-only: informational, not part of CONFIGURABLE_FIELDS/ConfigPatch --
    # exposed here so upload constraints are discoverable from the API
    # itself rather than a description that can drift out of date.
    api_max_upload_mb: int = Field(description="POST /v1/jobs rejects uploads larger than this.")
    audio_extensions: list[str] = Field(description="File extensions POST /v1/jobs accepts.")


class ConfigPatch(BaseModel):
    """All fields optional -- PATCH semantics, only supplied fields change."""
    asr_model: str | None = None
    llm_model: str | None = Field(default=None, description="Validated against `ollama list` before being accepted.")
    refine_prompt_id: str | None = Field(default=None, description="Must be a known id -- see prompts.py.")
    summary_language: Literal["pt", "en"] | None = None
    llm_temperature: float | None = Field(default=None, ge=0, le=2)
    llm_num_ctx: int | None = Field(default=None, gt=0)
    llm_num_predict: int | None = Field(default=None, gt=0)
    anonymize_metadata: bool | None = None

    model_config = {
        "json_schema_extra": {"examples": [{"llm_model": "qwen3.5:4b"}]}
    }


class ModelInfo(BaseModel):
    name: str
    size_bytes: int | None = None


class BenchmarkCreateRequest(BaseModel):
    dataset: Literal["fleurs", "common_voice"] = "fleurs"
    tier: Literal["smoke", "quick", "full"] = "smoke"
    tag: str = Field(description="A unique name for this run, e.g. 'qwen9b-baseline'. Re-used to fetch results.")
    n: int = Field(default=30, gt=0, description="Sample size for the 'quick' tier; ignored otherwise.")
    seed: int = Field(default=42, description="Sampling seed for the 'quick' tier, for reproducibility.")
    skip_refine: bool = False
    # SemDist loads sentence-transformers (sklearn/joblib/onnx...) on first
    # use -- a real, heavy, one-time cost. Defaulting to on matches the CLI;
    # exposed here so a caller (or a test) can skip paying it.
    semdist: bool = Field(default=True, description="Compute SemDist (loads a sentence-embedding model on first use).")

    model_config = {
        "json_schema_extra": {"examples": [{"dataset": "fleurs", "tier": "smoke", "tag": "qwen9b-baseline"}]}
    }


class BenchmarkRunOut(BaseModel):
    tag: str = Field(validation_alias="id")
    dataset: str
    tier: str
    status: BenchmarkStatus
    created_at: datetime
    finished_at: datetime | None = None
    error: str | None = None

    model_config = {"populate_by_name": True}

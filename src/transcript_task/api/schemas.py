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

from .db import JobStatus

# The subset of Settings fields exposed for live config mutation. Not every
# field belongs here -- paths and the DB location aren't something a running
# server should let a client repoint at will.
CONFIGURABLE_FIELDS = (
    "asr_model", "llm_model", "refine_prompt_id", "summary_language",
    "llm_temperature", "llm_num_ctx", "llm_num_predict", "anonymize_metadata",
)


class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobSummary(BaseModel):
    job_id: str = Field(validation_alias="id")
    filename: str
    status: JobStatus
    refine_requested: bool
    created_at: datetime
    updated_at: datetime
    error: str | None = None

    model_config = {"populate_by_name": True}


class JobResult(BaseModel):
    job_id: str = Field(validation_alias="id")
    filename: str
    status: JobStatus
    error: str | None = None
    duration_seconds: float | None = None
    source_format: str | None = None
    raw_transcript: str | None = None
    refined_transcript: str | None = None
    refine_rejected: dict | None = None
    summary: dict | None = None
    docx_available: bool = False

    model_config = {"populate_by_name": True}


class StageTimingOut(BaseModel):
    stage: str
    seconds: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
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


class ConfigPatch(BaseModel):
    """All fields optional -- PATCH semantics, only supplied fields change."""
    asr_model: str | None = None
    llm_model: str | None = None
    refine_prompt_id: str | None = None
    summary_language: Literal["pt", "en"] | None = None
    llm_temperature: float | None = None
    llm_num_ctx: int | None = None
    llm_num_predict: int | None = None
    anonymize_metadata: bool | None = None


class ModelInfo(BaseModel):
    name: str
    size_bytes: int | None = None


class BenchmarkCreateRequest(BaseModel):
    dataset: Literal["fleurs", "common_voice"] = "fleurs"
    tier: Literal["smoke", "quick", "full"] = "smoke"
    tag: str
    n: int = 30
    seed: int = 42
    skip_refine: bool = False
    # SemDist loads sentence-transformers (sklearn/joblib/onnx...) on first
    # use -- a real, heavy, one-time cost. Defaulting to on matches the CLI;
    # exposed here so a caller (or a test) can skip paying it.
    semdist: bool = True


class BenchmarkRunOut(BaseModel):
    tag: str = Field(validation_alias="id")
    dataset: str
    tier: str
    status: str
    created_at: datetime
    finished_at: datetime | None = None
    error: str | None = None

    model_config = {"populate_by_name": True}

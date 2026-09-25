"""FastAPI service (Phase 7).

Run with:
    uv run uvicorn transcript_task.api.app:app --reload

Async model: transcription is minutes-long, so endpoints never block on it.
`POST /v1/jobs` returns `202` with a `job_id` immediately; the actual work
runs in `JobWorker`'s thread pool (concurrency capped by
`Settings.api_concurrency`, since ASR+LLM calls are synchronous and
CPU/GPU-bound, not something asyncio helps with). Poll `GET /v1/jobs/{id}`
for status, then `GET /v1/jobs/{id}/result` or `/docx`.
"""

from __future__ import annotations

import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlmodel import Session, select

from ..pipeline import new_transcript_id
from ..prompts import get_refine_template, get_summarize_template
from ..settings import PROJECT_ROOT, Settings
from .db import (
    TERMINAL_STATUSES,
    BenchmarkRun,
    ConfigHistory,
    Job,
    JobStatus,
    StageTiming,
    get_session,
    init_db,
    latest_config_snapshot,
)
from .job_runner import job_settings
from .logging_config import RequestIDMiddleware, configure_logging, get_logger
from .schemas import (
    CONFIGURABLE_FIELDS,
    BenchmarkCreateRequest,
    BenchmarkRunOut,
    ConfigPatch,
    ConfigResponse,
    HealthResponse,
    JobCreateResponse,
    JobResult,
    JobSummary,
    ModelInfo,
    StageTimingOut,
)
from .worker import JobWorker, purge_job

logger = get_logger("transcript_task.api")


def build_settings() -> Settings:
    """Settings for a fresh server start: env/.env defaults, with the most
    recently PATCHed config (if any) layered on top so a restart doesn't
    silently forget a live config change."""
    base = Settings()
    init_db(base)  # tables must exist before latest_config_snapshot can query them
    snapshot = latest_config_snapshot(base)
    return base.model_copy(update=snapshot) if snapshot else base


def create_worker(settings: Settings) -> JobWorker:
    """A seam for tests: monkeypatching this (not JobWorker directly) lets a
    test swap in fakes for the ASR/LLM Protocols without touching the real
    startup path production runs through."""
    return JobWorker(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = build_settings()
    configure_logging(settings)
    init_db(settings)
    app.state.settings = settings
    app.state.worker = create_worker(settings)
    logger.info("api_startup", asr_model=settings.asr_model, llm_model=settings.llm_model,
                concurrency=settings.api_concurrency)
    yield
    app.state.worker.shutdown()
    logger.info("api_shutdown")


app = FastAPI(title="transcript_task API", lifespan=lifespan)
app.add_middleware(RequestIDMiddleware)


def _session(app_: FastAPI) -> Session:
    return get_session(app_.state.settings)


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = app.state.settings
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    ffprobe_ok = shutil.which("ffprobe") is not None

    ollama_reachable = False
    ollama_models: list[str] = []
    try:
        import ollama

        response = ollama.list()
        ollama_models = [m.model for m in response.models]
        ollama_reachable = True
    except Exception:  # noqa: BLE001 - health check must never 500
        pass

    status = "ok" if (ffmpeg_ok and ffprobe_ok and ollama_reachable) else "degraded"
    return HealthResponse(
        status=status, ffmpeg=ffmpeg_ok, ffprobe=ffprobe_ok,
        ollama_reachable=ollama_reachable, ollama_models=ollama_models,
        asr_model=settings.asr_model, llm_model=settings.llm_model,
    )


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------

def _new_job_dir(settings: Settings, job_id: str) -> Path:
    js = job_settings(settings, job_id)
    js.extract_dir.mkdir(parents=True, exist_ok=True)
    return js.extract_dir


@app.post("/v1/jobs", response_model=JobCreateResponse, status_code=202)
async def create_job(
    file: Annotated[UploadFile, File()],
    refine: Annotated[bool, Form()] = True,
) -> JobCreateResponse:
    settings: Settings = app.state.settings
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in settings.audio_extensions:
        raise HTTPException(400, f"unsupported audio extension {suffix!r}")

    job_id = new_transcript_id()
    max_bytes = settings.api_max_upload_mb * 1024 * 1024
    dest_dir = _new_job_dir(settings, job_id)
    dest = dest_dir / file.filename
    written = 0
    with open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                f.close()
                shutil.rmtree(dest_dir.parent, ignore_errors=True)
                raise HTTPException(413, f"upload exceeds {settings.api_max_upload_mb} MB limit")
            f.write(chunk)

    with _session(app) as session:
        job = Job(
            id=job_id, filename=file.filename, status=JobStatus.queued,
            refine_requested=refine,
            asr_model=settings.asr_model, llm_model=settings.llm_model,
            refine_prompt_id=settings.refine_prompt_id,
            summarize_prompt_id=get_summarize_template(settings.summary_language).id,
        )
        session.add(job)
        session.commit()

    logger.info("job_submitted", job_id=job_id, filename=file.filename, refine=refine, bytes=written)
    app.state.worker.submit(job_id, dest, file.filename, refine)
    return JobCreateResponse(job_id=job_id, status=JobStatus.queued)


@app.get("/v1/jobs", response_model=list[JobSummary])
def list_jobs(limit: int = 50) -> list[JobSummary]:
    with _session(app) as session:
        jobs = session.exec(select(Job).order_by(Job.created_at.desc()).limit(limit)).all()  # type: ignore[union-attr]
        return [JobSummary.model_validate(j, from_attributes=True) for j in jobs]


def _get_job_or_404(session: Session, job_id: str) -> Job:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, f"no such job: {job_id}")
    return job


@app.get("/v1/jobs/{job_id}", response_model=JobSummary)
def get_job(job_id: str) -> JobSummary:
    with _session(app) as session:
        job = _get_job_or_404(session, job_id)
        return JobSummary.model_validate(job, from_attributes=True)


@app.get("/v1/jobs/{job_id}/timings", response_model=list[StageTimingOut])
def get_job_timings(job_id: str) -> list[StageTimingOut]:
    with _session(app) as session:
        _get_job_or_404(session, job_id)
        timings = session.exec(select(StageTiming).where(StageTiming.job_id == job_id)).all()
        return [StageTimingOut(stage=t.stage, seconds=t.seconds) for t in timings]


@app.get("/v1/jobs/{job_id}/result", response_model=JobResult)
def get_job_result(job_id: str) -> JobResult:
    with _session(app) as session:
        job = _get_job_or_404(session, job_id)
        return JobResult.model_validate(
            {**job.model_dump(), "docx_available": bool(job.docx_path)},
            from_attributes=True,
        )


@app.get("/v1/jobs/{job_id}/docx")
def get_job_docx(job_id: str) -> FileResponse:
    with _session(app) as session:
        job = _get_job_or_404(session, job_id)
    if not job.docx_path:
        raise HTTPException(404, "no document available for this job (not done, or refine/summarize failed)")
    path = PROJECT_ROOT / job.docx_path
    if not path.exists():
        raise HTTPException(404, "document record exists but the file is missing on disk")
    return FileResponse(path, filename=path.name,
                         media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")


@app.delete("/v1/jobs/{job_id}", status_code=204)
def delete_job(job_id: str) -> None:
    worker: JobWorker = app.state.worker
    with _session(app) as session:
        job = _get_job_or_404(session, job_id)
        status = job.status

    if status not in TERMINAL_STATUSES:
        canceled = worker.cancel(job_id)
        if not canceled:
            raise HTTPException(
                409, "job is currently running and cannot be interrupted -- "
                     "retry once it reaches a terminal status"
            )
        with _session(app) as session:
            job = session.get(Job, job_id)
            if job is not None:
                job.status = JobStatus.canceled
                session.add(job)
                session.commit()

    purge_job(app.state.settings, job_id)
    logger.info("job_deleted", job_id=job_id)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@app.get("/v1/config", response_model=ConfigResponse)
def get_config() -> ConfigResponse:
    settings: Settings = app.state.settings
    return ConfigResponse(**{f: getattr(settings, f) for f in CONFIGURABLE_FIELDS})


@app.patch("/v1/config", response_model=ConfigResponse)
def patch_config(patch: ConfigPatch) -> ConfigResponse:
    settings: Settings = app.state.settings
    updates = patch.model_dump(exclude_unset=True)
    if not updates:
        return ConfigResponse(**{f: getattr(settings, f) for f in CONFIGURABLE_FIELDS})

    if "refine_prompt_id" in updates:
        try:
            get_refine_template(updates["refine_prompt_id"])
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    if "llm_model" in updates or "asr_model" in updates:
        try:
            import ollama

            available = {m.model for m in ollama.list().models}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(503, f"cannot reach Ollama to validate model name: {exc}") from None
        requested = updates.get("llm_model")
        if requested is not None and requested not in available:
            raise HTTPException(422, f"{requested!r} is not one of Ollama's available models: {sorted(available)}")

    changes = {
        field: {"from": getattr(settings, field), "to": value}
        for field, value in updates.items()
    }
    new_settings = settings.model_copy(update=updates)
    snapshot = {f: getattr(new_settings, f) for f in CONFIGURABLE_FIELDS}

    with _session(app) as session:
        session.add(ConfigHistory(changes=changes, snapshot=snapshot))
        session.commit()

    # Per PLAN.md's Phase 7 note: config is versioned per job (each job
    # records the config it actually ran under, in worker.py), not rejected
    # while jobs are in flight -- an in-flight job already has its own
    # per-job Settings copy (see job_runner.job_settings), so mutating the
    # server's live settings here can never change a run already underway.
    app.state.settings = new_settings
    app.state.worker.settings = new_settings
    logger.info("config_changed", **{k: v["to"] for k, v in changes.items()})
    return ConfigResponse(**snapshot)


@app.get("/v1/models", response_model=list[ModelInfo])
def list_models() -> list[ModelInfo]:
    try:
        import ollama

        response = ollama.list()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"cannot reach Ollama: {exc}") from None
    return [ModelInfo(name=m.model, size_bytes=m.size) for m in response.models]


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------

@app.post("/v1/benchmark", response_model=BenchmarkRunOut, status_code=202)
def create_benchmark(request: Annotated[BenchmarkCreateRequest, Body()]) -> BenchmarkRunOut:
    with _session(app) as session:
        existing = session.get(BenchmarkRun, request.tag)
        if existing is not None:
            raise HTTPException(409, f"a benchmark run tagged {request.tag!r} already exists")
        run = BenchmarkRun(id=request.tag, dataset=request.dataset, tier=request.tier, status="running")
        session.add(run)
        session.commit()
        session.refresh(run)
        out = BenchmarkRunOut.model_validate(run, from_attributes=True)

    app.state.worker.submit_benchmark(
        request.tag, dataset=request.dataset, tier=request.tier,
        n=request.n, seed=request.seed, skip_refine=request.skip_refine,
        semdist=request.semdist,
    )
    logger.info("benchmark_submitted", tag=request.tag, dataset=request.dataset, tier=request.tier)
    return out


@app.get("/v1/benchmark/{tag}", response_model=BenchmarkRunOut)
def get_benchmark(tag: str) -> BenchmarkRunOut:
    with _session(app) as session:
        run = session.get(BenchmarkRun, tag)
        if run is None:
            raise HTTPException(404, f"no such benchmark run: {tag}")
        return BenchmarkRunOut.model_validate(run, from_attributes=True)


@app.get("/v1/benchmark/{tag}/results")
def get_benchmark_results(tag: str) -> dict:
    import json

    with _session(app) as session:
        run = session.get(BenchmarkRun, tag)
        if run is None:
            raise HTTPException(404, f"no such benchmark run: {tag}")
        if run.status != "done" or not run.results_path:
            raise HTTPException(409, f"benchmark run {tag!r} is not finished yet (status={run.status})")
        path = PROJECT_ROOT / run.results_path
        return json.loads(path.read_text(encoding="utf-8"))

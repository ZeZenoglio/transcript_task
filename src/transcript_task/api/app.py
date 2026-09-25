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
from typing import Annotated, Any, Literal, cast

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from sqlmodel import Session, select

from ..pipeline import new_transcript_id
from ..prompts import get_refine_template, get_summarize_template
from ..settings import PROJECT_ROOT, Settings
from .db import (
    TERMINAL_STATUSES,
    BenchmarkRun,
    BenchmarkStatus,
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
    Problem,
    StageTimingOut,
)
from .worker import JobWorker, purge_job

logger = get_logger("transcript_task.api")

DESCRIPTION = """
A local speech-to-text service: submit an audio recording, get back a corrected,
punctuated transcript and a reviewed Word document. ASR runs on Whisper (MLX);
cleanup and summarisation run on a local Ollama model. **Nothing leaves this
machine** — no cloud calls, no telemetry, no third-party API keys.

## Quickstart
1. `POST /v1/jobs` with an audio file (multipart) → `202` and a `job_id`.
2. Poll `GET /v1/jobs/{job_id}` until `status` is `done` (or `failed`).
3. `GET /v1/jobs/{job_id}/result` for the transcript JSON, or
   `GET /v1/jobs/{job_id}/docx` for the generated Word document.

## Why the job endpoints are async
Transcription is minutes long, so `POST /v1/jobs` never blocks on it — it
returns immediately and the work runs in the background, **one job at a time
by default** (`Settings.api_concurrency`). That cap is deliberate, not a
bug: running Whisper and a local LLM at once already uses most of what a
16 GB machine has to give.

## Errors
Every non-2xx response is [RFC 7807](https://www.rfc-editor.org/rfc/rfc7807)
`application/problem+json` — see the `Problem` schema.
"""

TAGS_METADATA = [
    {
        "name": "health",
        "description": "Liveness and dependency reachability (ffmpeg/ffprobe/Ollama).",
    },
    {
        "name": "jobs",
        "description": "Submit a recording for transcription and poll the async result.",
    },
    {
        "name": "config",
        "description": "Read or live-patch the models/prompts/options new jobs run with.",
    },
    {
        "name": "models",
        "description": "What's actually available in the local Ollama instance right now.",
    },
    {
        "name": "benchmark",
        "description": "Score the current model/prompt configuration against public "
        "speech datasets (word/character error rate, semantic distance) "
        "and fetch the results.",
    },
]

_STATUS_TITLES = {
    400: "Bad Request",
    404: "Not Found",
    409: "Conflict",
    413: "Payload Too Large",
    422: "Unprocessable Entity",
    503: "Service Unavailable",
}


def problem_response(
    status_code: int, description: str, detail: str, instance: str = ""
) -> dict[str, Any]:
    """One documented error response, with an example specific to *this*
    status code -- not FastAPI's default of one example on the shared
    `Problem` schema, which Swagger then applies to every response
    referencing it regardless of what actually happened (found in a real
    visual QA pass: a 400 "bad extension" response showed a 404 "no such
    job" example -- see docs/visual-qa-report). This also fixes the
    declared media type: without an explicit `content` override here,
    FastAPI defaults every `{"model": Problem}` response to
    `application/json` even though the server actually sends
    `application/problem+json` (see the exception handlers above). The
    schema `$ref` is set explicitly here too, alongside the example --
    `custom_openapi()` above strips the auto-injected `application/json`
    entry that would otherwise carry it, so without this line Swagger would
    show an example but no formal schema for these responses."""
    return {
        "model": Problem,
        "description": description,
        "content": {
            "application/problem+json": {
                "schema": {"$ref": "#/components/schemas/Problem"},
                "example": {
                    "type": "about:blank",
                    "title": _STATUS_TITLES.get(status_code, "Error"),
                    "status": status_code,
                    "detail": detail,
                    "instance": instance,
                },
            }
        },
    }


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
    logger.info(
        "api_startup",
        asr_model=settings.asr_model,
        llm_model=settings.llm_model,
        concurrency=settings.api_concurrency,
    )
    yield
    app.state.worker.shutdown()
    logger.info("api_shutdown")


app = FastAPI(
    title="transcript_task API",
    description=DESCRIPTION,
    version="0.1.0",
    openapi_tags=TAGS_METADATA,
    lifespan=lifespan,
)
app.add_middleware(RequestIDMiddleware)


def custom_openapi() -> dict:
    """FastAPI's additional-response handling (`responses={...}`) always
    injects an `application/json` entry using the given `model`'s schema,
    *in addition to* any explicit `content` override in the same dict (see
    `fastapi/openapi/utils.py`'s handling of `route.responses`) -- there is
    no supported way to suppress this per-route while still getting the
    `Problem` schema correctly registered and referenced. Every response
    that carries a `Problem` body is only ever actually sent as
    `application/problem+json` (see the exception handlers above), so a
    real visual QA pass caught this as a genuine bug: the docs claimed two
    possible content types where only one is real (see docs/visual-qa-report).
    Post-processing the generated schema once, here, fixes it for every
    route uniformly instead of a per-route workaround repeated ~10 times.
    """
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
    )
    for path_item in schema.get("paths", {}).values():
        for operation in path_item.values():
            for response in operation.get("responses", {}).values():
                content = response.get("content", {})
                if "application/problem+json" in content and "application/json" in content:
                    del content["application/json"]
    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi  # type: ignore[method-assign]  # FastAPI's own documented pattern


def _session(app_: FastAPI) -> Session:
    return get_session(app_.state.settings)


def _problem(status_code: int, detail: str, instance: str) -> JSONResponse:
    problem = Problem(
        title=_STATUS_TITLES.get(status_code, "Error"),
        status=status_code,
        detail=detail,
        instance=instance,
    )
    return JSONResponse(
        status_code=status_code,
        content=problem.model_dump(exclude_none=True),
        media_type="application/problem+json",
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Every HTTPException raised anywhere in this app (see the routes
    below) is rendered as RFC 7807 rather than FastAPI's default
    `{"detail": ...}` -- see the Problem schema and DESCRIPTION above."""
    return _problem(exc.status_code, str(exc.detail), str(request.url.path))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    messages = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
    return _problem(422, messages, str(request.url.path))


NOT_FOUND: dict[int | str, dict[str, Any]] = {
    404: problem_response(
        404, "No resource with that id.", "no such job: deadbeef", "/v1/jobs/deadbeef"
    )
}


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Liveness and dependency check",
    description="Always returns `200` (never an error) -- check the `status` field, not the "
    "HTTP status code, to tell 'ok' from 'degraded'. Useful before submitting a job: "
    "if ffmpeg/ffprobe or Ollama are unreachable, every job will fail at that stage.",
    response_description="Current reachability of every dependency this service needs.",
)
def health() -> HealthResponse:
    settings = app.state.settings
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    ffprobe_ok = shutil.which("ffprobe") is not None

    ollama_reachable = False
    ollama_models: list[str] = []
    try:
        import ollama

        response = ollama.list()
        ollama_models = [m.model for m in response.models if m.model]
        ollama_reachable = True
    except Exception:  # noqa: BLE001 - health check must never 500
        pass

    status: Literal["ok", "degraded"]
    status = "ok" if (ffmpeg_ok and ffprobe_ok and ollama_reachable) else "degraded"
    return HealthResponse(
        status=status,
        ffmpeg=ffmpeg_ok,
        ffprobe=ffprobe_ok,
        ollama_reachable=ollama_reachable,
        ollama_models=ollama_models,
        asr_model=settings.asr_model,
        llm_model=settings.llm_model,
    )


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------


def _new_job_dir(settings: Settings, job_id: str) -> Path:
    js = job_settings(settings, job_id)
    js.extract_dir.mkdir(parents=True, exist_ok=True)
    return js.extract_dir


@app.post(
    "/v1/jobs",
    response_model=JobCreateResponse,
    status_code=202,
    tags=["jobs"],
    summary="Submit a recording for transcription",
    description="Accepts one audio file and returns immediately with a `job_id` -- "
    "transcription runs in the background (see the top-level description's "
    "'Why the job endpoints are async'). Allowed extensions and the upload size "
    "cap are both visible at `GET /v1/config` "
    "(`audio_extensions`/`api_max_upload_mb`), since they're config, not a fixed "
    "constant this description could drift out of sync with.\n\n"
    "`refine=true` (the default) runs the LLM cleanup pass and the result will "
    "carry **both** `raw_transcript` and `refined_transcript`; `refine=false` "
    "skips it entirely (faster, no LLM cost, `refined_transcript` stays null).",
    response_description="The new job's id and initial status (always 'queued').",
    responses={
        400: problem_response(
            400,
            "Unsupported audio file extension.",
            "unsupported audio extension '.txt'",
            "/v1/jobs",
        ),
        413: problem_response(
            413,
            "Upload exceeds the configured size limit.",
            "upload exceeds 500 MB limit",
            "/v1/jobs",
        ),
    },
)
async def create_job(
    file: Annotated[
        UploadFile,
        File(description="An audio recording. See GET /v1/config for accepted extensions."),
    ],
    refine: Annotated[
        bool, Form(description="Run the LLM cleanup pass. See the endpoint description.")
    ] = True,
) -> JobCreateResponse:
    settings: Settings = app.state.settings
    # UploadFile.filename is Optional at the type level (a multipart part
    # technically doesn't have to carry one); every real client sends one for
    # a file field, but the fallback keeps this from being a latent None-path
    # crash on `dest_dir / filename` below rather than a documented 400.
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in settings.audio_extensions:
        raise HTTPException(400, f"unsupported audio extension {suffix!r}")

    job_id = new_transcript_id()
    max_bytes = settings.api_max_upload_mb * 1024 * 1024
    dest_dir = _new_job_dir(settings, job_id)
    dest = dest_dir / filename
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
            id=job_id,
            filename=filename,
            status=JobStatus.queued,
            refine_requested=refine,
            asr_model=settings.asr_model,
            llm_model=settings.llm_model,
            refine_prompt_id=settings.refine_prompt_id,
            summarize_prompt_id=get_summarize_template(settings.summary_language).id,
        )
        session.add(job)
        session.commit()

    logger.info("job_submitted", job_id=job_id, filename=filename, refine=refine, bytes=written)
    app.state.worker.submit(job_id, dest, filename, refine)
    return JobCreateResponse(job_id=job_id, status=JobStatus.queued)


@app.get(
    "/v1/jobs",
    response_model=list[JobSummary],
    tags=["jobs"],
    summary="List jobs, most recent first",
    description="No filtering by status yet -- fetch the list and filter client-side, or "
    "poll a specific `GET /v1/jobs/{job_id}` if you already know its id.",
)
def list_jobs(
    limit: Annotated[int, Query(gt=0, le=500, description="Max jobs to return.")] = 50,
) -> list[JobSummary]:
    with _session(app) as session:
        # SQLModel's class-level Job.created_at is really a SQLAlchemy
        # InstrumentedAttribute at runtime (hence .desc() working), but mypy
        # sees the instance-level `datetime` annotation instead -- no plugin
        # configured for the gap, so this is the one place it needs telling.
        jobs = session.exec(select(Job).order_by(Job.created_at.desc()).limit(limit)).all()  # type: ignore[attr-defined]
        return [JobSummary.model_validate(j, from_attributes=True) for j in jobs]


def _get_job_or_404(session: Session, job_id: str) -> Job:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, f"no such job: {job_id}")
    return job


@app.get(
    "/v1/jobs/{job_id}",
    response_model=JobSummary,
    tags=["jobs"],
    summary="Job status",
    responses=NOT_FOUND,
    description="Poll this until `status` is a terminal value (`done`, `failed`, `canceled`) -- "
    "the intermediate values (`normalizing`, `transcribing`, `refining`, "
    "`summarizing`, `writing_docx`) show which stage is currently running.",
)
def get_job(job_id: str) -> JobSummary:
    with _session(app) as session:
        job = _get_job_or_404(session, job_id)
        return JobSummary.model_validate(job, from_attributes=True)


@app.get(
    "/v1/jobs/{job_id}/timings",
    response_model=list[StageTimingOut],
    tags=["jobs"],
    summary="Per-stage timing",
    responses=NOT_FOUND,
    description="Seconds spent in each completed stage. Empty until stages finish; a stage "
    "that's still running or was skipped (e.g. refine when `refine=false`) has no entry.",
)
def get_job_timings(job_id: str) -> list[StageTimingOut]:
    with _session(app) as session:
        _get_job_or_404(session, job_id)
        timings = session.exec(select(StageTiming).where(StageTiming.job_id == job_id)).all()
        # StageTiming.stage is a plain str column (SQLModel doesn't carry a
        # Literal through to the DB); only pipeline.py's own stage_* functions
        # ever write it, always one of these three names.
        return [
            StageTimingOut(
                stage=cast('Literal["transcribe", "refine", "summarize"]', t.stage),
                seconds=t.seconds,
            )
            for t in timings
        ]


@app.get(
    "/v1/jobs/{job_id}/result",
    response_model=JobResult,
    tags=["jobs"],
    summary="Transcript result",
    responses=NOT_FOUND,
    description="Returns the job's current fields regardless of status -- most are still "
    "null until the relevant stage completes. Check `status` (or poll "
    "`GET /v1/jobs/{job_id}` first) to know whether this is a finished result or "
    "a still-in-progress one.",
)
def get_job_result(job_id: str) -> JobResult:
    with _session(app) as session:
        job = _get_job_or_404(session, job_id)
        return JobResult.model_validate(
            {**job.model_dump(), "docx_available": bool(job.docx_path)},
            from_attributes=True,
        )


@app.get(
    "/v1/jobs/{job_id}/docx",
    tags=["jobs"],
    # response_model=None and response_class=FileResponse: without these,
    # FastAPI's OpenAPI generator derives the 200 response's content-type
    # from the *app's default response class* (JSONResponse), not from what
    # this route actually returns -- adding a stray `application/json`
    # entry (empty schema) alongside the real docx content-type below.
    # Swagger then defaults its dropdown to that one and shows a fabricated
    # `"string"` example, making the docs claim this endpoint returns JSON
    # (a real bug caught in a visual QA pass, not a runtime issue -- the
    # actual response was always a correct .docx file; see
    # docs/visual-qa-report). FileResponse.media_type is None by default,
    # which is what actually suppresses FastAPI's automatic content-type
    # injection (confirmed against fastapi/openapi/utils.py's
    # get_openapi_path: it only adds an entry when
    # `current_response_class.media_type` is truthy) -- response_model=None
    # alone does not fix this, since that logic keys off response_class,
    # not response_model.
    response_model=None,
    response_class=FileResponse,
    summary="Download the generated Word document",
    description="Only available once `status` is `done` (or, if refine was requested and "
    "rejected by the quality guard, still generated -- see `refine_rejected` in "
    "the result). The document embeds the raw transcript as an appendix "
    "regardless of whether refine ran.",
    response_description="The .docx file.",
    responses={
        404: problem_response(
            404,
            "Job not found, not finished, or the document record "
            "exists but the file is missing on disk.",
            "no document available for this job (not done, or refine/summarize failed)",
            "/v1/jobs/deadbeef/docx",
        ),
        200: {
            "content": {
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": {}
            }
        },
    },
)
def get_job_docx(job_id: str) -> FileResponse:
    with _session(app) as session:
        job = _get_job_or_404(session, job_id)
    if not job.docx_path:
        raise HTTPException(
            404, "no document available for this job (not done, or refine/summarize failed)"
        )
    path = PROJECT_ROOT / job.docx_path
    if not path.exists():
        raise HTTPException(404, "document record exists but the file is missing on disk")
    return FileResponse(
        path,
        filename=path.name,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


@app.delete(
    "/v1/jobs/{job_id}",
    status_code=204,
    tags=["jobs"],
    summary="Cancel or purge a job",
    description="A still-queued job is canceled outright. A **currently running** job "
    "cannot be interrupted -- Python threads running the ASR/LLM calls aren't "
    "preemptible and have no cancellation hook -- so this returns `409` rather "
    "than pretending to succeed; retry once the job reaches a terminal status. "
    "A finished job (done/failed/canceled) is purged immediately: its DB row, "
    "stage timings, and on-disk files (including any .docx) are all removed.",
    responses={
        404: problem_response(404, "No such job.", "no such job: deadbeef", "/v1/jobs/deadbeef"),
        409: problem_response(
            409,
            "Job is currently running and cannot be interrupted.",
            "job is currently running and cannot be interrupted -- "
            "retry once it reaches a terminal status",
            "/v1/jobs/a1b2c3d4",
        ),
    },
)
def delete_job(job_id: str) -> None:
    worker: JobWorker = app.state.worker
    with _session(app) as session:
        job = _get_job_or_404(session, job_id)
        status = job.status

    if status not in TERMINAL_STATUSES:
        canceled = worker.cancel(job_id)
        if not canceled:
            raise HTTPException(
                409,
                "job is currently running and cannot be interrupted -- "
                "retry once it reaches a terminal status",
            )
        with _session(app) as session:
            fresh = session.get(Job, job_id)
            if fresh is not None:
                fresh.status = JobStatus.canceled
                session.add(fresh)
                session.commit()

    purge_job(app.state.settings, job_id)
    logger.info("job_deleted", job_id=job_id)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def _config_response(settings: Settings) -> ConfigResponse:
    return ConfigResponse(
        **{f: getattr(settings, f) for f in CONFIGURABLE_FIELDS},
        api_max_upload_mb=settings.api_max_upload_mb,
        audio_extensions=sorted(settings.audio_extensions),
    )


@app.get(
    "/v1/config",
    response_model=ConfigResponse,
    tags=["config"],
    summary="Current effective configuration",
    description="What every *new* job will run under right now. A job already running keeps "
    "whatever config was in effect when its own stages started (see PATCH below).",
)
def get_config() -> ConfigResponse:
    return _config_response(app.state.settings)


@app.patch(
    "/v1/config",
    response_model=ConfigResponse,
    tags=["config"],
    summary="Live-patch the configuration",
    description="Only the fields you send are changed (PATCH semantics) -- omitted fields "
    "keep their current value. `llm_model`/`asr_model` are checked against "
    "`ollama list` and `refine_prompt_id` against the known prompt ids before "
    "being accepted. **Does not affect jobs already running**: each job runs "
    "under its own copy of the config taken when it started, so a change here is "
    "never retroactive and never needs to wait for in-flight jobs to finish. Every "
    "accepted change is recorded (append-only) and survives a server restart.",
    responses={
        422: problem_response(
            422,
            "Unknown refine_prompt_id, or llm_model not found in `ollama list`.",
            "'qwen3.5:99b' is not one of Ollama's available models: ['qwen3.5:4b', 'qwen3.5:9b']",
            "/v1/config",
        ),
        503: problem_response(
            503,
            "Ollama unreachable, needed to validate a model-name change.",
            "cannot reach Ollama to validate model name: Connection refused",
            "/v1/config",
        ),
    },
)
def patch_config(patch: ConfigPatch) -> ConfigResponse:
    settings: Settings = app.state.settings
    updates = patch.model_dump(exclude_unset=True)
    if not updates:
        return _config_response(settings)

    if "refine_prompt_id" in updates:
        try:
            get_refine_template(updates["refine_prompt_id"])
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    if "llm_model" in updates or "asr_model" in updates:
        try:
            import ollama

            available = {m.model for m in ollama.list().models if m.model}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(503, f"cannot reach Ollama to validate model name: {exc}") from None
        requested = updates.get("llm_model")
        if requested is not None and requested not in available:
            raise HTTPException(
                422, f"{requested!r} is not one of Ollama's available models: {sorted(available)}"
            )

    changes = {
        field: {"from": getattr(settings, field), "to": value} for field, value in updates.items()
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
    return _config_response(new_settings)


@app.get(
    "/v1/models",
    response_model=list[ModelInfo],
    tags=["models"],
    summary="Available Ollama models",
    description="A live `ollama list` call, not a cached/configured value -- reflects "
    "whatever's actually pulled on this machine right now. Useful before "
    "`PATCH /v1/config` to check a model name is valid.",
    responses={
        503: problem_response(
            503, "Ollama unreachable.", "cannot reach Ollama: Connection refused", "/v1/models"
        )
    },
)
def list_models() -> list[ModelInfo]:
    try:
        import ollama

        response = ollama.list()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"cannot reach Ollama: {exc}") from None
    return [ModelInfo(name=m.model, size_bytes=m.size) for m in response.models if m.model]


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


@app.post(
    "/v1/benchmark",
    response_model=BenchmarkRunOut,
    status_code=202,
    tags=["benchmark"],
    summary="Run a benchmark tier",
    description="Scores the current model/prompt configuration against FLEURS or Common Voice "
    "ground truth (word/character error rate, semantic distance, and the refine "
    "stage's before/after delta against that same ground truth). Runs in the "
    "background on the same thread pool as real jobs, so it shares the same "
    "concurrency cap and never runs at the same time as one. `tier='smoke'` uses "
    "4 small clips committed to the repo (no download needed); `'quick'`/`'full'` "
    "need the corresponding dataset fetched on the server first.",
    responses={
        409: problem_response(
            409,
            "A run with this `tag` already exists.",
            "a benchmark run tagged 'qwen9b-baseline' already exists",
            "/v1/benchmark",
        ),
    },
)
def create_benchmark(request: Annotated[BenchmarkCreateRequest, Body()]) -> BenchmarkRunOut:
    with _session(app) as session:
        existing = session.get(BenchmarkRun, request.tag)
        if existing is not None:
            raise HTTPException(409, f"a benchmark run tagged {request.tag!r} already exists")
        run = BenchmarkRun(
            id=request.tag,
            dataset=request.dataset,
            tier=request.tier,
            status=BenchmarkStatus.running,
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        out = BenchmarkRunOut.model_validate(run, from_attributes=True)

    app.state.worker.submit_benchmark(
        request.tag,
        dataset=request.dataset,
        tier=request.tier,
        n=request.n,
        seed=request.seed,
        skip_refine=request.skip_refine,
        semdist=request.semdist,
    )
    logger.info("benchmark_submitted", tag=request.tag, dataset=request.dataset, tier=request.tier)
    return out


@app.get(
    "/v1/benchmark/{tag}",
    response_model=BenchmarkRunOut,
    tags=["benchmark"],
    summary="Benchmark run status",
    responses=NOT_FOUND,
    description="Poll until `status` is `done` or `failed`, then fetch the full numbers from "
    "`GET /v1/benchmark/{tag}/results`.",
)
def get_benchmark(tag: str) -> BenchmarkRunOut:
    with _session(app) as session:
        run = session.get(BenchmarkRun, tag)
        if run is None:
            raise HTTPException(404, f"no such benchmark run: {tag}")
        return BenchmarkRunOut.model_validate(run, from_attributes=True)


@app.get(
    "/v1/benchmark/{tag}/results",
    tags=["benchmark"],
    summary="Full benchmark results",
    description="The complete `BenchmarkResult` JSON (per-clip WER/CER/SemDist, aggregates, "
    "settings snapshot) -- the same object `scripts/benchmark.py` writes to "
    "`benchmarks/<tag>/results.json`.",
    responses={
        404: problem_response(
            404,
            "No such benchmark run.",
            "no such benchmark run: qwen9b-baseline",
            "/v1/benchmark/qwen9b-baseline/results",
        ),
        409: problem_response(
            409,
            "Run exists but hasn't finished yet.",
            "benchmark run 'qwen9b-baseline' is not finished yet (status=running)",
            "/v1/benchmark/qwen9b-baseline/results",
        ),
    },
)
def get_benchmark_results(tag: str) -> dict:
    import json

    with _session(app) as session:
        run = session.get(BenchmarkRun, tag)
        if run is None:
            raise HTTPException(404, f"no such benchmark run: {tag}")
        if run.status != BenchmarkStatus.done or not run.results_path:
            raise HTTPException(
                409, f"benchmark run {tag!r} is not finished yet (status={run.status})"
            )
        path = PROJECT_ROOT / run.results_path
        return json.loads(path.read_text(encoding="utf-8"))

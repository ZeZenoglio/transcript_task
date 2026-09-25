"""structlog setup (Phase 7): JSON to a file, human-readable to the
console, both from the same log calls -- structlog's standard "log once,
render twice" pattern (`ProcessorFormatter` wrapping two different
renderers on two different stdlib handlers).

Request-ID propagation: `RequestIDMiddleware` binds a per-request id into
structlog's contextvars, so every log line emitted while handling that
request -- including from deep inside a stage function -- carries it
automatically, with no need to thread a request id through every call.

**Never log transcript content at INFO.** Every log call in this module and
`worker.py`/`app.py` logs metadata only (job id, stage name, duration,
counts) -- never `raw_transcript`/`refined_transcript`/summary text. This
mirrors `pipeline.py`'s own `log()` calls, which only ever log lengths and
timings, not content.
"""

from __future__ import annotations

import logging
import logging.config
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from ..settings import Settings


def configure_logging(settings: Settings) -> None:
    log_dir = settings.api_jobs_dir.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "api.jsonl"

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processor": structlog.processors.JSONRenderer(),
                },
                "console": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processor": structlog.dev.ConsoleRenderer(),
                },
            },
            "handlers": {
                "file": {
                    "class": "logging.FileHandler",
                    "filename": str(log_path),
                    "formatter": "json",
                },
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "console",
                },
            },
            "root": {"handlers": ["file", "console"], "level": "INFO"},
        }
    )


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Reads `X-Request-ID` if the caller supplied one, generates one
    otherwise; binds it into structlog's contextvars for the duration of
    the request and echoes it back in the response header."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()
        response.headers["X-Request-ID"] = request_id
        return response


def get_logger(name: str):
    return structlog.get_logger(name)

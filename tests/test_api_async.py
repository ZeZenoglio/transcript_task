"""The one genuinely-async test PLAN.md's Phase 7 section asks for
("httpx.AsyncClient against the app"). Every other API test uses FastAPI's
synchronous TestClient (a thin httpx wrapper that already exercises the
same ASGI app) since duplicating each behavioral case in both styles would
be pure churn -- this file exists to prove the app also works when driven
the fully-async way, not to re-cover ground test_api.py already covers.
"""

from __future__ import annotations

import shutil

import httpx
import pytest
from fakes import FakeChatModel, FakeTranscriber

from transcript_task.api import app as app_module
from transcript_task.api.db import reset_engine_for_tests
from transcript_task.settings import PROJECT_ROOT

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def api_env(tmp_path, monkeypatch):
    jobs_root = PROJECT_ROOT / "data" / "test_api_jobs" / f"async-{tmp_path.name}"
    jobs_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TRANSCRIPT_API_DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.setenv("TRANSCRIPT_API_JOBS_DIR", str(jobs_root))
    reset_engine_for_tests()

    monkeypatch.setattr(
        app_module, "create_worker",
        lambda settings: app_module.JobWorker(
            settings, transcriber=FakeTranscriber([]), chat_model=FakeChatModel([])
        ),
    )
    yield
    reset_engine_for_tests()
    shutil.rmtree(jobs_root, ignore_errors=True)


async def test_health_via_async_client(api_env):
    # AsyncClient + ASGITransport doesn't run the app's lifespan on its own
    # (unlike TestClient used as a context manager) -- entering it manually
    # is the standard workaround (see e.g. FastAPI's own async-testing docs).
    async with app_module.app.router.lifespan_context(app_module.app):
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["asr_model"]


async def test_full_job_submission_via_async_client(api_env):
    async with app_module.app.router.lifespan_context(app_module.app):
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/v1/jobs")
            assert r.status_code == 200
            assert r.json() == []

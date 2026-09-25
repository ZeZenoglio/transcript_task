"""Real-model integration test for the FastAPI service (Phase 7/10): no
fakes anywhere. Split out of tests/e2e/test_api.py (which covers the same
API surface with fakes for ASR/LLM) so a plain `pytest` run never needs
mlx-whisper/Ollama -- opt in with `pytest -m integration`.
"""

from __future__ import annotations

import shutil

import pytest
from fastapi.testclient import TestClient

from transcript_task.api import app as app_module
from transcript_task.api.db import reset_engine_for_tests
from transcript_task.settings import PROJECT_ROOT


@pytest.mark.integration
def test_real_job_end_to_end(tmp_path, monkeypatch):
    """A real committed fixture file goes in through the actual HTTP upload
    endpoint, gets transcribed by real mlx-whisper and refined/summarized by
    real Ollama, and comes back out as a downloadable real .docx -- the same
    real-data-first discipline as every other integration test in this
    project, just driven through the API instead of calling the pipeline
    functions directly."""
    try:
        import ollama

        ollama.list()
    except Exception:
        pytest.skip("Ollama is not reachable on this machine")

    jobs_root = PROJECT_ROOT / "data" / "test_api_jobs" / f"real-{tmp_path.name}"
    jobs_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TRANSCRIPT_API_DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.setenv("TRANSCRIPT_API_JOBS_DIR", str(jobs_root))
    reset_engine_for_tests()

    fixture = PROJECT_ROOT / "tests" / "fixtures" / "audio" / "fleurs_row00871.wav"
    ground_truth = "Pense na rota de esqui como uma rota de caminhada."

    try:
        with TestClient(app_module.app) as client:
            with open(fixture, "rb") as f:
                created = client.post(
                    "/v1/jobs",
                    files={"file": ("fleurs_row00871.wav", f, "audio/wav")},
                    data={"refine": "true"},
                )
            assert created.status_code == 202
            job_id = created.json()["job_id"]

            client.app.state.worker.wait_for(job_id, timeout=180)

            result = client.get(f"/v1/jobs/{job_id}/result").json()
            assert result["status"] == "done", result.get("error")
            assert result["raw_transcript"]
            assert result["refined_transcript"]
            # Close to ground truth, not exact -- same tolerance the other
            # real-model integration tests in this project use.
            from transcript_task.text_compare import normalize_pt

            assert normalize_pt(ground_truth) in normalize_pt(
                result["raw_transcript"]
            ) or normalize_pt(result["raw_transcript"]) == normalize_pt(ground_truth)

            docx_response = client.get(f"/v1/jobs/{job_id}/docx")
            assert docx_response.status_code == 200
            assert len(docx_response.content) > 0
    finally:
        reset_engine_for_tests()
        shutil.rmtree(jobs_root, ignore_errors=True)

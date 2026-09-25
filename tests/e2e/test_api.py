"""Tests for the FastAPI service (Phase 7).

Uses FastAPI's TestClient (a thin httpx wrapper; the plan's Phase 7 section
asks for "httpx.AsyncClient against the app" -- see test_api_async.py for
the one genuinely-async test that exercises, since sync TestClient already
covers every behavioral case here and duplicating each test in both styles
would be pure churn) against a real SQLite file per test and fakes for
ASR/LLM, so the whole submit -> background thread -> DB update -> poll ->
result path runs for real without touching mlx-whisper/Ollama.
"""

from __future__ import annotations

import io
import shutil
import threading
import time
import wave

import pytest
from fakes import FakeChatModel, FakeTranscriber
from fastapi.testclient import TestClient

from transcript_task.api import app as app_module
from transcript_task.api.db import reset_engine_for_tests
from transcript_task.settings import PROJECT_ROOT

VALID_SUMMARY_JSON = (
    '{"title": "Titulo", "description": "Uma descricao curta.", '
    '"topics": ["a", "b", "c"], "speakers_detected": 1, '
    '"language_variant": "pt-PT", "sensitivity": "low", "confidence": "high"}'
)


def _tiny_wav_bytes() -> bytes:
    """A real, valid 16kHz mono 16-bit WAV (a fraction of a second of
    silence) -- normalize.stage_normalize runs real ffprobe/ffmpeg on
    whatever gets uploaded (only ASR/LLM are faked), so this has to be
    genuinely valid audio, not placeholder bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)  # 0.1s of silence
    return buf.getvalue()


@pytest.fixture
def api(tmp_path, monkeypatch):
    """A fresh temp DB/jobs-dir per test, and a `client_with(transcriber,
    chat_model)` factory that swaps in fakes for the worker's ASR/LLM before
    the app's lifespan starts -- monkeypatch.setattr so it's undone
    automatically even if a test raises.

    `api_jobs_dir` is created *inside* the repo (under `data/`, gitignored)
    rather than using pytest's own `tmp_path` -- pipeline.py's stage
    functions store paths via `.relative_to(PROJECT_ROOT)` (used elsewhere
    to keep transcripts.json/docx paths portable), so job directories genuinely
    can't live outside the repo tree, in tests or production alike.
    """
    jobs_root = PROJECT_ROOT / "data" / "test_api_jobs" / tmp_path.name
    jobs_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TRANSCRIPT_API_DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.setenv("TRANSCRIPT_API_JOBS_DIR", str(jobs_root))
    reset_engine_for_tests()

    def client_with(transcriber, chat_model) -> TestClient:
        def fake_create_worker(settings):
            return app_module.JobWorker(settings, transcriber=transcriber, chat_model=chat_model)

        monkeypatch.setattr(app_module, "create_worker", fake_create_worker)
        return TestClient(app_module.app)

    yield client_with
    reset_engine_for_tests()
    shutil.rmtree(jobs_root, ignore_errors=True)
    # Real benchmark runs (see TestBenchmark) write to benchmarks/<tag>/ at
    # the project root, same as the CLI does -- clean up this suite's tags.
    for tag_dir in (PROJECT_ROOT / "benchmarks").glob("apitest-*"):
        shutil.rmtree(tag_dir, ignore_errors=True)


def _upload(client: TestClient, filename="clip.wav", refine=True, content: bytes | None = None):
    audio_bytes = io.BytesIO(content if content is not None else _tiny_wav_bytes())
    return client.post(
        "/v1/jobs",
        files={"file": (filename, audio_bytes, "audio/wav")},
        data={"refine": str(refine).lower()},
    )


class TestHealth:
    def test_health_reports_ok_when_everything_reachable(self, api):
        with api(FakeTranscriber([]), FakeChatModel([])) as client:
            r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["ffmpeg"] is True
        assert body["ffprobe"] is True

    def test_health_degrades_when_ollama_unreachable(self, api, monkeypatch):
        with api(FakeTranscriber([]), FakeChatModel([])) as client:
            import ollama

            def boom():
                raise ConnectionError("no ollama here")

            monkeypatch.setattr(ollama, "list", boom)
            r = client.get("/health")
        assert r.json()["status"] == "degraded"
        assert r.json()["ollama_reachable"] is False


class TestCreateJobValidation:
    def test_rejects_unsupported_extension(self, api):
        with api(FakeTranscriber([]), FakeChatModel([])) as client:
            r = _upload(client, filename="clip.txt")
        assert r.status_code == 400

    def test_rejects_oversized_upload(self, api, monkeypatch):
        with api(FakeTranscriber([]), FakeChatModel([])) as client:
            monkeypatch.setattr(client.app.state.settings, "api_max_upload_mb", 0)
            # 0 MB cap: even the tiny real WAV above trips it.
            r = _upload(client)
        assert r.status_code == 413


class TestJobLifecycleHappyPath:
    def test_refine_true_returns_both_transcripts(self, api):
        transcriber = FakeTranscriber(["ola mundo"])
        chat_model = FakeChatModel(["Olá, mundo!", VALID_SUMMARY_JSON])
        with api(transcriber, chat_model) as client:
            created = _upload(client, refine=True)
            assert created.status_code == 202
            job_id = created.json()["job_id"]

            client.app.state.worker.wait_for(job_id)

            result = client.get(f"/v1/jobs/{job_id}/result").json()
            assert result["status"] == "done"
            assert result["raw_transcript"] == "ola mundo"
            assert result["refined_transcript"] == "Olá, mundo!"
            assert result["summary"]["title"] == "Titulo"
            assert result["docx_available"] is True

    def test_refine_false_returns_only_raw(self, api):
        transcriber = FakeTranscriber(["ola mundo"])
        chat_model = FakeChatModel([VALID_SUMMARY_JSON])  # no refine call expected
        with api(transcriber, chat_model) as client:
            created = _upload(client, refine=False)
            job_id = created.json()["job_id"]
            client.app.state.worker.wait_for(job_id)

            result = client.get(f"/v1/jobs/{job_id}/result").json()
            assert result["status"] == "done"
            assert result["raw_transcript"] == "ola mundo"
            assert result["refined_transcript"] is None

    def test_docx_download_after_completion(self, api):
        transcriber = FakeTranscriber(["ola mundo"])
        chat_model = FakeChatModel(["Olá, mundo!", VALID_SUMMARY_JSON])
        with api(transcriber, chat_model) as client:
            job_id = _upload(client).json()["job_id"]
            client.app.state.worker.wait_for(job_id)

            r = client.get(f"/v1/jobs/{job_id}/docx")
            assert r.status_code == 200
            assert r.headers["content-type"].startswith(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

    def test_docx_not_available_before_completion(self, api):
        with api(FakeTranscriber([]), FakeChatModel([])) as client:
            # A job whose ASR fake is empty will fail at transcribe, never
            # reaching docx -- deterministic without needing real timing.
            job_id = _upload(client).json()["job_id"]
            client.app.state.worker.wait_for(job_id)
            r = client.get(f"/v1/jobs/{job_id}/docx")
        assert r.status_code == 404

    def test_list_jobs_includes_submitted_job(self, api):
        transcriber = FakeTranscriber(["ola"])
        chat_model = FakeChatModel(["Ola.", VALID_SUMMARY_JSON])
        with api(transcriber, chat_model) as client:
            job_id = _upload(client).json()["job_id"]
            client.app.state.worker.wait_for(job_id)
            listing = client.get("/v1/jobs").json()
        assert any(j["job_id"] == job_id for j in listing)

    def test_stage_timings_recorded(self, api):
        transcriber = FakeTranscriber(["ola"])
        chat_model = FakeChatModel(["Ola.", VALID_SUMMARY_JSON])
        with api(transcriber, chat_model) as client:
            job_id = _upload(client).json()["job_id"]
            client.app.state.worker.wait_for(job_id)
            timings = client.get(f"/v1/jobs/{job_id}/timings").json()
        stages = {t["stage"] for t in timings}
        assert "transcribe" in stages
        assert "refine" in stages
        assert "summarize" in stages


class TestJobFailure:
    def test_asr_failure_marks_job_failed(self, api):
        transcriber = FakeTranscriber([], fail=True)
        with api(transcriber, FakeChatModel([])) as client:
            job_id = _upload(client).json()["job_id"]
            client.app.state.worker.wait_for(job_id)
            result = client.get(f"/v1/jobs/{job_id}/result").json()
        assert result["status"] == "failed"
        assert result["error"] is not None


class TestConfig:
    def test_get_config_reflects_current_settings(self, api):
        with api(FakeTranscriber([]), FakeChatModel([])) as client:
            r = client.get("/v1/config")
        assert r.status_code == 200
        assert r.json()["refine_prompt_id"] == "refine-pt-v2"

    def test_get_config_states_upload_constraints(self, api):
        """Phase 8: upload limits must be discoverable from the API itself,
        not just a description string that can drift out of date."""
        with api(FakeTranscriber([]), FakeChatModel([])) as client:
            body = client.get("/v1/config").json()
        assert body["api_max_upload_mb"] > 0
        assert ".wav" in body["audio_extensions"]

    def test_patch_rejects_unknown_refine_prompt_id(self, api):
        with api(FakeTranscriber([]), FakeChatModel([])) as client:
            r = client.patch("/v1/config", json={"refine_prompt_id": "bogus"})
        assert r.status_code == 422

    def test_patch_rejects_llm_model_not_available_in_ollama(self, api, monkeypatch):
        with api(FakeTranscriber([]), FakeChatModel([])) as client:
            import ollama

            class FakeModel:
                model = "qwen3.5:9b"

            class FakeList:
                models = [FakeModel()]

            monkeypatch.setattr(ollama, "list", lambda: FakeList())
            r = client.patch("/v1/config", json={"llm_model": "does-not-exist:1b"})
        assert r.status_code == 422

    def test_patch_accepts_and_persists_a_valid_change(self, api, monkeypatch):
        with api(FakeTranscriber([]), FakeChatModel([])) as client:
            import ollama

            class FakeModel:
                model = "qwen3.5:4b"

            class FakeList:
                models = [FakeModel()]

            monkeypatch.setattr(ollama, "list", lambda: FakeList())
            r = client.patch("/v1/config", json={"llm_model": "qwen3.5:4b"})
            assert r.status_code == 200
            assert r.json()["llm_model"] == "qwen3.5:4b"

            # Reflected immediately in GET, not just the PATCH response.
            assert client.get("/v1/config").json()["llm_model"] == "qwen3.5:4b"

    def test_patch_with_no_fields_is_a_noop(self, api):
        with api(FakeTranscriber([]), FakeChatModel([])) as client:
            before = client.get("/v1/config").json()
            r = client.patch("/v1/config", json={})
            assert r.status_code == 200
            assert r.json() == before

    def test_config_survives_server_restart(self, api, monkeypatch):
        """PLAN.md's Phase 7 note: config changes must persist restarts, not
        just live in server memory. Simulated here by tearing down and
        re-entering the app's lifespan against the same DB path."""
        import ollama

        class FakeModel:
            model = "qwen3.5:4b"

        class FakeList:
            models = [FakeModel()]

        monkeypatch.setattr(ollama, "list", lambda: FakeList())

        with api(FakeTranscriber([]), FakeChatModel([])) as client:
            client.patch("/v1/config", json={"llm_model": "qwen3.5:4b"})

        # A fresh TestClient re-runs the lifespan (a "restart") against the
        # same DB path (env vars set by the fixture are still in effect).
        with TestClient(app_module.app) as restarted:
            assert restarted.get("/v1/config").json()["llm_model"] == "qwen3.5:4b"


class TestModels:
    def test_lists_available_ollama_models(self, api, monkeypatch):
        with api(FakeTranscriber([]), FakeChatModel([])) as client:
            import ollama

            class FakeModel:
                model = "qwen3.5:9b"
                size = 123

            class FakeList:
                models = [FakeModel()]

            monkeypatch.setattr(ollama, "list", lambda: FakeList())
            r = client.get("/v1/models")
        assert r.status_code == 200
        assert r.json() == [{"name": "qwen3.5:9b", "size_bytes": 123}]

    def test_returns_503_when_ollama_unreachable(self, api, monkeypatch):
        with api(FakeTranscriber([]), FakeChatModel([])) as client:
            import ollama

            def boom():
                raise ConnectionError("down")

            monkeypatch.setattr(ollama, "list", boom)
            r = client.get("/v1/models")
        assert r.status_code == 503


class TestDeleteAndCancel:
    def test_delete_a_finished_job_purges_it(self, api):
        transcriber = FakeTranscriber(["ola"])
        chat_model = FakeChatModel(["Ola.", VALID_SUMMARY_JSON])
        with api(transcriber, chat_model) as client:
            job_id = _upload(client).json()["job_id"]
            client.app.state.worker.wait_for(job_id)

            r = client.delete(f"/v1/jobs/{job_id}")
            assert r.status_code == 204
            assert client.get(f"/v1/jobs/{job_id}").status_code == 404

    def test_cancel_a_still_queued_job(self, api):
        """With concurrency=1, a second submitted job sits queued behind the
        first -- cancellable before it ever starts. The first job's fake
        transcriber blocks on an Event so the test controls exactly when it
        finishes, making the second job's "queued" state deterministic
        instead of a race."""
        release = threading.Event()

        class BlockingTranscriber(FakeTranscriber):
            def transcribe(self, audio_path, *, language):
                release.wait(timeout=5)
                return super().transcribe(audio_path, language=language)

        transcriber = BlockingTranscriber(["first", "second"])
        chat_model = FakeChatModel(["First.", VALID_SUMMARY_JSON, "Second.", VALID_SUMMARY_JSON])
        with api(transcriber, chat_model) as client:
            first_id = _upload(client, filename="first.wav").json()["job_id"]
            second_id = _upload(client, filename="second.wav").json()["job_id"]

            # Give the executor a moment to actually start the first job's
            # thread (so the second is genuinely queued, not just "not yet
            # submitted") before attempting to cancel the second.
            for _ in range(50):
                if client.get(f"/v1/jobs/{first_id}").json()["status"] != "queued":
                    break
                time.sleep(0.05)

            r = client.delete(f"/v1/jobs/{second_id}")
            assert r.status_code == 204

            release.set()
            client.app.state.worker.wait_for(first_id)
            assert client.get(f"/v1/jobs/{second_id}").status_code == 404

    def test_cannot_delete_a_currently_running_job(self, api):
        release = threading.Event()

        class BlockingTranscriber(FakeTranscriber):
            def transcribe(self, audio_path, *, language):
                release.wait(timeout=5)
                return super().transcribe(audio_path, language=language)

        transcriber = BlockingTranscriber(["ola"])
        chat_model = FakeChatModel(["Ola.", VALID_SUMMARY_JSON])
        with api(transcriber, chat_model) as client:
            job_id = _upload(client).json()["job_id"]

            for _ in range(50):
                if client.get(f"/v1/jobs/{job_id}").json()["status"] == "transcribing":
                    break
                time.sleep(0.05)

            r = client.delete(f"/v1/jobs/{job_id}")
            assert r.status_code == 409

            release.set()
            client.app.state.worker.wait_for(job_id)


class TestConcurrencyCap:
    def test_two_jobs_never_run_at_the_same_time_with_default_cap(self, api):
        """api_concurrency defaults to 1 -- this is the actual guarantee
        that matters (not just "the setting exists"), verified by having
        two jobs both record whether the other was mid-flight when they ran."""
        lock = threading.Lock()
        concurrent_count = 0
        peak = 0

        class TrackingTranscriber(FakeTranscriber):
            def transcribe(self, audio_path, *, language):
                nonlocal concurrent_count, peak
                with lock:
                    concurrent_count += 1
                    peak = max(peak, concurrent_count)
                time.sleep(0.1)
                with lock:
                    concurrent_count -= 1
                return super().transcribe(audio_path, language=language)

        transcriber = TrackingTranscriber(["a", "b"])
        chat_model = FakeChatModel(["A.", VALID_SUMMARY_JSON, "B.", VALID_SUMMARY_JSON])
        with api(transcriber, chat_model) as client:
            first_id = _upload(client, filename="a.wav").json()["job_id"]
            second_id = _upload(client, filename="b.wav").json()["job_id"]
            client.app.state.worker.wait_for(first_id)
            client.app.state.worker.wait_for(second_id)

        assert peak == 1


class TestBenchmark:
    """Uses the same transcriber/chat_model injection as job tests (added to
    eval/orchestrator.run_benchmark_tier specifically for this) so these
    exercise the real submit -> background thread -> DB path without ever
    calling mlx-whisper/Ollama -- a real full-tier run is already covered
    by Phase 6's own eval-harness tests and the smoke-tier CLI check."""

    def _fakes_for_smoke_tier(self):
        # tests/fixtures/manifest.jsonl has 4 clips; each needs 1 transcribe
        # + 1 refine + 1 summarize call.
        transcriber = FakeTranscriber(["frase um", "frase dois", "frase tres", "frase quatro"])
        chat_model = FakeChatModel(
            [
                "Frase um.",
                VALID_SUMMARY_JSON,
                "Frase dois.",
                VALID_SUMMARY_JSON,
                "Frase tres.",
                VALID_SUMMARY_JSON,
                "Frase quatro.",
                VALID_SUMMARY_JSON,
            ]
        )
        return transcriber, chat_model

    def test_create_returns_202_and_running_status(self, api):
        transcriber, chat_model = self._fakes_for_smoke_tier()
        with api(transcriber, chat_model) as client:
            r = client.post(
                "/v1/benchmark", json={"tag": "apitest-test-run", "tier": "smoke", "semdist": False}
            )
            assert r.status_code == 202
            assert r.json()["status"] == "running"
            client.app.state.worker.wait_for_benchmark("apitest-test-run")

    def test_completed_run_is_fetchable_with_results(self, api):
        transcriber, chat_model = self._fakes_for_smoke_tier()
        with api(transcriber, chat_model) as client:
            client.post(
                "/v1/benchmark", json={"tag": "apitest-done-run", "tier": "smoke", "semdist": False}
            )
            client.app.state.worker.wait_for_benchmark("apitest-done-run")

            status = client.get("/v1/benchmark/apitest-done-run").json()
            assert status["status"] == "done"

            results = client.get("/v1/benchmark/apitest-done-run/results")
            assert results.status_code == 200
            assert results.json()["tag"] == "apitest-done-run"
            assert len(results.json()["clips"]) == 4

    def test_duplicate_tag_is_rejected(self, api):
        transcriber, chat_model = self._fakes_for_smoke_tier()
        with api(transcriber, chat_model) as client:
            client.post(
                "/v1/benchmark", json={"tag": "apitest-dup", "tier": "smoke", "semdist": False}
            )
            r = client.post(
                "/v1/benchmark", json={"tag": "apitest-dup", "tier": "smoke", "semdist": False}
            )
            assert r.status_code == 409
            client.app.state.worker.wait_for_benchmark("apitest-dup")

    def test_unknown_tag_is_404(self, api):
        with api(FakeTranscriber([]), FakeChatModel([])) as client:
            r = client.get("/v1/benchmark/never-existed")
        assert r.status_code == 404

    def test_results_not_ready_while_running(self, api):
        release = threading.Event()

        class BlockingTranscriber(FakeTranscriber):
            def transcribe(self, audio_path, *, language):
                release.wait(timeout=5)
                return super().transcribe(audio_path, language=language)

        transcriber, chat_model = self._fakes_for_smoke_tier()
        blocking = BlockingTranscriber(transcriber._texts)
        with api(blocking, chat_model) as client:
            client.post(
                "/v1/benchmark",
                json={"tag": "apitest-still-running", "tier": "smoke", "semdist": False},
            )
            r = client.get("/v1/benchmark/apitest-still-running/results")
            assert r.status_code == 409

            release.set()
            client.app.state.worker.wait_for_benchmark("apitest-still-running")


class TestUnknownJob:
    def test_get_unknown_job_is_404(self, api):
        with api(FakeTranscriber([]), FakeChatModel([])) as client:
            r = client.get("/v1/jobs/deadbeef")
        assert r.status_code == 404

    def test_result_of_unknown_job_is_404(self, api):
        with api(FakeTranscriber([]), FakeChatModel([])) as client:
            r = client.get("/v1/jobs/deadbeef/result")
        assert r.status_code == 404

    def test_delete_unknown_job_is_404(self, api):
        with api(FakeTranscriber([]), FakeChatModel([])) as client:
            r = client.delete("/v1/jobs/deadbeef")
        assert r.status_code == 404


class TestProblemDetails:
    """Phase 8: every error response is RFC 7807, not FastAPI's default
    {"detail": "..."} shape."""

    def test_http_exception_is_rfc7807_shaped(self, api):
        with api(FakeTranscriber([]), FakeChatModel([])) as client:
            r = client.get("/v1/jobs/deadbeef")
        assert r.headers["content-type"] == "application/problem+json"
        body = r.json()
        assert body["type"] == "about:blank"
        assert body["title"] == "Not Found"
        assert body["status"] == 404
        assert "deadbeef" in body["detail"]
        assert body["instance"] == "/v1/jobs/deadbeef"

    def test_validation_error_is_also_rfc7807_shaped(self, api):
        with api(FakeTranscriber([]), FakeChatModel([])) as client:
            r = client.patch("/v1/config", json={"llm_temperature": "not-a-number"})
        assert r.status_code == 422
        assert r.headers["content-type"] == "application/problem+json"
        body = r.json()
        assert body["status"] == 422
        assert "llm_temperature" in body["detail"]


class TestOpenAPISpecQuality:
    """Regression tests for two real bugs found in an external visual QA
    pass over the rendered Swagger docs (see docs/visual-qa-report) --
    both were runtime-correct and docs-only bugs: the server always
    behaved right, but the *generated OpenAPI schema* was misleading."""

    def test_different_status_codes_get_different_examples(self, api):
        """Bug 1: every response referencing the shared Problem schema used
        to show the *same* hardcoded example (a 404 "no such job") no
        matter what status code it was actually documenting -- a 400 or
        413 on POST /v1/jobs showed that same wrong 404 example."""
        with api(FakeTranscriber([]), FakeChatModel([])) as client:
            spec = client.get("/openapi.json").json()
        responses = spec["paths"]["/v1/jobs"]["post"]["responses"]
        example_400 = responses["400"]["content"]["application/problem+json"]["example"]
        example_413 = responses["413"]["content"]["application/problem+json"]["example"]
        assert example_400["status"] == 400
        assert example_413["status"] == 413
        assert example_400["detail"] != example_413["detail"]
        assert "no such job" not in example_400["detail"]
        assert "no such job" not in example_413["detail"]

    def test_error_responses_declare_only_the_real_media_type(self, api):
        """Every Problem-shaped response must declare only
        application/problem+json -- not also a spurious application/json
        that FastAPI's default response-model inference adds on its own
        (the actual server never sends that content type for these)."""
        with api(FakeTranscriber([]), FakeChatModel([])) as client:
            spec = client.get("/openapi.json").json()
        response_404 = spec["paths"]["/v1/jobs/{job_id}"]["get"]["responses"]["404"]
        assert list(response_404["content"].keys()) == ["application/problem+json"]
        assert response_404["content"]["application/problem+json"]["schema"] == {
            "$ref": "#/components/schemas/Problem"
        }

    def test_docx_endpoint_does_not_claim_to_return_json(self, api):
        """Bug 2: GET /v1/jobs/{job_id}/docx's 200 response had a stray
        empty application/json entry (from FastAPI inferring a schema off
        the -> FileResponse return annotation) that Swagger defaulted to,
        showing a fabricated "string" example for an endpoint that always
        actually returned a real .docx file."""
        with api(FakeTranscriber([]), FakeChatModel([])) as client:
            spec = client.get("/openapi.json").json()
        response_200 = spec["paths"]["/v1/jobs/{job_id}/docx"]["get"]["responses"]["200"]
        assert list(response_200["content"].keys()) == [
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ]

    def test_benchmark_tag_description_is_self_contained(self, api):
        """The benchmark tag's description used to tell the reader to "see
        PLAN.md's Phase 6" -- a file in the repo, not something a consumer
        reading /docs in a browser has access to."""
        with api(FakeTranscriber([]), FakeChatModel([])) as client:
            spec = client.get("/openapi.json").json()
        benchmark_tag = next(t for t in spec["tags"] if t["name"] == "benchmark")
        assert "PLAN.md" not in benchmark_tag["description"]

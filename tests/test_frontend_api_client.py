"""Tests for frontend/api_client.py -- the RFC 7807 error-message extraction
in particular, since that's the one non-trivial piece of logic here. Uses
real `httpx.Response` objects constructed directly (no network, no mock
transport needed) monkeypatched over `httpx.get`/`httpx.post`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))

from api_client import ApiClient, ApiError  # noqa: E402


@pytest.fixture
def client():
    return ApiClient("http://fake-host:8000")


class TestHealth:
    def test_returns_the_parsed_body(self, client, monkeypatch):
        monkeypatch.setattr(
            "api_client.httpx.get",
            lambda url, timeout: httpx.Response(200, json={"status": "ok"}),
        )
        assert client.health() == {"status": "ok"}

    def test_unreachable_host_raises_a_readable_api_error(self, client, monkeypatch):
        def boom(url, timeout):
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr("api_client.httpx.get", boom)
        with pytest.raises(ApiError, match="Could not reach the API"):
            client.health()


class TestErrorExtraction:
    def test_rfc7807_detail_is_surfaced(self, client, monkeypatch):
        problem = {"type": "about:blank", "title": "Not Found", "status": 404,
                   "detail": "no such job: deadbeef", "instance": "/v1/jobs/deadbeef"}
        monkeypatch.setattr("api_client.httpx.get", lambda url, timeout: httpx.Response(404, json=problem))
        with pytest.raises(ApiError, match="no such job: deadbeef"):
            client.get_job("deadbeef")

    def test_non_json_error_body_falls_back_to_raw_text(self, client, monkeypatch):
        monkeypatch.setattr(
            "api_client.httpx.get",
            lambda url, timeout: httpx.Response(500, text="internal server error"),
        )
        with pytest.raises(ApiError, match="internal server error"):
            client.get_job("x")

    def test_success_does_not_raise(self, client, monkeypatch):
        monkeypatch.setattr(
            "api_client.httpx.get",
            lambda url, timeout: httpx.Response(200, json={"id": "a1b2c3d4", "status": "done"}),
        )
        assert client.get_job("a1b2c3d4")["status"] == "done"


class TestSubmitJob:
    def test_posts_multipart_with_filename_and_refine_flag(self, client, monkeypatch):
        captured = {}

        def fake_post(url, files, data, timeout):
            captured["url"] = url
            captured["files"] = files
            captured["data"] = data
            return httpx.Response(202, json={"job_id": "a1b2c3d4", "status": "queued"})

        monkeypatch.setattr("api_client.httpx.post", fake_post)
        job_id = client.submit_job("clip.wav", b"fake-bytes", refine=True)

        assert job_id == "a1b2c3d4"
        assert captured["url"] == "http://fake-host:8000/v1/jobs"
        assert captured["files"] == {"file": ("clip.wav", b"fake-bytes")}
        assert captured["data"] == {"refine": "true"}

    def test_refine_false_is_sent_as_lowercase_string(self, client, monkeypatch):
        captured = {}

        def fake_post(url, files, data, timeout):
            captured["data"] = data
            return httpx.Response(202, json={"job_id": "a1b2c3d4", "status": "queued"})

        monkeypatch.setattr("api_client.httpx.post", fake_post)
        client.submit_job("clip.wav", b"x", refine=False)
        assert captured["data"] == {"refine": "false"}


class TestGetDocx:
    def test_returns_raw_bytes(self, client, monkeypatch):
        monkeypatch.setattr(
            "api_client.httpx.get",
            lambda url, timeout: httpx.Response(200, content=b"docx-bytes-here"),
        )
        assert client.get_docx("a1b2c3d4") == b"docx-bytes-here"


def test_base_url_trailing_slash_is_stripped():
    client = ApiClient("http://localhost:8000/")
    assert client.base_url == "http://localhost:8000"

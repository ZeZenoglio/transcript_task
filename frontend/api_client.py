"""Thin HTTP client for the transcript_task API (Phase 7/8).

The Streamlit app talks to this, not to `transcript_task.pipeline` directly
-- PLAN.md's Phase 9 note is explicit that the demo "polls the API (does
**not** import the pipeline) so it exercises the real service." Every
method here raises `ApiError` with a short, human-readable message
extracted from the API's RFC 7807 `Problem` body (Phase 8) rather than
letting a raw `httpx` exception or traceback reach the UI.
"""

from __future__ import annotations

import httpx


class ApiError(Exception):
    """A readable API failure -- app.py shows `str(exc)` directly to the
    user via `st.error`, never a traceback."""


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    detail = response.text
    try:
        body = response.json()
        detail = body.get("detail", detail)
    except ValueError:
        pass  # not JSON -- fall back to the raw text set above
    raise ApiError(f"{response.reason_phrase} ({response.status_code}): {detail}")


class ApiClient:
    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str) -> dict:
        try:
            response = httpx.get(f"{self.base_url}{path}", timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise ApiError(f"Could not reach the API at {self.base_url}: {exc}") from exc
        _raise_for_status(response)
        return response.json()

    def health(self) -> dict:
        return self._get("/health")

    def config(self) -> dict:
        return self._get("/v1/config")

    def submit_job(self, filename: str, content: bytes, refine: bool) -> str:
        try:
            response = httpx.post(
                f"{self.base_url}/v1/jobs",
                files={"file": (filename, content)},
                data={"refine": str(refine).lower()},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise ApiError(f"Could not reach the API at {self.base_url}: {exc}") from exc
        _raise_for_status(response)
        return response.json()["job_id"]

    def get_job(self, job_id: str) -> dict:
        return self._get(f"/v1/jobs/{job_id}")

    def get_result(self, job_id: str) -> dict:
        return self._get(f"/v1/jobs/{job_id}/result")

    def get_docx(self, job_id: str) -> bytes:
        try:
            response = httpx.get(f"{self.base_url}/v1/jobs/{job_id}/docx", timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise ApiError(f"Could not reach the API at {self.base_url}: {exc}") from exc
        _raise_for_status(response)
        return response.content

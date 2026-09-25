"""Smoke test: the Streamlit app imports and renders headlessly, with no
API server running. PLAN.md's Phase 9 testing note is explicit that
Streamlit UIs resist deep automated testing -- this stays deliberately
shallow (it doesn't run a real server or submit a real job; that's what
tests/test_api.py's real end-to-end integration test already covers for
the API side) rather than faking a false sense of UI coverage.

`api_base_url` is pinned to a high, essentially-guaranteed-unused port
rather than relying on the app's real default (`localhost:8000`) being
unreachable in the test environment -- it isn't always: another local
project happened to have something listening on that exact default port
while writing this, and the app hitting it instead of a "connection
refused" surfaced a real bug (see app.py's render_sidebar).
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "frontend" / "app.py")
UNREACHABLE_URL = "http://127.0.0.1:59999"


def _app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.session_state["api_base_url"] = UNREACHABLE_URL
    return at


def test_app_renders_without_raising_with_no_api_reachable():
    at = _app()
    at.run(timeout=15)
    assert not at.exception


def test_upload_screen_is_shown_by_default():
    at = _app()
    at.run(timeout=15)
    assert any("transcript_task" in h.value for h in at.title)
    assert len(at.file_uploader) == 1


def test_sidebar_shows_an_error_when_the_api_is_unreachable():
    at = _app()
    at.run(timeout=15)
    # The sidebar's health check should fail gracefully (a readable
    # st.error, not an unhandled exception propagating out of the script).
    assert not at.exception
    assert len(at.sidebar.error) >= 1
    assert "Could not reach the API" in at.sidebar.error[0].value


def test_transcribe_button_disabled_without_a_file():
    at = _app()
    at.run(timeout=15)
    transcribe_buttons = [b for b in at.button if b.label == "Transcribe"]
    assert len(transcribe_buttons) == 1
    assert transcribe_buttons[0].disabled


def test_a_200_response_with_an_unexpected_shape_degrades_to_a_warning(monkeypatch):
    """Regression test for the real bug found above: some other service
    answering at the configured URL with an unrelated 200 JSON body must
    not crash the app with a KeyError."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))
    import httpx

    monkeypatch.setattr(
        "api_client.httpx.get",
        lambda url, timeout: httpx.Response(200, json={"unrelated": "shape"}),
    )
    at = _app()
    at.run(timeout=15)
    assert not at.exception
    assert any("doesn't look like the transcript_task API" in w.value for w in at.sidebar.warning)

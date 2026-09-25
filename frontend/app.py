"""Streamlit demo (Phase 9) for the transcript_task API.

Run with:
    uv run streamlit run frontend/app.py

A deliberately minimal single-screen flow (upload -> run -> result ->
reset), written from scratch following Streamlit's own standard
upload-and-process idiom rather than forked from an external community
template -- this environment has no way to fetch one, and the plan's actual
intent ("keep it minimal") is served just as well by a small app written
directly. Talks to the API over HTTP the whole time (`api_client.py`),
never importing `transcript_task.pipeline` -- this is a demo of the
*service*, not another way to invoke the library.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import streamlit as st

from api_client import ApiClient, ApiError
from helpers import (
    docx_download_filename,
    format_duration,
    is_terminal,
    pick_transcript,
    refined_available,
    sensitivity_banner,
    stage_progress,
)

LOG_DIR = Path(__file__).resolve().parents[1] / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "streamlit.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("transcript_task.frontend")

POLL_SECONDS = 1.5

st.set_page_config(page_title="transcript_task", page_icon="🎙️", layout="centered")


def _reset() -> None:
    for key in ("job_id", "view"):
        st.session_state.pop(key, None)


def _client() -> ApiClient:
    return ApiClient(st.session_state.get("api_base_url", "http://localhost:8000"))


def _api_call(action: str, fn, *args, **kwargs):
    """Runs an ApiClient call, logging the real exception and showing a
    short readable message instead -- PLAN.md's Phase 9 note: "API errors
    surfaced as readable messages, not tracebacks."."""
    try:
        return fn(*args, **kwargs)
    except ApiError as exc:
        logger.warning("%s failed: %s", action, exc)
        st.error(f"{action} failed: {exc}")
        return None


def render_sidebar() -> None:
    with st.sidebar:
        st.text_input("API base URL", value="http://localhost:8000", key="api_base_url")
        health = _api_call("Health check", _client().health)
        if health is None:
            return
        # Anything answering at this URL is treated as untrusted input --
        # the wrong host/port could easily point at some other service
        # entirely (seen for real while testing this: another local project
        # happened to have something listening on the same default port).
        # A 200 response that doesn't look like this API's health check
        # degrades to a warning instead of a KeyError crash.
        if not isinstance(health, dict) or "asr_model" not in health:
            st.warning("Got a response, but it doesn't look like the transcript_task API -- "
                       "check the URL.")
            return
        if health.get("status") == "ok":
            st.success("API reachable")
        else:
            st.warning("API degraded -- check ffmpeg/Ollama on the server")
        st.caption(f"ASR: {health['asr_model']}\n\nLLM: {health['llm_model']}")

        config = _api_call("Fetching config", _client().config)
        if isinstance(config, dict) and "audio_extensions" in config:
            # Extensions from the server's own config (GET /v1/config, Phase 8),
            # not a hardcoded list here that could drift out of sync with it.
            st.session_state["audio_extensions"] = [
                ext.lstrip(".") for ext in config["audio_extensions"]
            ]


def render_upload_screen() -> None:
    st.title("🎙️ transcript_task")
    st.write("Upload a recording to get a corrected transcript and a reviewed Word document.")

    uploaded = st.file_uploader("Audio file", type=st.session_state.get("audio_extensions"))
    refine = st.checkbox(
        "Clean up with the LLM (refine)", value=True,
        help="If off, you'll only get the raw ASR transcript -- faster, no LLM cost.",
    )

    if st.button("Transcribe", type="primary", disabled=uploaded is None):
        job_id = _api_call(
            "Submitting the recording", _client().submit_job,
            uploaded.name, uploaded.getvalue(), refine,
        )
        if job_id is not None:
            logger.info("submitted job %s (%s, refine=%s)", job_id, uploaded.name, refine)
            st.session_state["job_id"] = job_id
            st.rerun()


def render_progress_screen(job_id: str) -> None:
    st.title("🎙️ transcript_task")
    job = _api_call("Checking job status", _client().get_job, job_id)
    if job is None:
        st.button("Start over", on_click=_reset)
        return

    status = job["status"]
    st.progress(stage_progress(status), text=f"Status: {status}")

    if status == "failed":
        st.error(f"Transcription failed: {job.get('error') or 'unknown error'}")
        st.button("Start over", on_click=_reset)
        return

    if not is_terminal(status):
        time.sleep(POLL_SECONDS)
        st.rerun()
        return

    render_result_screen(job_id, job)


def render_result_screen(job_id: str, job: dict) -> None:
    result = _api_call("Fetching the result", _client().get_result, job_id)
    if result is None:
        st.button("Start over", on_click=_reset)
        return

    summary = result.get("summary")
    banner = sensitivity_banner(summary)
    if banner:
        st.warning(banner)

    if summary:
        st.header(summary["title"])
        st.write(summary["description"])
        if summary.get("topics"):
            st.caption("Topics: " + ", ".join(summary["topics"]))

    st.caption(f"Duration: {format_duration(result.get('duration_seconds'))}")

    if refined_available(result):
        view = st.radio("Show", ["Refined", "Raw"], horizontal=True, key="view")
    else:
        view = "Raw"
        if result.get("refine_rejected"):
            st.info("The automatic cleanup was skipped by an internal quality check -- "
                    "showing the raw transcript.")
    st.text_area("Transcript", pick_transcript(result, view.lower()), height=300)

    if result.get("docx_available"):
        docx_bytes = _api_call("Fetching the document", _client().get_docx, job_id)
        if docx_bytes is not None:
            st.download_button(
                "Download .docx", data=docx_bytes,
                file_name=docx_download_filename(job),
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

    st.button("Start over", on_click=_reset)


def main() -> None:
    render_sidebar()
    job_id = st.session_state.get("job_id")
    if job_id is None:
        render_upload_screen()
    else:
        render_progress_screen(job_id)


if __name__ == "__main__":
    main()

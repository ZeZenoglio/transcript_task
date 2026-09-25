"""Tests for the pure logic behind the Streamlit demo (Phase 9). Imported
the same way tests/test_build_fixtures.py reaches into scripts/ -- frontend/
isn't part of the installed package.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "frontend"))

from helpers import (  # noqa: E402
    docx_download_filename,
    format_duration,
    is_terminal,
    pick_transcript,
    refined_available,
    sensitivity_banner,
    stage_progress,
)


class TestStageProgress:
    def test_queued_is_not_zero_and_not_complete(self):
        p = stage_progress("queued")
        assert 0 < p < 1

    def test_progress_increases_through_the_pipeline(self):
        stages = [
            "queued",
            "normalizing",
            "transcribing",
            "refining",
            "summarizing",
            "writing_docx",
        ]
        values = [stage_progress(s) for s in stages]
        assert values == sorted(values)

    def test_done_is_complete(self):
        assert stage_progress("done") == 1.0

    def test_failed_and_canceled_are_also_complete(self):
        assert stage_progress("failed") == 1.0
        assert stage_progress("canceled") == 1.0

    def test_unknown_status_does_not_raise(self):
        assert stage_progress("some-future-status") == 0.0


class TestIsTerminal:
    def test_terminal_statuses(self):
        for status in ("done", "failed", "canceled"):
            assert is_terminal(status)

    def test_non_terminal_statuses(self):
        for status in (
            "queued",
            "normalizing",
            "transcribing",
            "refining",
            "summarizing",
            "writing_docx",
        ):
            assert not is_terminal(status)


class TestPickTranscript:
    def test_refined_view_returns_refined(self):
        result = {"raw_transcript": "raw", "refined_transcript": "refined"}
        assert pick_transcript(result, "refined") == "refined"

    def test_raw_view_returns_raw_even_if_refined_exists(self):
        result = {"raw_transcript": "raw", "refined_transcript": "refined"}
        assert pick_transcript(result, "raw") == "raw"

    def test_refined_view_falls_back_to_raw_when_refined_missing(self):
        result = {"raw_transcript": "raw", "refined_transcript": None}
        assert pick_transcript(result, "refined") == "raw"

    def test_refined_view_falls_back_when_refine_was_never_requested(self):
        result = {"raw_transcript": "raw"}
        assert pick_transcript(result, "refined") == "raw"


class TestRefinedAvailable:
    def test_true_when_present(self):
        assert refined_available({"refined_transcript": "x"})

    def test_false_when_none(self):
        assert not refined_available({"refined_transcript": None})

    def test_false_when_absent(self):
        assert not refined_available({})


class TestSensitivityBanner:
    def test_high_sensitivity_gets_a_banner(self):
        assert sensitivity_banner({"sensitivity": "high"}) is not None

    def test_medium_sensitivity_gets_no_banner(self):
        # Deliberate: PLAN.md's Phase 9 note asks specifically for "high",
        # unlike docx_writer.py's own banner which also covers "medium".
        assert sensitivity_banner({"sensitivity": "medium"}) is None

    def test_low_sensitivity_gets_no_banner(self):
        assert sensitivity_banner({"sensitivity": "low"}) is None

    def test_no_summary_gets_no_banner(self):
        assert sensitivity_banner(None) is None


class TestFormatDuration:
    def test_formats_minutes_and_seconds(self):
        assert format_duration(184.2) == "3:04"

    def test_under_a_minute(self):
        assert format_duration(45) == "0:45"

    def test_none_is_an_em_dash(self):
        assert format_duration(None) == "—"

    def test_rounds_rather_than_truncates(self):
        assert format_duration(59.6) == "1:00"


class TestDocxDownloadFilename:
    def test_replaces_the_extension(self):
        assert docx_download_filename({"filename": "interview.m4a"}) == "interview.docx"

    def test_handles_a_filename_with_multiple_dots(self):
        assert docx_download_filename({"filename": "my.recording.wav"}) == "my.recording.docx"

    def test_falls_back_when_filename_missing(self):
        assert docx_download_filename({}) == "transcript.docx"

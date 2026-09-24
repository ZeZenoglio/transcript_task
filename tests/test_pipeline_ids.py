"""Tests for transcript_id: assignment, stability across --force, and
migration of legacy state files that predate Phase 3b."""

from __future__ import annotations

import json

from transcript_task.pipeline import load_state, new_transcript_id, save_state
from transcript_task.settings import Settings


def _settings(tmp_path) -> Settings:
    return Settings(output_dir=tmp_path / "output", tmp_dir=tmp_path / "tmp")


class TestNewTranscriptId:
    def test_is_a_short_hex_string(self):
        tid = new_transcript_id()
        assert len(tid) == 8
        int(tid, 16)  # raises if not valid hex

    def test_ids_are_unique(self):
        ids = {new_transcript_id() for _ in range(1000)}
        assert len(ids) == 1000


class TestLoadStateMigration:
    def test_fresh_state_has_no_items_to_migrate(self, tmp_path):
        state = load_state(_settings(tmp_path))
        assert state["items"] == {}

    def test_legacy_item_without_transcript_id_gets_one_assigned(self, tmp_path):
        settings = _settings(tmp_path)
        settings.output_dir.mkdir(parents=True)
        settings.transcripts_json.write_text(json.dumps({
            "generated_at": None, "asr_model": None, "llm_model": None,
            "items": {"old-recording.m4a": {"raw_transcript": "texto"}},
        }), encoding="utf-8")

        state = load_state(settings)

        assert "transcript_id" in state["items"]["old-recording.m4a"]
        assert len(state["items"]["old-recording.m4a"]["transcript_id"]) == 8

    def test_item_that_already_has_an_id_keeps_it(self, tmp_path):
        settings = _settings(tmp_path)
        settings.output_dir.mkdir(parents=True)
        settings.transcripts_json.write_text(json.dumps({
            "generated_at": None, "asr_model": None, "llm_model": None,
            "items": {"recording.m4a": {"transcript_id": "deadbeef"}},
        }), encoding="utf-8")

        state = load_state(settings)

        assert state["items"]["recording.m4a"]["transcript_id"] == "deadbeef"

    def test_id_is_stable_across_a_save_and_reload_cycle(self, tmp_path):
        """The identity of a recording must not change just because the
        pipeline reran on it -- only its results should."""
        settings = _settings(tmp_path)
        state = load_state(settings)
        state["items"]["recording.m4a"] = {"transcript_id": new_transcript_id()}
        first_id = state["items"]["recording.m4a"]["transcript_id"]
        save_state(state, settings)

        reloaded = load_state(settings)

        assert reloaded["items"]["recording.m4a"]["transcript_id"] == first_id

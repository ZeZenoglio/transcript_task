"""Tests for the refine-quality guard added after Phase 6's live regression-
gate demo found a real bug: a model given the refine prompt can run away
(one incident generated 163,840 tokens over 51 minutes for a single clip)
with nothing catching it. `refine_rejection_reason` is the ground-truth-free
check (content_recall/length_ratio, see text_compare.py); `stage_refine`
wires it in so a bad refine call falls back to the raw transcript instead of
shipping garbage.
"""

from __future__ import annotations

from fakes import FakeChatModel
from transcript_task.pipeline import refine_rejection_reason, stage_refine
from transcript_task.settings import Settings


class TestRefineRejectionReason:
    def test_normal_cleanup_passes(self):
        settings = Settings()
        raw = "entao eu acho que sim, sim, vamos fazer isso"
        refined = "Então, eu acho que sim, vamos fazer isso."
        assert refine_rejection_reason(raw, refined, settings) is None

    def test_legitimate_disfluency_removal_passes(self):
        # Heavy hesitation removal can shrink a lot -- must not be flagged.
        settings = Settings()
        raw = "hum hum entao entao eu eu acho que sim sim sim vamos fazer isso isso isso"
        refined = "Eu acho que sim, vamos fazer isso."
        result = refine_rejection_reason(raw, refined, settings)
        assert result is None

    def test_runaway_generation_is_rejected(self):
        # Mirrors the real incident: refined output vastly longer than raw,
        # but *contains* the original words (as a real repetition-loop
        # would, still echoing fragments of the prompt) -- isolates the
        # length_ratio check from the content_recall one below.
        settings = Settings()
        raw = "isso é um teste curto"
        refined = raw + " " + ("palavra repetida " * 500)
        result = refine_rejection_reason(raw, refined, settings)
        assert result is not None
        assert result["reason"] == "length_ratio out of bounds"
        assert result["length_ratio"] > settings.refine_max_length_ratio

    def test_near_total_content_loss_is_rejected(self):
        settings = Settings()
        raw = "primeiro segundo terceiro quarto quinto sexto setimo oitavo"
        refined = "algo completamente diferente sem relacao"
        result = refine_rejection_reason(raw, refined, settings)
        assert result is not None
        assert result["reason"] == "content_recall too low"

    def test_truncation_below_min_length_ratio_is_rejected(self):
        settings = Settings()
        raw = "uma frase razoavelmente longa com bastante conteudo para preencher"
        refined = "uma"
        result = refine_rejection_reason(raw, refined, settings)
        assert result is not None
        assert result["length_ratio"] < settings.refine_min_length_ratio

    def test_result_always_includes_the_metrics_for_auditability(self):
        settings = Settings()
        result = refine_rejection_reason("a b c d", "x", settings)
        assert result is not None
        assert "content_recall" in result
        assert "length_ratio" in result

    def test_thresholds_are_configurable(self):
        raw = "isso é um teste"
        refined = raw + " " + ("palavra repetida " * 20)
        strict = Settings(refine_max_length_ratio=1.5)
        lenient = Settings(refine_max_length_ratio=1000.0)
        assert refine_rejection_reason(raw, refined, strict) is not None
        assert refine_rejection_reason(raw, refined, lenient) is None


class TestStageRefineGuardIntegration:
    def _settings(self, tmp_path) -> Settings:
        return Settings(output_dir=tmp_path / "output", tmp_dir=tmp_path / "tmp")

    def test_good_refine_output_is_kept(self, tmp_path):
        settings = self._settings(tmp_path)
        state = {"items": {"a.m4a": {"raw_transcript": "ola tudo bem"}}}
        model = FakeChatModel(["Olá, tudo bem?"])

        stage_refine(state, settings, force=False, model=model)

        item = state["items"]["a.m4a"]
        assert item["refined_transcript"] == "Olá, tudo bem?"
        assert "refine_rejected" not in item

    def test_runaway_output_is_rejected_and_falls_back_to_raw(self, tmp_path):
        settings = self._settings(tmp_path)
        raw = "isso é um teste curto"
        state = {"items": {"a.m4a": {"raw_transcript": raw}}}
        model = FakeChatModel([raw + " " + ("palavra repetida " * 500)])

        stage_refine(state, settings, force=False, model=model)

        item = state["items"]["a.m4a"]
        assert "refined_transcript" not in item
        assert item["refine_rejected"]["reason"] == "length_ratio out of bounds"

    def test_rejected_text_is_preserved_for_review(self, tmp_path):
        settings = self._settings(tmp_path)
        raw = "isso é um teste curto"
        bad_output = "palavra repetida " * 500
        state = {"items": {"a.m4a": {"raw_transcript": raw}}}
        model = FakeChatModel([bad_output])

        stage_refine(state, settings, force=False, model=model)

        assert state["items"]["a.m4a"]["refine_rejected"]["text"] == bad_output

    def test_a_previously_accepted_result_is_cleared_on_a_later_rejection(self, tmp_path):
        """--force re-running with a worse model shouldn't leave a stale
        refined_transcript from a better one sitting around."""
        settings = self._settings(tmp_path)
        raw = "isso é um teste curto"
        state = {"items": {"a.m4a": {
            "raw_transcript": raw, "refined_transcript": "Isso é um teste curto.",
        }}}
        model = FakeChatModel(["palavra repetida " * 500])

        stage_refine(state, settings, force=True, model=model)

        item = state["items"]["a.m4a"]
        assert "refined_transcript" not in item
        assert "refine_rejected" in item

    def test_a_previously_rejected_result_is_cleared_on_a_later_success(self, tmp_path):
        settings = self._settings(tmp_path)
        raw = "ola tudo bem"
        state = {"items": {"a.m4a": {
            "raw_transcript": raw,
            "refine_rejected": {"reason": "x", "content_recall": 0.1, "length_ratio": 9.0, "text": "lixo"},
        }}}
        model = FakeChatModel(["Olá, tudo bem?"])

        stage_refine(state, settings, force=True, model=model)

        item = state["items"]["a.m4a"]
        assert item["refined_transcript"] == "Olá, tudo bem?"
        assert "refine_rejected" not in item

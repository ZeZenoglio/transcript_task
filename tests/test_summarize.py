"""Unit tests for the summarize stage (Phase 3): schema validation, the
retry-then-fallback path, filename generation, and prompt rendering.

All against a fake LLM client -- no Ollama required. The one exception is
`test_summarize_transcript_against_real_ollama`, marked `integration`.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from fakes import FakeChatModel
from transcript_task.prompts import get_summarize_template
from transcript_task.summarize import (
    TranscriptSummary,
    docx_filename,
    slugify,
    summarize_transcript,
)

VALID_SUMMARY = {
    "title": "Chamada sobre partilha de custos",
    "description": "Duas pessoas discutem quem paga o quê numa despesa recente.",
    "topics": ["dinheiro", "conflito", "família"],
    "speakers_detected": 2,
    "language_variant": "pt-PT",
    "sensitivity": "medium",
    "confidence": "high",
}


def _json(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False)


# ---------------------------------------------------------------------------
# TranscriptSummary schema
# ---------------------------------------------------------------------------

class TestTranscriptSummarySchema:
    def test_valid_payload_parses(self):
        summary = TranscriptSummary.model_validate(VALID_SUMMARY)
        assert summary.title == "Chamada sobre partilha de custos"
        assert summary.speakers_detected == 2
        assert summary.language_variant == "pt-PT"

    def test_trailing_period_is_stripped_from_title(self):
        payload = {**VALID_SUMMARY, "title": "Um título qualquer."}
        summary = TranscriptSummary.model_validate(payload)
        assert summary.title == "Um título qualquer"

    def test_rejects_unknown_sensitivity_value(self):
        payload = {**VALID_SUMMARY, "sensitivity": "extreme"}
        with pytest.raises(ValidationError):
            TranscriptSummary.model_validate(payload)

    def test_rejects_empty_topics(self):
        payload = {**VALID_SUMMARY, "topics": []}
        with pytest.raises(ValidationError):
            TranscriptSummary.model_validate(payload)

    def test_tolerates_extra_fields(self):
        payload = {**VALID_SUMMARY, "unexpected_field": "whatever"}
        summary = TranscriptSummary.model_validate(payload)
        assert summary.title == VALID_SUMMARY["title"]

    def test_fallback_is_itself_valid_and_flagged_low_confidence(self):
        summary = TranscriptSummary.fallback("pt")
        assert summary.confidence == "low"
        assert summary.sensitivity == "medium"  # safer default than "low"
        assert summary.topics  # never empty, even in the fallback

    def test_fallback_defaults_to_english_for_unknown_language(self):
        summary = TranscriptSummary.fallback("fr")
        assert summary.title == TranscriptSummary.fallback("en").title


# ---------------------------------------------------------------------------
# summarize_transcript: happy path, retry, fallback
# ---------------------------------------------------------------------------

class TestSummarizeTranscript:
    def test_happy_path_returns_parsed_summary(self):
        model = FakeChatModel([_json(VALID_SUMMARY)])
        summary = summarize_transcript(
            "transcrição bruta", "transcrição revista", model,
            language="pt", options={},
        )
        assert summary.title == VALID_SUMMARY["title"]
        assert len(model.calls) == 1

    def test_passes_json_schema_as_format(self):
        model = FakeChatModel([_json(VALID_SUMMARY)])
        summarize_transcript("raw", "refined", model, language="pt", options={})
        assert model.calls[0]["format"] == TranscriptSummary.model_json_schema()

    def test_retries_once_on_invalid_json_then_succeeds(self):
        model = FakeChatModel(["not valid json at all", _json(VALID_SUMMARY)])
        summary = summarize_transcript(
            "raw", "refined", model, language="pt", options={},
        )
        assert summary.title == VALID_SUMMARY["title"]
        assert len(model.calls) == 2
        # the retry prompt should reference the failure so the model can fix it
        assert "json" in model.calls[1]["user"].lower()

    def test_retries_once_on_schema_violation_then_succeeds(self):
        bad = {**VALID_SUMMARY, "sensitivity": "extreme"}
        model = FakeChatModel([_json(bad), _json(VALID_SUMMARY)])
        summary = summarize_transcript(
            "raw", "refined", model, language="pt", options={},
        )
        assert summary.sensitivity == "medium"
        assert len(model.calls) == 2

    def test_falls_back_after_two_failures(self):
        model = FakeChatModel(["still not json", "also not json"])
        summary = summarize_transcript(
            "raw", "refined", model, language="pt", options={},
        )
        assert summary.confidence == "low"
        assert len(model.calls) == 2  # never a third attempt

    def test_falls_back_in_requested_language(self):
        model = FakeChatModel(["bad", "bad"])
        summary = summarize_transcript(
            "raw", "refined", model, language="en", options={},
        )
        assert summary.title == TranscriptSummary.fallback("en").title

    def test_uses_correct_template_for_language(self):
        model_pt = FakeChatModel([_json(VALID_SUMMARY)])
        model_en = FakeChatModel([_json(VALID_SUMMARY)])
        summarize_transcript("raw", "refined", model_pt, language="pt", options={})
        summarize_transcript("raw", "refined", model_en, language="en", options={})
        assert model_pt.calls[0]["system"] == get_summarize_template("pt").system
        assert model_en.calls[0]["system"] == get_summarize_template("en").system
        assert model_pt.calls[0]["system"] != model_en.calls[0]["system"]

    def test_both_transcript_versions_reach_the_prompt(self):
        model = FakeChatModel([_json(VALID_SUMMARY)])
        summarize_transcript(
            "RAW_MARKER_TEXT", "REFINED_MARKER_TEXT", model, language="pt", options={},
        )
        prompt = model.calls[0]["user"]
        assert "RAW_MARKER_TEXT" in prompt
        assert "REFINED_MARKER_TEXT" in prompt


# ---------------------------------------------------------------------------
# prompt templates
# ---------------------------------------------------------------------------

class TestPromptTemplates:
    def test_pt_and_en_templates_have_distinct_ids(self):
        assert get_summarize_template("pt").id == "summarize-pt-v2"
        assert get_summarize_template("en").id == "summarize-en-v2"

    def test_unknown_language_raises(self):
        with pytest.raises(ValueError):
            get_summarize_template("fr")

    def test_render_substitutes_both_placeholders(self):
        rendered = get_summarize_template("pt").render(
            raw_transcript="AAA", refined_transcript="BBB"
        )
        assert "AAA" in rendered
        assert "BBB" in rendered


# ---------------------------------------------------------------------------
# slugify / docx_filename
# ---------------------------------------------------------------------------

class TestSlugify:
    def test_basic_slug(self):
        assert slugify("Uma Conversa Sobre Dinheiro") == "uma-conversa-sobre-dinheiro"

    def test_strips_accents_and_punctuation(self):
        assert slugify("Reunião: próximos passos!") == "reuniao-proximos-passos"

    def test_falls_back_when_nothing_survives(self):
        assert slugify("!!!???", language="pt") == "transcricao"
        assert slugify("!!!???", language="en") == "transcript"

    def test_truncates_to_max_length(self):
        slug = slugify("palavra " * 30, max_length=20)
        assert len(slug) <= 20

    def test_never_ends_with_a_hyphen_after_truncation(self):
        slug = slugify("abcde " * 10, max_length=7)
        assert not slug.endswith("-")


class TestDocxFilename:
    def test_id_and_slug(self):
        summary = TranscriptSummary.model_validate(VALID_SUMMARY)
        name = docx_filename("a1b2c3d4", summary)
        assert name == "a1b2c3d4_chamada-sobre-partilha-de-custos.docx"

    def test_no_summary_falls_back_to_id_only(self):
        assert docx_filename("a1b2c3d4", None) == "a1b2c3d4.docx"

    def test_same_title_different_id_never_collides(self):
        """Two recordings that happen to summarize to the same title must
        still produce distinct filenames -- traceability to source audio
        can't depend on the summary being unique, only on the id."""
        summary = TranscriptSummary.model_validate(VALID_SUMMARY)
        name_a = docx_filename("id-one", summary)
        name_b = docx_filename("id-two", summary)
        assert name_a != name_b
        assert name_a.startswith("id-one_")
        assert name_b.startswith("id-two_")

    def test_anonymizes_title_by_default(self):
        summary = TranscriptSummary.model_validate({
            **VALID_SUMMARY, "title": "Chamada com Marta sobre dinheiro",
        })
        name = docx_filename("a1b2c3d4", summary)
        assert "marta" not in name.lower()
        assert name.startswith("a1b2c3d4_")

    def test_anonymize_false_keeps_the_real_name(self):
        summary = TranscriptSummary.model_validate({
            **VALID_SUMMARY, "title": "Chamada com Marta sobre dinheiro",
        })
        name = docx_filename("a1b2c3d4", summary, anonymize=False)
        assert "marta" in name.lower()


# ---------------------------------------------------------------------------
# integration: real Ollama
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_summarize_transcript_against_real_ollama():
    from transcript_task.refine import OllamaChatModel
    from transcript_task.settings import Settings

    settings = Settings()
    try:
        import ollama
        ollama.list()
    except Exception:
        pytest.skip("Ollama is not reachable on this machine")

    model = OllamaChatModel(settings.llm_model)
    summary = summarize_transcript(
        "Boa tarde. Só queria confirmar a hora da reunião de amanhã.",
        "Boa tarde. Só queria confirmar a hora da reunião de amanhã.",
        model,
        language="pt",
        options=settings.llm_options,
    )
    assert isinstance(summary, TranscriptSummary)
    assert summary.title
    assert summary.confidence in ("low", "medium", "high")

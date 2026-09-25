"""Tests for the PII safety net (Phase 3b).

Uses the real spaCy pt_core_news_sm model rather than a fake -- it's small
(12 MB), loads in well under a second, and the whole point of this module is
its actual entity-recognition behaviour, which a fake would tell us nothing
about. No `integration` marker: this has no network dependency once the
model is installed (it's a pinned project dependency, not a runtime
download), so it belongs with the rest of the fast unit tests.
"""

from __future__ import annotations

from transcript_task.anonymize import redact_summary_fields, redact_text


class TestRedactText:
    def test_empty_string_is_a_noop(self):
        assert redact_text("") == ""

    def test_text_with_no_pii_is_unchanged(self):
        text = "Reunião de trabalho sem menção a pessoas específicas."
        assert redact_text(text) == text

    def test_redacts_a_person_name(self):
        result = redact_text("Chamada com Marta sobre dinheiro.")
        assert "marta" not in result.lower()
        assert "[nome]" in result

    def test_redacts_a_name_misclassified_as_a_non_person_entity(self):
        # spaCy's small pt model mislabels unfamiliar bare first names as
        # LOC/ORG rather than PER -- this is the case the "single-token,
        # any label" heuristic exists for (see module docstring).
        result = redact_text("Disputa sobre escola privada da Sofia entre os pais.")
        assert "sofia" not in result.lower()

    def test_leaves_multi_word_institution_names_alone(self):
        result = redact_text("Pagamento à Câmara Municipal de Lisboa.")
        assert "Câmara Municipal de Lisboa" in result

    def test_redacts_email(self):
        result = redact_text("Contacto: jose.silva@example.com para mais informação.")
        assert "example.com" not in result
        assert "[contacto]" in result

    def test_redacts_phone_number(self):
        result = redact_text("Ligar para 912345678 amanhã de manhã.")
        assert "912345678" not in result
        assert "[contacto]" in result

    def test_redacts_nine_digit_id_number(self):
        result = redact_text("Número de contribuinte 123456789 associado.")
        assert "123456789" not in result

    def test_contact_and_name_placeholders_are_distinct(self):
        result = redact_text("Contactar Marta pelo número 912345678.")
        assert "[nome]" in result
        assert "[contacto]" in result

    def test_english_placeholder_when_requested(self):
        result = redact_text("Chamada com Marta sobre dinheiro.", language="en")
        assert "[name]" in result
        assert "[nome]" not in result

    def test_unknown_language_falls_back_to_english_placeholder(self):
        result = redact_text("Chamada com Marta.", language="fr")
        assert "[name]" in result


class TestRedactSummaryFields:
    def test_redacts_title_description_and_topics_independently(self):
        title, description, topics = redact_summary_fields(
            "Chamada com Marta",
            "Marta discute divórcio com o advogado.",
            ["Marta", "divórcio", "advogado"],
        )
        assert "marta" not in title.lower()
        assert "marta" not in description.lower()
        assert all("marta" not in t.lower() for t in topics)
        # topics unrelated to the name pass through untouched
        assert "divórcio" in topics
        assert "advogado" in topics

    def test_returns_new_values_without_mutating_input_list(self):
        original_topics = ["Marta", "dinheiro"]
        _, _, topics = redact_summary_fields("t", "d", original_topics)
        assert original_topics == ["Marta", "dinheiro"]  # caller's list untouched
        assert topics != original_topics

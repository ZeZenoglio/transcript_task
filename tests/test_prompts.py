import pytest

from transcript_task.prompts import (
    REFINE_PT_V1,
    REFINE_PT_V2,
    get_refine_template,
)


class TestGetRefineTemplate:
    def test_v1_lookup(self):
        assert get_refine_template("refine-pt-v1") is REFINE_PT_V1

    def test_v2_lookup(self):
        assert get_refine_template("refine-pt-v2") is REFINE_PT_V2

    def test_unknown_id_raises_with_available_ids_listed(self):
        with pytest.raises(ValueError, match="refine-pt-v1"):
            get_refine_template("bogus")

    def test_render_fills_in_the_transcript(self):
        rendered = get_refine_template("refine-pt-v2").render(transcript="ola mundo")
        assert "ola mundo" in rendered


class TestRefinePromptContents:
    def test_v1_and_v2_have_distinct_ids(self):
        assert REFINE_PT_V1.id != REFINE_PT_V2.id

    def test_v2_states_the_minimal_edit_principle(self):
        # The concrete fix for the over-correction pattern real benchmark
        # runs found (v1 rewording already-correct text, e.g. "uma"->"numa"
        # with no rule justifying it) -- v2's whole point is this rule.
        assert "mantém-na exatamente como está" in REFINE_PT_V2.user_template

    def test_v2_still_forbids_inventing_summarizing_or_translating(self):
        # Must not have regressed any of v1's existing safety rules while
        # tightening the over-correction behavior.
        for keyword in ("NÃO inventes", "NÃO resumas", "NÃO omitas", "NÃO traduzas"):
            assert keyword in REFINE_PT_V2.user_template

    def test_v2_still_preserves_register_and_variant(self):
        assert "PRESERVA o registo" in REFINE_PT_V2.user_template
        assert "português original do falante" in REFINE_PT_V2.user_template

    def test_v2_still_supports_uncertain_marker(self):
        assert "[?]" in REFINE_PT_V2.user_template

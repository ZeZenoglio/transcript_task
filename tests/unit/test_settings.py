from transcript_task.settings import Settings


class TestLlmOptions:
    def test_includes_num_predict_cap(self):
        # Regression test for a real incident (Phase 6): a smaller model
        # given the refine prompt generated 163,840 tokens over ~50 minutes
        # for a single short clip, with no cap on generation length. See
        # settings.py's comment on `llm_num_predict`.
        settings = Settings()
        assert settings.llm_options["num_predict"] == settings.llm_num_predict
        assert settings.llm_num_predict > 0

    def test_num_predict_overridable_via_env(self, monkeypatch):
        monkeypatch.setenv("TRANSCRIPT_LLM_NUM_PREDICT", "256")
        settings = Settings()
        assert settings.llm_num_predict == 256
        assert settings.llm_options["num_predict"] == 256


class TestRefinePromptId:
    def test_resolves_to_a_real_prompt_template(self):
        from transcript_task.prompts import get_refine_template

        settings = Settings()
        get_refine_template(settings.refine_prompt_id)  # must not raise

    def test_default_is_v2(self):
        # Locks in a measured decision, not an assumption: v2 cut refine's
        # own WER regressions 7/30->2/30 (Common Voice) and 13/30->6/30
        # (FLEURS) vs. v1 -- see PLAN.md's third Phase 6 follow-up. A silent
        # revert to v1 here would undo that without anyone noticing.
        assert Settings().refine_prompt_id == "refine-pt-v2"

    def test_overridable_via_env_for_ab_testing(self, monkeypatch):
        monkeypatch.setenv("TRANSCRIPT_REFINE_PROMPT_ID", "refine-pt-v2")
        settings = Settings()
        assert settings.refine_prompt_id == "refine-pt-v2"

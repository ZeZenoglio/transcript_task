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

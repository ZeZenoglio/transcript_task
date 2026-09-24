from fakes import FakeChatModel
from transcript_task.eval.interpretation import write_interpretation
from transcript_task.eval.results import BenchmarkResult, ClipResult


def _result() -> BenchmarkResult:
    return BenchmarkResult(
        tier="quick", seed=1, tag="t", asr_model="m", llm_model="l",
        refine_prompt_id="r", summarize_prompt_id="s", summary_language="pt",
        clips=[ClipResult(clip_id="c", duration=5.0, wer_refined=0.1)],
    )


class TestWriteInterpretation:
    def test_returns_model_response(self):
        model = FakeChatModel(["Interpretação de teste."])
        text = write_interpretation(_result(), model)
        assert text == "Interpretação de teste."

    def test_prompt_includes_the_metrics_table(self):
        model = FakeChatModel(["ok"])
        write_interpretation(_result(), model)
        assert "wer_refined" in model.calls[0]["user"]

    def test_never_scores_is_documented_not_enforced_in_prompt(self):
        # (documentation check, not a runtime guarantee -- see module docstring)
        from transcript_task.eval.interpretation import _INSTRUCTIONS_PT
        assert "pontuação" in _INSTRUCTIONS_PT

    def test_model_failure_does_not_raise(self):
        class BoomModel:
            def chat(self, **kwargs):
                raise RuntimeError("ollama unreachable")

        text = write_interpretation(_result(), BoomModel())
        assert "indisponível" in text

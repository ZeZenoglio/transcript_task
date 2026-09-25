"""Real-model integration test for the summarize stage. Split out of
tests/unit/test_summarize.py (fake-LLM-only) so a plain `pytest` run never
needs Ollama -- opt in with `pytest -m integration`.
"""

from __future__ import annotations

import pytest

from transcript_task.summarize import TranscriptSummary, summarize_transcript


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

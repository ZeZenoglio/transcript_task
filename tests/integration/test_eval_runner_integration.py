"""Real-model integration test for the eval harness's per-clip runner. Split
out of tests/eval/test_eval_runner.py (fakes only) so a plain `pytest` run
never needs mlx-whisper/Ollama -- opt in with `pytest -m integration`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from transcript_task.eval.runner import evaluate_clip
from transcript_task.settings import Settings

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
MANIFEST = [json.loads(line) for line in (FIXTURES_DIR / "manifest.jsonl").read_text().splitlines()]
SHORT_CLIP = min(MANIFEST, key=lambda c: c["duration"])


@pytest.mark.integration
def test_evaluate_clip_against_real_models(tmp_path):
    from transcript_task.asr import MlxWhisperTranscriber
    from transcript_task.refine import OllamaChatModel

    settings = Settings()
    try:
        import ollama

        ollama.list()
    except Exception:
        pytest.skip("Ollama is not reachable on this machine")

    transcriber = MlxWhisperTranscriber(settings.asr_model)
    chat_model = OllamaChatModel(settings.llm_model)

    result = evaluate_clip(
        SHORT_CLIP,
        FIXTURES_DIR,
        settings,
        transcriber,
        chat_model,
        embedder=None,
        tmp_dir=tmp_path,
    )
    assert result.error is None
    # Real ASR against a clean, clearly-spoken FLEURS clip should be close
    # to the ground truth, not a loose "did it run" check.
    assert result.wer_raw < 0.3
    assert result.schema_valid is True

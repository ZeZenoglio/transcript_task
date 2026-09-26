"""CI-only smoke test for the macOS Actions runner (Phase 11).

Not a pytest test: this deliberately runs *outside* any test tier, since
its whole point is proving mlx-whisper actually imports and runs a real
transcription on this exact runner, using a small/fast model
(`mlx-community/whisper-tiny`) instead of the production
`whisper-large-v3-turbo` -- the tiny model's transcription quality doesn't
matter here, only that the ASR stack itself works end to end on
macos-latest's Apple Silicon hardware. Ollama isn't installed on hosted
runners, so this only covers the ASR half; the LLM half (refine/summarize)
is exercised by tests/integration/ against the real models, self-hosted only.

Usage: uv run python scripts/ci_macos_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    # Import checks -- catches anything that only breaks at import time on a
    # fresh machine (a missing extra, a platform-specific import gone wrong).
    import transcript_task.api.app  # noqa: F401
    import transcript_task.pipeline  # noqa: F401
    from transcript_task.asr import MlxWhisperTranscriber

    fixture = PROJECT_ROOT / "tests" / "fixtures" / "audio" / "fleurs_row00871.wav"
    transcriber = MlxWhisperTranscriber("mlx-community/whisper-tiny")
    result = transcriber.transcribe(fixture, language="pt")

    if not result.text.strip():
        sys.exit(f"whisper-tiny smoke test produced an empty transcript for {fixture}")

    print(f"OK: whisper-tiny transcribed {fixture.name} -> {result.text!r}")


if __name__ == "__main__":
    main()

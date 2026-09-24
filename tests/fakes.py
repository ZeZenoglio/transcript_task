"""Fakes shared across test modules.

`FakeChatModel` implements the same shape as `transcript_task.refine.ChatModel`
without calling Ollama, so refine/summarize logic can be tested without a
running model. `FakeTranscriber` does the same for `transcript_task.asr.Transcriber`.
"""

from __future__ import annotations

from pathlib import Path

from transcript_task.asr import TranscriptionResult


class FakeChatModel:
    """Returns queued responses in order; records every call it received."""

    def __init__(self, responses: list[str], usage: dict | None = None) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []
        # Mirrors OllamaChatModel.last_usage (see refine.py) so the eval
        # harness's token/timing instrumentation can be tested without a
        # real Ollama call. None (the default) means "no usage data",
        # exactly like a ChatModel implementation that doesn't report it.
        self._usage = usage
        self.last_usage: dict | None = None

    def chat(self, *, system: str, user: str, options: dict, format: dict | None = None) -> str:
        self.calls.append({"system": system, "user": user, "options": options, "format": format})
        if not self._responses:
            raise AssertionError("FakeChatModel: no more queued responses")
        if self._usage is not None:
            self.last_usage = dict(self._usage)
        return self._responses.pop(0)


class FakeTranscriber:
    """Returns queued transcriptions in order, or raises if `fail` is set."""

    def __init__(self, texts: list[str], fail: bool = False) -> None:
        self._texts = list(texts)
        self.fail = fail
        self.calls: list[Path] = []

    def transcribe(self, audio_path: Path, *, language: str) -> TranscriptionResult:
        self.calls.append(audio_path)
        if self.fail:
            raise RuntimeError("fake ASR failure")
        if not self._texts:
            raise AssertionError("FakeTranscriber: no more queued transcriptions")
        return TranscriptionResult(text=self._texts.pop(0))

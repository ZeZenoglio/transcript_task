"""Speech-to-text, behind a Protocol.

The pipeline depends on `Transcriber`, not on mlx_whisper directly, so tests
can inject a fake and the eval harness (Phase 6) can swap ASR models without
touching pipeline.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    text: str
    segments: list[Segment] = field(default_factory=list)


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path, *, language: str) -> TranscriptionResult:
        """Transcribe one audio file. Raises on failure; the caller decides
        whether to skip the file or abort the batch."""
        ...


class MlxWhisperTranscriber:
    """Whisper large-v3-turbo (or any mlx-community repo) via mlx-whisper."""

    def __init__(self, model: str) -> None:
        self.model = model

    def transcribe(self, audio_path: Path, *, language: str) -> TranscriptionResult:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=self.model,
            language=language,
            verbose=False,
        )
        segments = [
            Segment(start=round(s["start"], 2), end=round(s["end"], 2), text=s["text"].strip())
            for s in result.get("segments", [])
        ]
        return TranscriptionResult(text=result["text"].strip(), segments=segments)

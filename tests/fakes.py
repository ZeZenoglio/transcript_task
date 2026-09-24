"""Fakes shared across test modules.

`FakeChatModel` implements the same shape as `transcript_task.refine.ChatModel`
without calling Ollama, so refine/summarize logic can be tested without a
running model.
"""

from __future__ import annotations


class FakeChatModel:
    """Returns queued responses in order; records every call it received."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def chat(self, *, system: str, user: str, options: dict, format: dict | None = None) -> str:
        self.calls.append({"system": system, "user": user, "options": options, "format": format})
        if not self._responses:
            raise AssertionError("FakeChatModel: no more queued responses")
        return self._responses.pop(0)

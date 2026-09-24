"""LLM text cleanup, behind a Protocol.

The pipeline depends on `ChatModel`, not on the ollama package directly, so
tests can inject a fake and the eval harness (Phase 6) can swap models or
prompts without touching pipeline.py. Phase 3's summarisation stage reuses
this same Protocol.
"""

from __future__ import annotations

from typing import Protocol

from .prompts import PromptTemplate


class ChatModel(Protocol):
    def chat(self, *, system: str, user: str, options: dict, format: dict | None = None) -> str:
        """Send one turn to the model and return its text response.

        `format`, when given, is a JSON Schema dict requesting Ollama's
        structured-output (constrained decoding) mode -- used by the
        summarize stage to get back parseable JSON instead of prose.

        Implementations are responsible for disabling any "thinking"/reasoning
        mode -- it adds latency and nothing to a deterministic task, as
        measured in this repo's README (a naive default-thinking call took
        276s vs. 10s with it forced off, for an identical prompt).
        """
        ...


class OllamaChatModel:
    def __init__(self, model: str) -> None:
        self.model = model

    def chat(self, *, system: str, user: str, options: dict, format: dict | None = None) -> str:
        import ollama

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            think=False,
            format=format,
            options=options,
        )
        text = response["message"]["content"].strip()
        return _strip_markdown_fence(text)


def _strip_markdown_fence(text: str) -> str:
    """Some models wrap their answer in a ``` fence despite being told not to."""
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    body = lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:]
    return "\n".join(body).strip()


def refine_transcript(raw_transcript: str, model: ChatModel, template: PromptTemplate, options: dict) -> str:
    """Run one transcript through the cleanup prompt and return the result."""
    prompt = template.render(transcript=raw_transcript)
    return model.chat(system=template.system, user=prompt, options=options)

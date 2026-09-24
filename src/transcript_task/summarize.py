"""Structured title/description generation (Phase 3).

Runs after `refine`: the model sees both the raw ASR output and the cleaned
transcript, and returns a small JSON summary -- title, description, topics,
speaker count, language variant, a privacy-sensitivity flag, and the model's
own confidence. Giving it both versions lets it use the raw transcript's
disfluencies as evidence about speaker count and register, and flag where
cleanup may have changed meaning.

Ollama's structured-output mode (a JSON Schema passed as `format=`) constrains
decoding to the schema's shape, so malformed JSON should be rare -- but
`summarize_transcript` validates the result anyway, retries once with the
validation error appended to the prompt, and falls back to a low-confidence
stub rather than failing the whole file.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .prompts import get_summarize_template
from .refine import ChatModel

_FALLBACK_TITLE = {"pt": "Transcrição sem resumo", "en": "Untitled transcript"}
_FALLBACK_DESCRIPTION = {
    "pt": "Não foi possível gerar um resumo automático fiável para esta gravação.",
    "en": "An automatic summary could not be reliably generated for this recording.",
}
_FALLBACK_TOPIC = {"pt": "não classificado", "en": "unclassified"}
_SLUG_FALLBACK = {"pt": "transcricao", "en": "transcript"}


class TranscriptSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=1200)
    # The prompt asks for 3-8 topics; validation only requires at least one.
    # Rejecting a 2- or 9-item list would trigger a needless retry for zero
    # real benefit -- this field is informational, not load-bearing.
    topics: list[str] = Field(min_length=1, max_length=12)
    speakers_detected: int = Field(ge=0, le=20)
    language_variant: Literal["pt-PT", "pt-BR", "unknown"]
    sensitivity: Literal["low", "medium", "high"]
    confidence: Literal["low", "medium", "high"]

    @field_validator("title")
    @classmethod
    def _clean_title(cls, v: str) -> str:
        """Titles are meant to become filenames (see slugify/docx_filename)
        and read oddly with a trailing full stop, which models add out of
        habit despite being told not to. Strip it rather than reject it --
        rejecting on something this cosmetic would waste a retry."""
        cleaned = v.strip().rstrip(".").strip()
        return cleaned if cleaned else v.strip()

    @classmethod
    def fallback(cls, language: str = "pt") -> "TranscriptSummary":
        """A safe stand-in when the model can't produce a valid summary.

        `sensitivity` defaults to "medium", not "low": if we don't actually
        know what a recording contains, treating it as possibly sensitive is
        the safer failure mode. `confidence="low"` is what signals to a
        reviewer (and to the docx/UI) that this is a stub, not a real result.
        """
        lang = language if language in _FALLBACK_TITLE else "en"
        return cls(
            title=_FALLBACK_TITLE[lang],
            description=_FALLBACK_DESCRIPTION[lang],
            topics=[_FALLBACK_TOPIC[lang]],
            speakers_detected=0,
            language_variant="unknown",
            sensitivity="medium",
            confidence="low",
        )


def summarize_transcript(
    raw_transcript: str,
    refined_transcript: str,
    model: ChatModel,
    *,
    language: str = "pt",
    options: dict,
) -> TranscriptSummary:
    """Summarize one transcript. Never raises on a malformed model response --
    validation failures fall back to `TranscriptSummary.fallback()` after one
    retry. Exceptions from `model.chat` itself (e.g. Ollama unreachable) are
    not caught here; the caller (pipeline.stage_summarize) handles those the
    same way it handles a transcription or refine failure.
    """
    template = get_summarize_template(language)
    schema = TranscriptSummary.model_json_schema()
    prompt = template.render(raw_transcript=raw_transcript, refined_transcript=refined_transcript)

    response = model.chat(system=template.system, user=prompt, options=options, format=schema)
    try:
        return TranscriptSummary.model_validate_json(response)
    except ValidationError as first_error:
        retry_note = (
            f"\n\nA resposta anterior não é um JSON válido conforme o schema pedido "
            f"(erro: {first_error}). Responde novamente APENAS com um objeto JSON válido."
            if language == "pt" else
            f"\n\nYour previous response was not valid JSON matching the requested "
            f"schema (error: {first_error}). Respond again with ONLY a valid JSON object."
        )
        retry_response = model.chat(
            system=template.system, user=prompt + retry_note, options=options, format=schema
        )
        try:
            return TranscriptSummary.model_validate_json(retry_response)
        except ValidationError:
            return TranscriptSummary.fallback(language)


def slugify(text: str, *, max_length: int = 60, language: str = "pt") -> str:
    """Turn a title into a filesystem-safe slug, ASCII and hyphenated.

    Falls back to a generic word (localized) if nothing safe survives --
    e.g. a title that's all punctuation or non-Latin script.
    """
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", normalized).strip("-").lower()
    slug = slug[:max_length].strip("-")
    return slug or _SLUG_FALLBACK.get(language, _SLUG_FALLBACK["en"])


def docx_filename(original_key: str, summary: TranscriptSummary | None, *, language: str = "pt") -> str:
    """Build a human-readable filename for the generated document.

    The original file's stem is always appended, so two recordings can never
    collide on their output filename even if their titles slugify to the same
    string, or a summary is missing entirely -- traceability back to the
    source audio never depends on the summary having worked.
    """
    stem = Path(original_key).stem
    if summary is None:
        return f"{stem}.docx"
    return f"{slugify(summary.title, language=language)}__{stem}.docx"

"""Deterministic PII safety net for LLM-generated summary metadata.

This is a *safety net*, not the primary control -- the primary control is the
prompt instruction in prompts.py asking the model to avoid embedding
identifying details in title/description/topics in the first place. This
module exists because an LLM can ignore that instruction under real content
pressure, and unlike the transcript body (which stays untouched, in full, for
the reviewer), a title/description can end up in a filename or a document
metadata field that travels more casually than the full .docx.

Scope: only ever applied to summarize()'s output (title/description/topics)
before it's used to build a filename or written to docx core properties.
**Never applied to the transcript itself.**

The approach is deliberately recall-oriented, not precision-oriented: a false
positive here (redacting a real place name that happens to look like a
person's name) costs almost nothing on a short generated title. A false
negative -- a real name slipping through -- is exactly the failure this
module exists to catch. So it errs toward over-redaction.
"""

from __future__ import annotations

import re
from functools import lru_cache

_NAME_PLACEHOLDER = {"pt": "[nome]", "en": "[name]"}
_CONTACT_PLACEHOLDER = {"pt": "[contacto]", "en": "[contact]"}

# Structured identifiers a name-focused NER model won't catch. Patterns are
# deliberately loose -- see module docstring on why over-matching is fine here.
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"\b(?:\+\d{1,3}[\s.-]?)?\d{2,3}(?:[\s.-]?\d{2,4}){2,3}\b")
# Portuguese NIF and similar 9-digit tax/ID numbers.
_ID_NUMBER_RE = re.compile(r"\b\d{9}\b")


@lru_cache(maxsize=1)
def _nlp():
    """Load the small Portuguese spaCy model once per process.

    Import and load are lazy and cached: a run with anonymize_metadata off
    (see settings.py) never pays spaCy's model-load cost. Parser/lemmatizer/
    morphologizer are disabled -- only the NER component is needed here,
    and skipping the rest is a meaningful chunk of the load/inference cost.
    """
    import spacy

    return spacy.load("pt_core_news_sm", disable=["parser", "lemmatizer", "morphologizer"])


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping/touching (start, end) character spans."""
    if not spans:
        return []
    spans = sorted(spans)
    merged = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def redact_text(text: str, *, language: str = "pt") -> str:
    """Replace likely person names and structured PII in `text` with a
    placeholder. Safe to call on empty strings.

    Two passes, in order: structured identifiers (email/phone/ID number) via
    regex first, then names via NER on the result. Doing regex first avoids
    ever having to reconcile overlapping name/contact spans -- a phone number
    is never going to also look like part of a person's name.
    """
    if not text:
        return text

    contact_placeholder = _CONTACT_PLACEHOLDER.get(language, _CONTACT_PLACEHOLDER["en"])
    for pattern in (_EMAIL_RE, _PHONE_RE, _ID_NUMBER_RE):
        text = pattern.sub(contact_placeholder, text)

    name_placeholder = _NAME_PLACEHOLDER.get(language, _NAME_PLACEHOLDER["en"])
    doc = _nlp()(text)

    spans: list[tuple[int, int]] = []
    for ent in doc.ents:
        is_person = ent.label_ == "PER"
        # pt_core_news_sm frequently mislabels an unfamiliar bare first name
        # as LOC, ORG or MISC instead of PER -- verified against this
        # project's own real generated titles, where an uncommon first name
        # (name changed here for privacy) came back as LOC in one sentence
        # and ORG in another. A single-token entity of
        # *any* label is redacted too; a multi-word entity ("Câmara
        # Municipal de Lisboa") is left alone, since a genuine institution
        # or place name is far more likely there than a misclassified
        # person's name.
        is_probable_name = len(ent) == 1 and ent.label_ != "PER" and ent.text[:1].isupper()
        if is_person or is_probable_name:
            spans.append((ent.start_char, ent.end_char))
            continue

        # Narrower case: a name immediately following a sentence-initial
        # capitalized word (e.g. "Contactar Marta...") gets merged into one
        # multi-token non-PER entity, because the model reads the sentence's
        # first word as capitalized-therefore-proper-noun and folds the real
        # name into the same span (also verified against this project's own
        # real generated titles). Redacting the whole span would eat the leading
        # word too; redact just the entity's *last* token instead, which is
        # almost always the actual name in this pattern. Cost: a genuine
        # multi-word place/institution name that happens to open a title
        # loses its last word too -- accepted, per this module's
        # recall-over-precision stance (see docstring).
        if len(ent) >= 2 and ent.label_ != "PER" and ent.start == 0:
            last_token = ent[-1]
            spans.append((last_token.idx, last_token.idx + len(last_token.text)))

    if not spans:
        return text

    redacted = text
    for start, end in sorted(_merge_spans(spans), reverse=True):
        redacted = redacted[:start] + name_placeholder + redacted[end:]
    return redacted


def redact_summary_fields(
    title: str, description: str, topics: list[str], *, language: str = "pt"
) -> tuple[str, str, list[str]]:
    """Redact the free-text fields of a summary that can end up in a filename
    or docx metadata. Returns new (title, description, topics); the caller's
    objects are left untouched."""
    return (
        redact_text(title, language=language),
        redact_text(description, language=language),
        [redact_text(t, language=language) for t in topics],
    )

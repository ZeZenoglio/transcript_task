"""Dependency-light text comparison utilities, shared by the production
pipeline (as a runtime quality guard on the refine stage, see pipeline.py's
`stage_refine`) and the evaluation harness (as benchmark metrics, see
`eval/metrics.py`). Deliberately kept dependency-light (only `num2words`,
already a core dependency) rather than living inside `eval/`, whose other
modules pull in jiwer/sentence-transformers/mlflow -- none of which the
shipped pipeline needs just to guard against a bad refine call.

`normalize_pt` is a Portuguese-aware text normalizer for comparing two
pieces of text (used for both WER/CER scoring in eval/metrics.py and the
production content-recall guard below). Whisper's own normalizer (used
upstream by `openai-whisper`'s eval code) is English-only -- it strips
English contractions and spells out currency symbols in English. Applying
it to Portuguese text produces bogus diffs, so this is a small pt-specific
equivalent.

Steps, in order:
  1. Unicode NFKC normalize (fixes composed/decomposed accent variants).
  2. Casefold.
  3. Expand standalone digit runs to Portuguese number words via `num2words`,
     since Whisper sometimes writes "12" where a human transcript (and
     FLEURS' ground truth) writes "doze" -- a real, spurious source of WER
     that has nothing to do with transcription quality.
  4. Strip punctuation (keeping only letters, digits-that-survived, and
     whitespace).
  5. Optionally strip accents (off by default: "más"/"mas" are different
     words in Portuguese, so folding them away by default would hide real
     errors; exposed as an option for callers who want a looser comparison).
  6. Collapse whitespace.

Known limitation: `num2words` doesn't do Portuguese grammatical gender
agreement, so "2 vezes" (feminine, "duas vezes") normalizes to "dois vezes,"
not the grammatically correct form. This can occasionally cost a spurious
WER point when a hypothesis writes a digit next to a feminine noun -- judged
not worth the complexity of noun-gender detection for how rarely it matters
in practice.
"""

from __future__ import annotations

import re
import unicodedata

from num2words import num2words

_DIGIT_RUN = re.compile(r"\d+")
# Keep letters (incl. accented), digits, and whitespace; drop everything else
# (punctuation, symbols, quotes -- see the fixture manifest, which has stray
# leading quotes FLEURS itself didn't strip).
_NOT_WORD_OR_SPACE = re.compile(r"[^\w\s]", re.UNICODE)
_MULTI_SPACE = re.compile(r"\s+")


def _expand_digits(text: str) -> str:
    def repl(match: re.Match) -> str:
        try:
            return num2words(int(match.group()), lang="pt")
        except (ValueError, OverflowError):
            # A digit run so large num2words chokes on it (or one starting
            # with enough leading zeros to not parse as a clean int) is rare
            # enough in speech transcripts that leaving it as digits and
            # taking the WER hit is better than crashing the whole benchmark.
            return match.group()

    return _DIGIT_RUN.sub(repl, text)


def normalize_pt(text: str, *, strip_accents: bool = False) -> str:
    """Normalize Portuguese text for comparison. Idempotent."""
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = _expand_digits(text)
    text = _NOT_WORD_OR_SPACE.sub(" ", text)
    if strip_accents:
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


def content_recall(raw_transcript: str, refined_transcript: str) -> float:
    """Fraction of the raw transcript's normalized word *set* still present
    (as a substring token) in the refined transcript. A cheap, ground-truth-
    free guard against refine silently dropping content: doesn't care about
    order or exact phrasing, just "is this word present anywhere," so it
    tolerates refine's normal rephrasing while still catching an outright
    omission -- a name, a number, a fact dropped from otherwise-fluent
    output, which reads fine and is easy to miss without checking against
    the original."""
    raw_words = set(normalize_pt(raw_transcript).split())
    if not raw_words:
        return 1.0
    refined_words = set(normalize_pt(refined_transcript).split())
    return len(raw_words & refined_words) / len(raw_words)


def length_ratio(raw_transcript: str, refined_transcript: str) -> float:
    """refined/raw character-length ratio. A ground-truth-free guard against
    refine truncating (ratio << 1) or rambling/hallucinating (ratio >> 1) --
    this is what would have caught Phase 6's runaway-generation incident
    immediately (that clip's ratio was 4716), with no ground truth needed."""
    raw_len = len(raw_transcript)
    if raw_len == 0:
        return 1.0
    return len(refined_transcript) / raw_len

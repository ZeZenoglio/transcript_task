"""Portuguese-aware text normalizer for WER/CER comparisons.

Whisper's own normalizer (used upstream by `openai-whisper`'s eval code) is
English-only -- it strips English contractions and spells out currency
symbols in English. Applying it to Portuguese text produces bogus diffs, so
this is a small pt-specific equivalent, applied identically to both the
reference (FLEURS ground truth) and the hypothesis (ASR/refine output)
before every WER/CER computation in this package.

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
    """Normalize Portuguese text for WER/CER comparison. Idempotent."""
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = _expand_digits(text)
    text = _NOT_WORD_OR_SPACE.sub(" ", text)
    if strip_accents:
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text

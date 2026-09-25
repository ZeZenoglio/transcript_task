import numpy as np
import pytest

from transcript_task.eval.metrics import (
    character_error_rate,
    content_recall,
    evaluate_summary,
    length_ratio,
    percentile,
    refine_delta,
    semantic_distance,
    slug_uniqueness,
    word_error_rate,
)
from transcript_task.summarize import TranscriptSummary


class FakeEmbedder:
    """Deterministic bag-of-words embedder: no model download, exercises the
    same cosine-similarity math the real one uses."""

    def embed(self, texts: list[str]) -> np.ndarray:
        vocab: dict[str, int] = {}
        for text in texts:
            for word in text.lower().split():
                vocab.setdefault(word, len(vocab))
        vectors = np.zeros((len(texts), max(len(vocab), 1)))
        for i, text in enumerate(texts):
            for word in text.lower().split():
                vectors[i, vocab[word]] += 1
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms


class TestWordErrorRate:
    def test_identical_strings_is_zero(self):
        assert word_error_rate("a casa é grande", "a casa é grande") == 0.0

    def test_identical_after_normalization_is_zero(self):
        # Differs only in case and punctuation -- accents are preserved by
        # normalize_pt's default, so "é" must stay "é" on both sides here.
        assert word_error_rate("A Casa é Grande.", "a casa é grande") == 0.0

    def test_known_pair(self):
        # reference: 4 words. hypothesis substitutes "grande" -> "pequena":
        # 1 substitution / 4 reference words = 0.25.
        assert word_error_rate("a casa é grande", "a casa é pequena") == pytest.approx(0.25)

    def test_both_empty_is_zero(self):
        assert word_error_rate("", "") == 0.0

    def test_empty_reference_nonempty_hypothesis_is_one(self):
        assert word_error_rate("", "algo") == 1.0

    def test_nonempty_reference_empty_hypothesis_is_one(self):
        assert word_error_rate("algo", "") == 1.0


class TestCharacterErrorRate:
    def test_identical_is_zero(self):
        assert character_error_rate("teste", "teste") == 0.0

    def test_both_empty_is_zero(self):
        assert character_error_rate("", "") == 0.0

    def test_one_substitution(self):
        # "casa" -> "casp": 1 char sub / 4 chars = 0.25.
        assert character_error_rate("casa", "casp") == pytest.approx(0.25)


class TestSemanticDistance:
    def test_identical_text_is_zero_distance(self):
        embedder = FakeEmbedder()
        assert semantic_distance("o gato dorme", "o gato dorme", embedder) == pytest.approx(
            0.0, abs=1e-9
        )

    def test_disjoint_text_is_far(self):
        embedder = FakeEmbedder()
        d = semantic_distance("o gato dorme", "carro azul rapido", embedder)
        assert d == pytest.approx(1.0, abs=1e-9)

    def test_both_empty_is_zero(self):
        embedder = FakeEmbedder()
        assert semantic_distance("", "", embedder) == 0.0


class TestRefineDelta:
    def test_refine_improves_wer(self):
        gt = "a casa é grande"
        delta = refine_delta(
            gt, raw_transcript="a casa e grandi", refined_transcript="a casa é grande"
        )
        assert delta.wer_refined == 0.0
        assert delta.wer_raw > 0.0
        assert delta.wer_delta < 0

    def test_refine_hurts_wer_when_it_paraphrases(self):
        gt = "a casa é grande"
        delta = refine_delta(
            gt, raw_transcript="a casa é grande", refined_transcript="a residência é enorme"
        )
        assert delta.wer_raw == 0.0
        assert delta.wer_refined > 0.0
        assert delta.wer_delta > 0


class TestContentRecall:
    def test_full_overlap_is_one(self):
        assert content_recall("o gato dorme", "o gato dorme calmamente") == 1.0

    def test_dropped_word_is_partial(self):
        # raw has 3 distinct words, 2 survive in refined.
        assert content_recall("o gato preto", "o gato") == pytest.approx(2 / 3)

    def test_empty_raw_is_one(self):
        assert content_recall("", "qualquer coisa") == 1.0


class TestLengthRatio:
    def test_same_length_is_one(self):
        assert length_ratio("abcde", "fghij") == 1.0

    def test_truncation_below_one(self):
        assert length_ratio("abcdefghij", "abcde") == pytest.approx(0.5)

    def test_empty_raw_is_one(self):
        assert length_ratio("", "anything") == 1.0


class TestEvaluateSummary:
    def _summary(self, **overrides) -> TranscriptSummary:
        defaults = dict(
            title="Título",
            description="Uma descrição qualquer.",
            topics=["a", "b", "c"],
            speakers_detected=1,
            language_variant="pt-PT",
            sensitivity="low",
            confidence="high",
        )
        defaults.update(overrides)
        return TranscriptSummary(**defaults)

    def test_none_summary_is_not_schema_valid(self):
        result = evaluate_summary(
            None, used_retry=True, used_fallback=True, expected_language_variant=None
        )
        assert result.schema_valid is False
        assert result.used_fallback is True

    def test_valid_summary_passes_all_conformance_checks(self):
        result = evaluate_summary(
            self._summary(),
            used_retry=False,
            used_fallback=False,
            expected_language_variant="pt-PT",
        )
        assert result.schema_valid is True
        assert result.title_len_ok is True
        assert result.description_len_ok is True
        assert result.topic_count_ok is True
        assert result.language_variant_matches is True

    def test_topic_count_outside_prompt_guidance_flagged(self):
        # Pydantic allows 1-12 topics (see summarize.py); the prompt asks for
        # 3-8. A 1-topic summary is schema-valid but not prompt-conformant.
        result = evaluate_summary(
            self._summary(topics=["a"]),
            used_retry=False,
            used_fallback=False,
            expected_language_variant=None,
        )
        assert result.schema_valid is True
        assert result.topic_count_ok is False

    def test_language_variant_mismatch_flagged(self):
        result = evaluate_summary(
            self._summary(language_variant="pt-BR"),
            used_retry=False,
            used_fallback=False,
            expected_language_variant="pt-PT",
        )
        assert result.language_variant_matches is False

    def test_no_expected_variant_is_none(self):
        result = evaluate_summary(
            self._summary(),
            used_retry=False,
            used_fallback=False,
            expected_language_variant=None,
        )
        assert result.language_variant_matches is None

    def test_description_transcript_semdist_computed_when_given(self):
        result = evaluate_summary(
            self._summary(description="o gato dorme"),
            used_retry=False,
            used_fallback=False,
            expected_language_variant=None,
            transcript="o gato dorme",
            embedder=FakeEmbedder(),
        )
        assert result.description_transcript_semdist == pytest.approx(0.0, abs=1e-9)

    def test_description_transcript_semdist_none_without_embedder(self):
        result = evaluate_summary(
            self._summary(),
            used_retry=False,
            used_fallback=False,
            expected_language_variant=None,
        )
        assert result.description_transcript_semdist is None


class TestSlugUniqueness:
    def test_all_unique_is_one(self):
        assert slug_uniqueness(["a.docx", "b.docx", "c.docx"]) == 1.0

    def test_duplicates_reduce_the_score(self):
        assert slug_uniqueness(["a.docx", "a.docx", "b.docx"]) == pytest.approx(2 / 3)

    def test_empty_list_is_one(self):
        assert slug_uniqueness([]) == 1.0


class TestPercentile:
    def test_p50_of_odd_list_is_the_middle_value(self):
        assert percentile([1, 2, 3], 50) == 2

    def test_p0_is_min(self):
        assert percentile([5, 1, 3], 0) == 1

    def test_p100_is_max(self):
        assert percentile([5, 1, 3], 100) == 5

    def test_empty_list_is_zero(self):
        assert percentile([], 50) == 0.0

    def test_single_value(self):
        assert percentile([7], 90) == 7

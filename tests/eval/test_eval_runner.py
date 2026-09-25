import json
from pathlib import Path

from fakes import FakeChatModel, FakeTranscriber

from transcript_task.eval.runner import evaluate_clip, peak_rss_mb
from transcript_task.settings import Settings

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
MANIFEST = [json.loads(line) for line in (FIXTURES_DIR / "manifest.jsonl").read_text().splitlines()]
SHORT_CLIP = min(MANIFEST, key=lambda c: c["duration"])

VALID_SUMMARY_JSON = json.dumps(
    {
        "title": "Título de teste",
        "description": "Uma descrição de teste com conteúdo suficiente.",
        "topics": ["a", "b", "c"],
        "speakers_detected": 1,
        "language_variant": "pt-BR",
        "sensitivity": "low",
        "confidence": "high",
    }
)


class TestEvaluateClipHappyPath:
    def test_scores_wer_and_summary_against_ground_truth(self, tmp_path):
        settings = Settings()
        transcriber = FakeTranscriber([SHORT_CLIP["ground_truth"]])
        chat_model = FakeChatModel([SHORT_CLIP["ground_truth"], VALID_SUMMARY_JSON])

        result = evaluate_clip(
            SHORT_CLIP,
            FIXTURES_DIR,
            settings,
            transcriber,
            chat_model,
            embedder=None,
            tmp_dir=tmp_path,
        )

        assert result.error is None
        assert result.wer_raw == 0.0
        assert result.wer_refined == 0.0
        assert result.schema_valid is True
        assert result.docx_filename is not None
        assert result.docx_filename.endswith(".docx")

    def test_records_asr_and_refine_timing(self, tmp_path):
        settings = Settings()
        transcriber = FakeTranscriber([SHORT_CLIP["ground_truth"]])
        chat_model = FakeChatModel([SHORT_CLIP["ground_truth"], VALID_SUMMARY_JSON])

        result = evaluate_clip(
            SHORT_CLIP,
            FIXTURES_DIR,
            settings,
            transcriber,
            chat_model,
            embedder=None,
            tmp_dir=tmp_path,
        )
        assert result.asr_seconds is not None and result.asr_seconds >= 0
        assert result.refine_seconds is not None and result.refine_seconds >= 0
        assert result.realtime_factor is not None

    def test_skip_refine_leaves_refine_fields_from_raw(self, tmp_path):
        settings = Settings()
        transcriber = FakeTranscriber([SHORT_CLIP["ground_truth"]])
        chat_model = FakeChatModel([VALID_SUMMARY_JSON])

        result = evaluate_clip(
            SHORT_CLIP,
            FIXTURES_DIR,
            settings,
            transcriber,
            chat_model,
            embedder=None,
            tmp_dir=tmp_path,
            skip_refine=True,
        )
        assert result.refine_seconds is None
        # wer_refined still gets computed, against the raw transcript
        # standing in for "refined" (see runner.py: refined_transcript
        # defaults to raw_transcript when refine is skipped).
        assert result.wer_refined == result.wer_raw


class TestEvaluateClipFailureHandling:
    def test_asr_failure_is_recorded_not_raised(self, tmp_path):
        settings = Settings()
        transcriber = FakeTranscriber([], fail=True)
        chat_model = FakeChatModel([])

        result = evaluate_clip(
            SHORT_CLIP,
            FIXTURES_DIR,
            settings,
            transcriber,
            chat_model,
            embedder=None,
            tmp_dir=tmp_path,
        )
        assert result.error is not None
        assert "asr failed" in result.error
        assert result.wer_raw is None

    def test_missing_audio_file_is_recorded_not_raised(self, tmp_path):
        settings = Settings()
        bad_clip = {**SHORT_CLIP, "audio_path": "audio/does_not_exist.wav"}
        transcriber = FakeTranscriber([])
        chat_model = FakeChatModel([])

        result = evaluate_clip(
            bad_clip,
            FIXTURES_DIR,
            settings,
            transcriber,
            chat_model,
            embedder=None,
            tmp_dir=tmp_path,
        )
        assert result.error is not None
        assert "normalize failed" in result.error

    def test_summarize_fallback_is_flagged(self, tmp_path):
        settings = Settings()
        transcriber = FakeTranscriber([SHORT_CLIP["ground_truth"]])
        # Two malformed responses -> summarize_transcript exhausts its retry
        # and falls back (see summarize.py).
        chat_model = FakeChatModel(
            [
                SHORT_CLIP["ground_truth"],
                "not json",
                "still not json",
            ]
        )

        result = evaluate_clip(
            SHORT_CLIP,
            FIXTURES_DIR,
            settings,
            transcriber,
            chat_model,
            embedder=None,
            tmp_dir=tmp_path,
        )
        assert result.used_fallback is True
        assert result.used_retry is True
        assert result.schema_valid is True  # fallback is itself a valid TranscriptSummary

    def test_retry_success_is_flagged_without_fallback(self, tmp_path):
        settings = Settings()
        transcriber = FakeTranscriber([SHORT_CLIP["ground_truth"]])
        chat_model = FakeChatModel(
            [
                SHORT_CLIP["ground_truth"],
                "not json",
                VALID_SUMMARY_JSON,
            ]
        )

        result = evaluate_clip(
            SHORT_CLIP,
            FIXTURES_DIR,
            settings,
            transcriber,
            chat_model,
            embedder=None,
            tmp_dir=tmp_path,
        )
        assert result.used_retry is True
        assert result.used_fallback is False


def test_peak_rss_mb_is_positive():
    assert peak_rss_mb() > 0

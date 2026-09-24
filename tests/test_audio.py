"""Tests for audio.py against the real, diversified fixture set (Phase 4) --
four real speech clips spanning ~4s to ~37s, one each in wav/mp3/m4a/opus.

Deliberately real audio rather than only synthetic tones or fakes: this
class of bug (see TestIsAlreadyTargetFormat below) only shows up against a
real file's actual codec, the same lesson Phase 3's 255-char docx bug and
Phase 3b's spaCy mislabeling taught -- fakes exercise the code path, real
content exercises whether the code path is actually correct.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from transcript_task.audio import AudioError, convert_to_target, is_already_target_format, probe
from transcript_task.settings import Settings

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MANIFEST = [json.loads(l) for l in (FIXTURES_DIR / "manifest.jsonl").read_text().splitlines()]


def fixture_path(suffix: str) -> Path:
    entry = next(e for e in MANIFEST if e["audio_path"].endswith(suffix))
    return FIXTURES_DIR / entry["audio_path"]


class TestProbe:
    @pytest.mark.parametrize("suffix,expected_codec", [
        (".wav", "pcm_f32le"),   # FLEURS' own format -- float32, not int16
        (".mp3", "mp3"),
        (".m4a", "aac"),
        (".opus", "opus"),
    ])
    def test_reads_real_codec_per_format(self, suffix, expected_codec):
        info = probe(fixture_path(suffix))
        assert info.codec == expected_codec
        assert info.sample_rate is not None
        assert info.channels == 1

    def test_duration_matches_manifest_within_rounding(self):
        for entry in MANIFEST:
            info = probe(FIXTURES_DIR / entry["audio_path"])
            assert abs(info.duration - entry["duration"]) < 0.5

    def test_raises_audio_error_on_missing_file(self, tmp_path):
        with pytest.raises(AudioError):
            probe(tmp_path / "does-not-exist.wav")


class TestIsAlreadyTargetFormat:
    """Regression coverage for a real bug found via this fixture set: FLEURS'
    own WAV files are 16kHz mono, matching the target rate/channel count, but
    encoded as float32 PCM rather than the 16-bit PCM the pipeline actually
    normalizes to. The check used to only look at rate/channels/extension and
    would have "passed through" a float32 file unconverted."""

    def test_float32_wav_is_not_already_target_format(self):
        path = fixture_path(".wav")
        info = probe(path)
        assert info.codec == "pcm_f32le"
        assert not is_already_target_format(path, info, Settings())

    @pytest.mark.parametrize("suffix", [".mp3", ".m4a", ".opus"])
    def test_non_wav_formats_are_never_already_target_format(self, suffix):
        path = fixture_path(suffix)
        info = probe(path)
        assert not is_already_target_format(path, info, Settings())

    def test_a_real_16bit_pcm_wav_is_recognised_as_already_target_format(self, tmp_path):
        settings = Settings()
        converted = tmp_path / "converted.wav"
        convert_to_target(fixture_path(".wav"), converted, settings)
        info = probe(converted)
        assert is_already_target_format(converted, info, settings)


class TestConvertToTarget:
    @pytest.mark.parametrize("suffix", [".wav", ".mp3", ".m4a", ".opus"])
    def test_converts_every_fixture_format_to_16k_mono_pcm_s16le(self, suffix, tmp_path):
        settings = Settings()
        dst = tmp_path / "out.wav"

        convert_to_target(fixture_path(suffix), dst, settings)

        info = probe(dst)
        assert info.codec == "pcm_s16le"
        assert info.sample_rate == 16000
        assert info.channels == 1

    def test_duration_survives_conversion(self, tmp_path):
        settings = Settings()
        src = fixture_path(".opus")  # lossy source -> good stress case
        before = probe(src).duration
        dst = tmp_path / "out.wav"
        convert_to_target(src, dst, settings)
        after = probe(dst).duration
        assert abs(before - after) < 0.5

    def test_raises_audio_error_on_invalid_source(self, tmp_path):
        bad = tmp_path / "not_audio.wav"
        bad.write_text("this is not an audio file")
        with pytest.raises(AudioError):
            convert_to_target(bad, tmp_path / "out.wav", Settings())

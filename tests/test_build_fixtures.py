"""Tests for the fixture-selection logic in scripts/build_fixtures.py.

Pure logic only (no network, no ffmpeg) -- picking which rows of a manifest
become fixtures is a deterministic function of the manifest's durations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_fixtures import pick_diverse_clips, pick_formats  # noqa: E402


def _write_manifest(tmp_path: Path, durations: list[float]) -> Path:
    manifest = tmp_path / "manifest.jsonl"
    with open(manifest, "w", encoding="utf-8") as f:
        for i, d in enumerate(durations):
            f.write(json.dumps({
                "id": i, "audio_path": f"audio/fleurs_{i:05d}.wav",
                "ground_truth": f"transcript {i}", "duration": d, "split": "test",
            }) + "\n")
    return manifest


class TestPickDiverseClips:
    def test_includes_shortest_and_longest(self, tmp_path):
        manifest = _write_manifest(tmp_path, [5.0, 12.0, 3.0, 40.0, 8.0, 20.0])
        picked = pick_diverse_clips(manifest, n=4)
        durations = [p["duration"] for p in picked]
        assert min(durations) == 3.0
        assert max(durations) == 40.0

    def test_returns_requested_count(self, tmp_path):
        manifest = _write_manifest(tmp_path, [float(i) for i in range(50)])
        picked = pick_diverse_clips(manifest, n=4)
        assert len(picked) == 4

    def test_picks_are_spread_across_the_range_not_clustered(self, tmp_path):
        manifest = _write_manifest(tmp_path, [float(i) for i in range(100)])
        picked = pick_diverse_clips(manifest, n=5)
        durations = sorted(p["duration"] for p in picked)
        gaps = [b - a for a, b in zip(durations, durations[1:])]
        # no two picks should be right next to each other when 100 rows
        # spread across only 5 picks -- a naive "first N" would fail this
        assert min(gaps) > 5

    def test_small_manifest_returns_everything(self, tmp_path):
        manifest = _write_manifest(tmp_path, [1.0, 2.0, 3.0])
        picked = pick_diverse_clips(manifest, n=10)
        assert len(picked) == 3

    def test_no_duplicate_rows_picked(self, tmp_path):
        manifest = _write_manifest(tmp_path, [float(i) for i in range(20)])
        picked = pick_diverse_clips(manifest, n=4)
        ids = [p["id"] for p in picked]
        assert len(ids) == len(set(ids))


class TestPickFormats:
    def test_native_format_is_first(self):
        assert pick_formats("mp3", 4)[0] == "mp3"

    def test_no_duplicate_formats(self):
        formats = pick_formats("wav", 4)
        assert len(formats) == len(set(formats))

    def test_returns_requested_count(self):
        assert len(pick_formats("mp3", 3)) == 3

    def test_native_format_used_even_when_not_in_all_formats(self):
        # Common Voice's mp3 is a recognised format, but this should hold
        # even for a hypothetical native extension outside ALL_FORMATS.
        formats = pick_formats("flac", 2)
        assert formats[0] == "flac"
        assert len(formats) == 2

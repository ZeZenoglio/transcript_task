"""Build the small, diversified, committed test-fixture set from the full
FLEURS download (Phase 4).

Picks clips spanning the duration range (not just average-length ones --
long-form audio is exactly where Phase 2 found Whisper's repetition-loop
hallucination) and re-encodes each into a different audio format with
ffmpeg, so the committed set exercises `audio.py`'s format handling too.
FLEURS itself ships one canonical format (16kHz mono float32 WAV); this is
how the original personal test set's format variety (5 .m4a + 4 .opus)
gets reproduced deliberately instead of by accident.

Requires data/fleurs_pt/ to already exist -- run fetch_dataset.py first.

Usage:
    uv run python scripts/fetch_dataset.py
    uv run python scripts/build_fixtures.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

SOURCE_DIR = Path("data/fleurs_pt")
FIXTURES_DIR = Path("tests/fixtures")

# (target format extension, ffmpeg audio codec). "wav" needs no re-encode --
# FLEURS' own format is kept as-is for one clip, so the fixture set also
# covers the "already close to what Whisper wants" path in stage_normalize.
FORMATS: list[tuple[str, str | None]] = [
    ("wav", None),
    ("mp3", "libmp3lame"),
    ("m4a", "aac"),
    ("opus", "libopus"),
]


def pick_diverse_clips(manifest_path: Path, n: int) -> list[dict]:
    """Pick n clips spanning the duration range: shortest, longest, and
    evenly spaced points between them by duration rank."""
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    rows.sort(key=lambda r: r["duration"])
    if len(rows) <= n:
        return rows
    # Evenly spaced indices across the sorted-by-duration list, always
    # including the first (shortest) and last (longest).
    step = (len(rows) - 1) / (n - 1)
    indices = sorted({round(i * step) for i in range(n)})
    return [rows[i] for i in indices]


def convert(src: Path, dst: Path, codec: str | None) -> None:
    if codec is None:
        shutil.copyfile(src, dst)
        return
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c:a", codec, str(dst)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.exit(f"ffmpeg failed converting {src} -> {dst}: {proc.stderr.strip()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=len(FORMATS),
                     help=f"how many clips to pick (default: {len(FORMATS)}, one per format)")
    args = ap.parse_args()

    manifest_path = SOURCE_DIR / "manifest.jsonl"
    if not manifest_path.exists():
        sys.exit(f"{manifest_path} not found -- run scripts/fetch_dataset.py first.")

    clips = pick_diverse_clips(manifest_path, args.count)
    formats = FORMATS[: len(clips)]

    audio_dir = FIXTURES_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for clip, (ext, codec) in zip(clips, formats):
        src = SOURCE_DIR / clip["audio_path"]
        # Keyed by the source clip's own filename stem (itself keyed by row
        # position -- see fetch_dataset.py), not fleurs_id: that field is a
        # shared sentence/prompt id, not unique per row, and using it here
        # would risk the same silent-overwrite bug fetch_dataset.py had.
        dst_name = f"{Path(clip['audio_path']).stem}.{ext}"
        dst = audio_dir / dst_name
        convert(src, dst, codec)
        entries.append({
            **{k: v for k, v in clip.items() if k != "audio_path"},
            "audio_path": f"audio/{dst_name}",
            "source": "google/fleurs (CC-BY-4.0)",
        })
        print(f"{clip['duration']:>6.1f}s  {clip['audio_path']} -> {dst_name}")

    fixtures_manifest = FIXTURES_DIR / "manifest.jsonl"
    with open(fixtures_manifest, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    durations = [e["duration"] for e in entries]
    print(f"\n{len(entries)} fixture clips written to {audio_dir}")
    print(f"Duration range: {min(durations):.1f}s - {max(durations):.1f}s")
    print(f"Formats: {[Path(e['audio_path']).suffix for e in entries]}")
    print(f"Manifest: {fixtures_manifest}")


if __name__ == "__main__":
    main()

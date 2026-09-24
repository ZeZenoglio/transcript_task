"""Build a small, diversified, committed test-fixture set from a full
dataset download (Phase 4: FLEURS; Phase 6: Common Voice, for a noisier
tier -- see fetch_common_voice.py's docstring for why a second dataset).

Picks clips spanning the duration range (not just average-length ones --
long-form audio is exactly where Phase 2 found Whisper's repetition-loop
hallucination) and re-encodes each into a different audio format with
ffmpeg, so the committed set exercises `audio.py`'s format handling too.
Whichever format the source dataset ships natively is kept as-is for one
clip (exercising the "already close to what Whisper wants" path in
stage_normalize), and the rest are re-encoded into the other formats --
this is how the original personal test set's format variety (5 .m4a + 4
.opus) gets reproduced deliberately instead of by accident.

Usage:
    uv run python scripts/fetch_dataset.py
    uv run python scripts/build_fixtures.py                                  # FLEURS (default)

    uv run python scripts/fetch_common_voice.py
    uv run python scripts/build_fixtures.py --source-dir data/common_voice_pt \\
        --dest-dir tests/fixtures_noisy --source-ext mp3 \\
        --source-label "fsicoli/common_voice_17_0 (CC0-1.0)"
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# extension -> ffmpeg audio codec, for every format *other* than the
# source's own native one (which is always just copied, never re-encoded).
CODECS = {"wav": "pcm_s16le", "mp3": "libmp3lame", "m4a": "aac", "opus": "libopus"}
ALL_FORMATS = ["wav", "mp3", "m4a", "opus"]


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


def pick_formats(source_ext: str, n: int) -> list[str]:
    """Native format first (copy -- exercises the passthrough path), then
    up to n-1 others for re-encode diversity."""
    others = [f for f in ALL_FORMATS if f != source_ext]
    return ([source_ext] + others)[:n]


def convert(src: Path, dst: Path, target_ext: str, source_ext: str) -> None:
    if target_ext == source_ext:
        shutil.copyfile(src, dst)
        return
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c:a", CODECS[target_ext], str(dst)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.exit(f"ffmpeg failed converting {src} -> {dst}: {proc.stderr.strip()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-dir", type=Path, default=Path("data/fleurs_pt"),
                     help="dataset directory containing manifest.jsonl + audio/ (default: data/fleurs_pt)")
    ap.add_argument("--dest-dir", type=Path, default=Path("tests/fixtures"),
                     help="where to write the fixture subset (default: tests/fixtures)")
    ap.add_argument("--source-ext", default="wav",
                     help="the source dataset's native audio extension, kept as-is for one "
                          "clip rather than re-encoded (default: wav, FLEURS' own format)")
    ap.add_argument("--source-label", default="google/fleurs (CC-BY-4.0)",
                     help="short attribution string recorded in each manifest entry")
    ap.add_argument("--count", type=int, default=len(ALL_FORMATS),
                     help=f"how many clips to pick (default: {len(ALL_FORMATS)}, one per format)")
    args = ap.parse_args()

    manifest_path = args.source_dir / "manifest.jsonl"
    if not manifest_path.exists():
        sys.exit(f"{manifest_path} not found -- fetch the dataset first (see this script's docstring).")

    clips = pick_diverse_clips(manifest_path, args.count)
    formats = pick_formats(args.source_ext, len(clips))

    audio_dir = args.dest_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for clip, ext in zip(clips, formats):
        src = args.source_dir / clip["audio_path"]
        # Keyed by the source clip's own filename stem, not any dataset id
        # field -- fetch_dataset.py's docstring documents why a dataset's
        # own "id" can't be trusted as unique.
        dst_name = f"{Path(clip['audio_path']).stem}.{ext}"
        dst = audio_dir / dst_name
        convert(src, dst, ext, args.source_ext)
        entries.append({
            **{k: v for k, v in clip.items() if k != "audio_path"},
            "audio_path": f"audio/{dst_name}",
            "source": args.source_label,
        })
        print(f"{clip['duration']:>6.1f}s  {clip['audio_path']} -> {dst_name}")

    fixtures_manifest = args.dest_dir / "manifest.jsonl"
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

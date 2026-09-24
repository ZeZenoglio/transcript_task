"""Download the FLEURS Portuguese benchmark and materialize it in this
project's native shape: plain audio files + a manifest.jsonl.

FLEURS (Google, CC-BY 4.0) is clean read speech, not the noisy conversational
audio this pipeline was built for -- see docs/research-stt-landscape.md and
PLAN.md Phase 4 for why it's still the right pick: it has published Whisper
baselines to check against, and its value here is as a *relative* regression
signal ("did my change make things worse"), not an absolute quality claim.

Why not the `datasets` library's own audio decoding: newer versions require
`torchcodec` (-> torch) to decode the `audio` column automatically. Phase 2
deliberately dropped torch from this project's dependencies (it came in via
`parakeet-mlx`, which nothing here uses), and pulling it back in just to
decode a WAV file would undo that. So this script disables automatic
decoding (`Audio(decode=False)`) and writes FLEURS' own audio bytes straight
to disk -- they're already valid WAV files (16kHz mono, float32 PCM), and
this pipeline's existing ffmpeg-based normalize stage handles that codec the
same way it handles anything else a real recording throws at it.

Duration is measured with this project's own `audio.probe()` (ffprobe) on
the file actually written to disk, **not** derived from the dataset's
`num_samples` field. That field looked reliable (duration = num_samples /
16000) but turned out not to be: a spot check across the fetched test split
found it disagreed with the real, ffprobe-measured duration of the
corresponding audio bytes on roughly 60% of a random sample, by as much as
several seconds on a ~13s clip. Whether that's a genuine desync in this
dataset revision or something about how `num_samples` was computed upstream
wasn't worth chasing -- the fix is the same either way: never trust a
metadata field for a fact you can measure directly from the bytes you're
about to ship.

Filenames are keyed by each row's **position in the split** (`row_000000`,
`row_000001`, ...), not the dataset's own `id` field, which turned out to be
a shared sentence/prompt id, not a unique row id -- FLEURS records multiple
speakers reading the same prompt sentence, and they all carry the same `id`.
Keying filenames by `id` was silently overwriting one recording with another
whenever two rows shared one: out of 919 rows in this split, only 349 `id`
values were actually unique, so the first version of this script overwrote
570 of 919 clips without any error, each replaced file's manifest entry
left pointing at the wrong audio. Position in the split has no such
collision, by construction. `id` is kept in the manifest as informational
metadata only, never as a key.

Usage:
    uv run python scripts/fetch_dataset.py                  # full test split (~740 MB, 919 clips)
    uv run python scripts/fetch_dataset.py --limit 20        # a quick, small download to sanity-check
    uv run python scripts/fetch_dataset.py --out data/other  # different destination
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DATASET = "google/fleurs"
CONFIG = "pt_br"
SPLIT = "test"
# Pinned so re-running this script later reproduces the same data even if
# the upstream dataset repo changes. Re-resolve deliberately (see README in
# data/fleurs_pt/ once fetched) rather than silently floating to "main".
REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=Path("data/fleurs_pt"),
                     help="destination directory (default: data/fleurs_pt)")
    ap.add_argument("--limit", type=int, default=None,
                     help="only fetch the first N rows, for a quick check")
    args = ap.parse_args()

    from datasets import Audio, load_dataset

    from transcript_task.audio import probe

    audio_dir = args.out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # A --limit run streams rather than downloading the full ~740MB parquet
    # split just to keep 20 rows of it -- non-streaming `.select()` after
    # `load_dataset()` would download everything first regardless of limit.
    streaming = args.limit is not None
    print(f"Loading {DATASET}/{CONFIG} split={SPLIT} @ {REVISION[:12]}"
          f"{' (streaming)' if streaming else ''}...")
    ds = load_dataset(DATASET, CONFIG, split=SPLIT, revision=REVISION, streaming=streaming)
    ds = ds.cast_column("audio", Audio(decode=False))
    if args.limit:
        ds = ds.take(args.limit) if streaming else ds.select(range(min(args.limit, len(ds))))

    manifest_path = args.out / "manifest.jsonl"
    written = 0
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for position, row in enumerate(ds):
            # Keyed by position, not row["id"] -- see the module docstring.
            filename = f"fleurs_row{position:05d}.wav"
            dst = audio_dir / filename
            dst.write_bytes(row["audio"]["bytes"])
            entry = {
                "fleurs_id": row["id"],  # informational only; not unique, see above
                "audio_path": f"audio/{filename}",
                # The full-case, punctuated reference. This is what a WER
                # computation (Phase 6) should normalize consistently
                # alongside the pipeline's own output, rather than relying
                # on FLEURS' own pre-normalized version below matching
                # whatever normalizer this project ends up using.
                "ground_truth": row["raw_transcription"],
                # FLEURS' own lowercase/no-punctuation version, kept for
                # cross-checking a from-scratch normalizer against.
                "ground_truth_normalized": row["transcription"],
                # Measured from the file just written, not row["num_samples"]
                # -- see the module docstring for why that field isn't trusted.
                "duration": round(probe(dst).duration, 2),
                "split": SPLIT,
            }
            mf.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1
            if written % 100 == 0:
                print(f"  {written} clips written...")

    checksum = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    info = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "revision": REVISION,
        "license": "CC-BY-4.0",
        "source": f"https://huggingface.co/datasets/{DATASET}",
        "num_rows": written,
        "manifest_sha256": checksum,
    }
    (args.out / "DATASET_INFO.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nWrote {written} clips to {audio_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Manifest sha256: {checksum}")


if __name__ == "__main__":
    main()

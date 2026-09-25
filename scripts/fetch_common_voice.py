"""Download the Common Voice Portuguese test split and materialize it in
this project's native shape: plain audio files + a manifest.jsonl, in the
same shape `fetch_dataset.py` uses for FLEURS.

Why a second dataset at all: FLEURS (Phase 4) is clean, professionally
recorded read speech. Phase 6's first real benchmark run found that the
refine stage makes WER *worse* on FLEURS, and the honest explanation was
"there's nothing disfluent for it to clean up on studio-quality read
sentences." Common Voice is a better test of that claim specifically because
it's noisier in the way real usage is noisy: contributors record themselves
on whatever microphone they have (phone, laptop, webcam) in whatever room
they're in, so clips carry real background noise, room echo, and mic
quality variance that FLEURS' recordings don't. It's still prompted sentence
reading, not free conversation -- so it doesn't fully close the gap to noisy
*conversational* audio either, but it's a real, licensed, easy-to-fetch step
in that direction, which is what the plan's risk table originally asked for.

Why `fsicoli/common_voice_17_0` and not `mozilla-foundation/common_voice_17_0`
(the official Hugging Face org): the official repo ships a Python *loading
script* rather than plain data files, and `datasets` 5.x removed loading-
script support entirely (`trust_remote_code` "is not supported anymore" --
confirmed by actually trying it, not assumed). `fsicoli/common_voice_17_0` is
a well-known, actively maintained community mirror of the same upstream
Mozilla Common Voice release, republished as plain per-language `.tar` audio
shards + `.tsv` transcripts that don't need a script -- and it carries the
same `cc0-1.0` license Common Voice itself is released under (verified from
the repo's own README frontmatter, not assumed from the parent project's
license).

This bypasses `datasets.load_dataset` entirely and uses `huggingface_hub`
to fetch the two raw files directly (one `.tar` of audio, one `.tsv` of
transcripts), then joins them by filename -- there's no parquet/arrow layer
here to decode, so the torch/torchcodec concern from `fetch_dataset.py`
doesn't even arise.

Only what's needed for a WER benchmark is kept in the manifest: the audio
path, the transcript, and duration measured the same way `fetch_dataset.py`
learned to (via this project's own `audio.probe()`, never a metadata field
-- see that script's docstring for why that lesson exists). Common Voice's
own per-clip `client_id` (a hashed, pseudonymous contributor id) is
deliberately dropped rather than carried into the manifest: nothing here
needs to know which clips share a speaker, so there's no reason to persist
even a hashed identifier that isn't used for anything.

Usage:
    uv run python scripts/fetch_common_voice.py                  # full test split (~292 MB, ~9.5k clips)
    uv run python scripts/fetch_common_voice.py --limit 20        # a quick, small check
    uv run python scripts/fetch_common_voice.py --out data/other  # different destination
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
from pathlib import Path

REPO = "fsicoli/common_voice_17_0"
CONFIG = "pt"
SPLIT = "test"
# Pinned so re-running this script later reproduces the same data even if
# the upstream mirror changes -- see fetch_dataset.py's docstring for why
# this matters (the same reasoning applies here).
REVISION = "8262c16bf297c87a9cd88c51997c4758ed7a8ba2"
TAR_MEMBER_PREFIX = f"{CONFIG}_{SPLIT}_0"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/common_voice_pt"),
        help="destination directory (default: data/common_voice_pt)",
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="only fetch the first N rows, for a quick check"
    )
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    from transcript_task.audio import probe

    audio_dir = args.out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching transcript/{CONFIG}/{SPLIT}.tsv from {REPO} @ {REVISION[:12]}...")
    tsv_path = hf_hub_download(
        repo_id=REPO,
        repo_type="dataset",
        revision=REVISION,
        filename=f"transcript/{CONFIG}/{SPLIT}.tsv",
    )
    with open(tsv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if args.limit:
        rows = rows[: args.limit]
    wanted = {row["path"] for row in rows}

    # The whole split ships as one tar per shard (`pt_test_0.tar` for this
    # split); there's no index to seek into, so even a --limit run downloads
    # the full ~292MB shard once, exactly like fetch_dataset.py's own
    # --limit still downloads what streaming can't avoid.
    print(f"Fetching audio/{CONFIG}/{SPLIT}/{TAR_MEMBER_PREFIX}.tar (~292MB, one-time)...")
    tar_path = hf_hub_download(
        repo_id=REPO,
        repo_type="dataset",
        revision=REVISION,
        filename=f"audio/{CONFIG}/{SPLIT}/{TAR_MEMBER_PREFIX}.tar",
    )

    by_path = {row["path"]: row for row in rows}
    manifest_path = args.out / "manifest.jsonl"
    written = 0
    with tarfile.open(tar_path) as tar, open(manifest_path, "w", encoding="utf-8") as mf:
        for member in tar:
            if not member.isfile():
                continue
            clip_name = Path(member.name).name  # "pt_test_0/<path>.mp3" -> "<path>.mp3"
            if clip_name not in wanted:
                continue
            row = by_path[clip_name]
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            dst = audio_dir / clip_name
            dst.write_bytes(extracted.read())

            entry = {
                "audio_path": f"audio/{clip_name}",
                "ground_truth": row["sentence"],
                # Measured from the file just written -- see fetch_dataset.py's
                # docstring for why a dataset's own metadata isn't trusted for
                # a fact this project can just measure directly.
                "duration": round(probe(dst).duration, 2),
                "split": SPLIT,
                "accents": row.get("accents") or None,
                "variant": row.get("variant") or None,
            }
            mf.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1
            if written % 500 == 0:
                print(f"  {written} clips written...")
            if args.limit and written >= args.limit:
                break

    checksum = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    info = {
        "dataset": REPO,
        "config": CONFIG,
        "split": SPLIT,
        "revision": REVISION,
        "license": "CC0-1.0",
        "source": f"https://huggingface.co/datasets/{REPO}",
        "upstream": "https://commonvoice.mozilla.org/",
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

"""End-to-end audio -> transcript -> refined text -> Word document pipeline.

Takes a single audio file or a zip archive of several, and produces one
reviewed Word document per recording.

Stages:
  1. extract    unpack the input (zip -> many files, or a single file as-is)
                into tmp/extracted
  2. normalize  convert every audio file to 16 kHz mono WAV via ffmpeg
  3. transcribe run Whisper large-v3-turbo (MLX) over each file
  4. refine     clean each transcript with a local Ollama SLM
  5. summarize  generate a title/description/topics as structured JSON
  6. docx       write one Word document per audio file

Each stage caches its result in output/transcripts.json, so re-running skips
work that is already done. Use --force to redo everything.

Usage:
    uv run python -m transcript_task.pipeline --input recordings.zip
    uv run python -m transcript_task.pipeline --input one_interview.m4a
    uv run python -m transcript_task.pipeline                          # auto-picks the lone .zip in the project root, if any
    uv run python -m transcript_task.pipeline --input recordings.zip --force
    uv run python -m transcript_task.pipeline --only transcribe refine
    uv run python -m transcript_task.pipeline --input recordings.zip --skip-refine
"""

from __future__ import annotations

import argparse
import json
import secrets
import shutil
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

from .asr import Transcriber, MlxWhisperTranscriber
from .audio import AudioError, convert_to_target, is_already_target_format, probe
from .docx_writer import human_duration, write_docx
from .prompts import REFINE_PROMPT_TEMPLATE
from .refine import ChatModel, OllamaChatModel, refine_transcript
from .settings import PROJECT_ROOT, Settings
from .summarize import TranscriptSummary, docx_filename, summarize_transcript

STAGES = ("extract", "normalize", "transcribe", "refine", "summarize", "docx")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def log(stage: str, msg: str) -> None:
    print(f"[{stage:<9}] {msg}", flush=True)


def require(binary: str) -> None:
    if shutil.which(binary) is None:
        sys.exit(f"Required binary '{binary}' not found on PATH.")


def new_transcript_id() -> str:
    """A short, filesystem-friendly, effectively-collision-free id (2^32
    space -- ample at this tool's personal scale). Deliberately not
    time-sortable; a real `created_at` timestamp does that job once Phase 7's
    SQLite table exists, so the id itself doesn't need to double as one."""
    return secrets.token_hex(4)


def load_state(settings: Settings) -> dict:
    if settings.transcripts_json.exists():
        state = json.loads(settings.transcripts_json.read_text(encoding="utf-8"))
    else:
        state = {"generated_at": None, "asr_model": None, "llm_model": None, "items": {}}
    # Migration: items from before transcript_id existed (or a legacy state
    # file) get one assigned on load, so every downstream stage can rely on
    # it being present unconditionally.
    for item in state["items"].values():
        item.setdefault("transcript_id", new_transcript_id())
    return state


def save_state(state: dict, settings: Settings) -> None:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    state["generated_at"] = datetime.now().isoformat(timespec="seconds")
    state["asr_model"] = settings.asr_model
    state["llm_model"] = settings.llm_model
    settings.transcripts_json.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# stage 1 - extract
# ---------------------------------------------------------------------------

def resolve_input(explicit: Path | None) -> Path:
    """Figure out what to process: an explicit path, or the lone zip in ROOT."""
    if explicit is not None:
        if not explicit.exists():
            sys.exit(f"Input not found: {explicit}")
        return explicit

    zips = sorted(PROJECT_ROOT.glob("*.zip"))
    if len(zips) == 1:
        return zips[0]
    if len(zips) > 1:
        sys.exit(
            f"Multiple zip files found in {PROJECT_ROOT}; "
            f"pick one with --input <path>."
        )
    sys.exit(
        "No input given and no .zip file found in the project root. "
        "Pass a zip archive of recordings or a single audio file, e.g.\n"
        "  uv run python -m transcript_task.pipeline --input recordings.zip\n"
        "  uv run python -m transcript_task.pipeline --input interview.m4a"
    )


def stage_extract(input_path: Path, settings: Settings, force: bool) -> list[Path]:
    """Unpack the input into tmp/extracted.

    Accepts either a zip archive of recordings (skipping macOS resource-fork
    junk) or a single audio file, which is just copied in as-is so every
    later stage can treat "one file" and "many files" identically.
    """
    if force and settings.extract_dir.exists():
        shutil.rmtree(settings.extract_dir)
    settings.extract_dir.mkdir(parents=True, exist_ok=True)

    if input_path.suffix.lower() != ".zip":
        if input_path.suffix.lower() not in settings.audio_extensions:
            sys.exit(f"Unrecognised input type: {input_path.suffix} ({input_path})")
        target = settings.extract_dir / input_path.name
        if force or not target.exists():
            shutil.copyfile(input_path, target)
        log("extract", f"single audio file: {input_path.name}")
        return [target]

    extracted = 0
    with zipfile.ZipFile(input_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if info.is_dir():
                continue
            # macOS archives carry a parallel __MACOSX/._name tree of AppleDouble
            # metadata files. They share the real file's extension but hold no audio.
            if "__MACOSX" in name or Path(name).name.startswith("._"):
                continue
            if Path(name).suffix.lower() not in settings.audio_extensions:
                log("extract", f"skipping non-audio entry: {name}")
                continue
            # Flatten any directory structure, keep the original basename so a
            # transcript can always be traced back to its source recording.
            target = settings.extract_dir / Path(name).name
            if target.exists() and not force:
                continue
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1

    files = sorted(
        p for p in settings.extract_dir.iterdir()
        if p.is_file() and p.suffix.lower() in settings.audio_extensions
    )
    log("extract", f"{extracted} newly extracted, {len(files)} audio files total")
    return files


# ---------------------------------------------------------------------------
# stage 2 - normalize
# ---------------------------------------------------------------------------

def stage_normalize(files: list[Path], state: dict, settings: Settings, force: bool) -> None:
    """Convert everything to 16 kHz mono PCM WAV, which is what Whisper wants."""
    settings.normalized_dir.mkdir(parents=True, exist_ok=True)

    for path in files:
        key = path.name
        item = state["items"].setdefault(key, {})
        item.setdefault("transcript_id", new_transcript_id())
        wav = settings.normalized_dir / f"{path.stem}.wav"

        if wav.exists() and not force and item.get("normalized"):
            continue

        try:
            info = probe(path)
        except AudioError as exc:
            log("normalize", f"FAILED to probe {key}: {exc}")
            item["error"] = "probe failed"
            continue

        if info.codec is None:
            log("normalize", f"no audio stream in {key}, skipping")
            item["error"] = "no audio stream"
            continue

        already_ok = is_already_target_format(path, info, settings)
        action = "copy (already 16k mono)" if already_ok else (
            f"convert {info.codec} {info.sample_rate}Hz "
            f"{info.channels}ch -> 16k mono"
        )

        if already_ok:
            shutil.copyfile(path, wav)
        else:
            try:
                convert_to_target(path, wav, settings)
            except AudioError as exc:
                log("normalize", f"FAILED {key}: {exc}")
                item["error"] = "ffmpeg conversion failed"
                continue

        item.update({
            "source_file": key,
            "source_format": info.codec,
            "source_sample_rate": info.sample_rate,
            "source_channels": info.channels,
            "duration_seconds": round(info.duration, 2) if info.duration else None,
            "normalized": str(wav.relative_to(PROJECT_ROOT)),
        })
        item.pop("error", None)
        log("normalize", f"{key}: {action}")


# ---------------------------------------------------------------------------
# stage 3 - transcribe
# ---------------------------------------------------------------------------

def stage_transcribe(state: dict, settings: Settings, force: bool, transcriber: Transcriber | None = None) -> None:
    transcriber = transcriber or MlxWhisperTranscriber(settings.asr_model)

    pending = [
        (k, v) for k, v in state["items"].items()
        if v.get("normalized") and (force or not v.get("raw_transcript"))
    ]
    if not pending:
        log("transcribe", "nothing to do (all cached)")
        return

    log("transcribe", f"{len(pending)} file(s) with {settings.asr_model}")
    for key, item in pending:
        wav = PROJECT_ROOT / item["normalized"]
        started = time.time()
        try:
            result = transcriber.transcribe(wav, language=settings.asr_language)
        except Exception as exc:  # noqa: BLE001 - keep the batch going
            log("transcribe", f"FAILED {key}: {exc}")
            item["error"] = f"asr failed: {exc}"
            continue

        elapsed = time.time() - started
        item["raw_transcript"] = result.text
        item["segments"] = [
            {"start": s.start, "end": s.end, "text": s.text} for s in result.segments
        ]
        item["asr_seconds"] = round(elapsed, 2)
        item.pop("error", None)

        dur = item.get("duration_seconds") or 0
        speed = f"{dur / elapsed:.1f}x realtime" if elapsed > 0 and dur else ""
        log("transcribe", f"{key}: {len(item['raw_transcript'])} chars "
                          f"in {elapsed:.1f}s {speed}")
        save_state(state, settings)  # checkpoint after every file


# ---------------------------------------------------------------------------
# stage 4 - refine
# ---------------------------------------------------------------------------

def stage_refine(state: dict, settings: Settings, force: bool, model: ChatModel | None = None) -> None:
    model = model or OllamaChatModel(settings.llm_model)

    pending = [
        (k, v) for k, v in state["items"].items()
        if v.get("raw_transcript") and (force or not v.get("refined_transcript"))
    ]
    if not pending:
        log("refine", "nothing to do (all cached)")
        return

    log("refine", f"{len(pending)} file(s) with {settings.llm_model}")
    for key, item in pending:
        started = time.time()
        try:
            text = refine_transcript(
                item["raw_transcript"], model, REFINE_PROMPT_TEMPLATE, settings.llm_options
            )
        except Exception as exc:  # noqa: BLE001
            log("refine", f"FAILED {key}: {exc}")
            item["refine_error"] = str(exc)
            continue

        item["refined_transcript"] = text
        item["refine_seconds"] = round(time.time() - started, 2)
        item.pop("refine_error", None)
        log("refine", f"{key}: {len(text)} chars in {item['refine_seconds']:.1f}s")
        save_state(state, settings)


# ---------------------------------------------------------------------------
# stage 5 - summarize
# ---------------------------------------------------------------------------

def stage_summarize(state: dict, settings: Settings, force: bool, model: ChatModel | None = None) -> None:
    model = model or OllamaChatModel(settings.llm_model)

    pending = [
        (k, v) for k, v in state["items"].items()
        if v.get("raw_transcript") and (force or not v.get("summary"))
    ]
    if not pending:
        log("summarize", "nothing to do (all cached)")
        return

    log("summarize", f"{len(pending)} file(s) with {settings.llm_model} ({settings.summary_language})")
    for key, item in pending:
        started = time.time()
        try:
            summary = summarize_transcript(
                item["raw_transcript"],
                item.get("refined_transcript") or item["raw_transcript"],
                model,
                language=settings.summary_language,
                options=settings.llm_options,
            )
        except Exception as exc:  # noqa: BLE001 - keep the batch going
            log("summarize", f"FAILED {key}: {exc}")
            item["summarize_error"] = str(exc)
            continue

        item["summary"] = summary.model_dump()
        item["summarize_seconds"] = round(time.time() - started, 2)
        item.pop("summarize_error", None)
        log("summarize", f"{key}: \"{summary.title}\" "
                          f"(confidence={summary.confidence}, sensitivity={summary.sensitivity}) "
                          f"in {item['summarize_seconds']:.1f}s")
        save_state(state, settings)


# ---------------------------------------------------------------------------
# stage 6 - docx
# ---------------------------------------------------------------------------

def stage_docx(state: dict, settings: Settings) -> None:
    settings.docx_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for key, item in sorted(state["items"].items()):
        if not (item.get("refined_transcript") or item.get("raw_transcript")):
            continue

        summary = TranscriptSummary.model_validate(item["summary"]) if item.get("summary") else None
        filename = docx_filename(
            item["transcript_id"], summary,
            language=settings.summary_language,
            anonymize=settings.anonymize_metadata,
        )
        out = settings.docx_dir / filename
        write_docx(key, item, out, settings)
        item["docx"] = str(out.relative_to(PROJECT_ROOT))
        written += 1
        log("docx", f"{out.name}")

    log("docx", f"{written} document(s) written to {settings.docx_dir.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=None, metavar="PATH",
                    help="a zip archive of recordings, or a single audio file. "
                         "If omitted, uses the only .zip in the project root.")
    ap.add_argument("--force", action="store_true",
                    help="redo every stage, ignoring cached results")
    ap.add_argument("--only", nargs="+", choices=STAGES, metavar="STAGE",
                    help=f"run only these stages ({', '.join(STAGES)})")
    ap.add_argument("--skip-refine", action="store_true",
                    help="transcribe only, no LLM cleanup pass")
    args = ap.parse_args()

    stages = set(args.only) if args.only else set(STAGES)
    if args.skip_refine:
        stages.discard("refine")

    require("ffmpeg")
    require("ffprobe")

    settings = Settings()
    state = load_state(settings)
    t0 = time.time()

    files: list[Path] = []
    if "extract" in stages:
        input_path = resolve_input(args.input)
        files = stage_extract(input_path, settings, args.force)
    else:
        files = sorted(
            p for p in settings.extract_dir.glob("*")
            if p.is_file() and p.suffix.lower() in settings.audio_extensions
        ) if settings.extract_dir.exists() else []

    if "normalize" in stages:
        stage_normalize(files, state, settings, args.force)
        save_state(state, settings)

    if "transcribe" in stages:
        stage_transcribe(state, settings, args.force)
        save_state(state, settings)

    if "refine" in stages:
        stage_refine(state, settings, args.force)
        save_state(state, settings)

    if "summarize" in stages:
        stage_summarize(state, settings, args.force)
        save_state(state, settings)

    if "docx" in stages:
        stage_docx(state, settings)
        save_state(state, settings)

    ok = sum(1 for v in state["items"].values() if v.get("raw_transcript"))
    audio = sum(v.get("duration_seconds") or 0 for v in state["items"].values())
    log("done", f"{ok}/{len(state['items'])} transcribed · "
                f"{human_duration(audio)} of audio · "
                f"{time.time() - t0:.1f}s wall clock")
    log("done", f"JSON: {settings.transcripts_json.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

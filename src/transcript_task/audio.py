"""Audio probing and normalization: thin wrappers around ffprobe/ffmpeg.

Kept free of pipeline bookkeeping (state dicts, logging, caching) so it can
be unit-tested with fake subprocess results and reused by anything else that
needs "what format is this file, and can I get it to 16kHz mono".
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .settings import Settings


class AudioError(Exception):
    """Raised when ffprobe/ffmpeg fail on a given file."""


@dataclass
class AudioInfo:
    codec: str | None
    sample_rate: int | None
    channels: int | None
    duration: float | None


def probe(path: Path) -> AudioInfo:
    """Return codec/sample-rate/channels/duration for an audio file.

    Raises AudioError if ffprobe itself fails (e.g. the file is not valid
    media). A file with no audio stream at all is not an error here -- it
    comes back as an AudioInfo with codec=None, and it's the caller's call
    whether that's fatal.
    """
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_name,sample_rate,channels",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise AudioError(exc.stderr.strip()[:200]) from exc

    data = json.loads(out)
    streams = data.get("streams") or [{}]
    stream = streams[0]
    duration = data.get("format", {}).get("duration")
    return AudioInfo(
        codec=stream.get("codec_name"),
        sample_rate=int(stream["sample_rate"]) if stream.get("sample_rate") else None,
        channels=stream.get("channels"),
        duration=float(duration) if duration else None,
    )


def is_already_target_format(path: Path, info: AudioInfo, settings: Settings) -> bool:
    """True if `path` is already 16kHz mono 16-bit PCM WAV and can just be
    copied, rather than needing an ffmpeg pass.

    Checks the actual sample format (`info.codec`), not just the sample rate,
    channel count and `.wav` extension -- a WAV can just as easily hold
    float32 PCM as 16-bit PCM (FLEURS' own files do; see
    tests/fixtures/audio/fleurs_01839.wav, kept specifically to exercise this
    path). mlx-whisper's own audio loader tolerates float32 fine, so this
    wasn't silently producing bad transcripts, but this function's job is to
    guarantee "already normalized", and a codec it never checked wasn't
    actually guaranteeing that.
    """
    return (
        path.suffix.lower() == ".wav"
        and info.codec == settings.target_codec
        and info.sample_rate == settings.target_sample_rate
        and info.channels == settings.target_channels
    )


def convert_to_target(src: Path, dst: Path, settings: Settings) -> None:
    """Downmix/resample `src` into `dst` as 16-bit PCM WAV at the target rate.

    Raises AudioError on ffmpeg failure.
    """
    proc = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(src),
            "-ac",
            str(settings.target_channels),
            "-ar",
            str(settings.target_sample_rate),
            "-c:a",
            settings.target_codec,
            str(dst),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AudioError(proc.stderr.strip()[:200])

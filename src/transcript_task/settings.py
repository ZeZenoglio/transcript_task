"""Central configuration for the pipeline.

This used to be a flat module of constants (config.py) imported directly by
stage functions -- convenient for a script, but it meant every stage was
implicitly coupled to global state, and nothing could be overridden per-run.
`Settings` is now a pydantic model instantiated once and **passed into** each
stage explicitly, which is what will let a future API read and patch the
running configuration (Phase 7) without restarting the process.

Any field can be overridden by an environment variable prefixed
`TRANSCRIPT_` (e.g. `TRANSCRIPT_LLM_MODEL=qwen3.5:4b`) or via a `.env` file
at the project root -- both are just pydantic-settings' defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# The repo root, computed once from this file's location:
# src/transcript_task/settings.py -> src/transcript_task -> src -> <root>.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRANSCRIPT_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- paths ---------------------------------------------------------
    tmp_dir: Path = PROJECT_ROOT / "tmp"
    output_dir: Path = PROJECT_ROOT / "output"

    @property
    def extract_dir(self) -> Path:
        return self.tmp_dir / "extracted"

    @property
    def normalized_dir(self) -> Path:
        return self.tmp_dir / "normalized"

    @property
    def transcripts_json(self) -> Path:
        return self.output_dir / "transcripts.json"

    @property
    def docx_dir(self) -> Path:
        return self.output_dir / "docx"

    # --- models ----------------------------------------------------------
    # Whisper large-v3-turbo via MLX: best Portuguese accuracy measured on
    # real recordings, ~15x realtime on an M4 after warmup. See README.md.
    asr_model: str = "mlx-community/whisper-large-v3-turbo"
    asr_language: str = "pt"

    # Ollama SLM used to clean up the raw ASR output. Thinking is forced off
    # in refine.py regardless of this option -- see the module docstring
    # there for why. 9b measured better content preservation than 4b on real
    # recordings (0.981 vs 0.969 mean content recall) for ~1.9x the runtime;
    # switch to "qwen3.5:4b" if throughput matters more than fidelity.
    llm_model: str = "qwen3.5:9b"
    llm_temperature: float = 0.2
    llm_num_ctx: int = 16384
    # A hard cap on generated tokens per call. Without one, a model that
    # doesn't reliably stop can run away: benchmarking a smaller model
    # (Phase 6) hit this for real -- llama3.2:1b given the refine prompt
    # generated 163,840 tokens (a wall of repeated text) over 3041 seconds
    # for a single ~10s audio clip, instead of the ~30 tokens a working
    # refine call takes. 8192 is generous for a single recording's refine
    # or summary output (FLEURS clips need under 100; the longest real
    # recording measured so far needed far less) while bounding the worst
    # case to minutes, not the better part of an hour.
    llm_num_predict: int = 8192

    @property
    def llm_options(self) -> dict:
        return {
            "temperature": self.llm_temperature,
            "num_ctx": self.llm_num_ctx,
            "num_predict": self.llm_num_predict,
        }

    # --- refine quality guard --------------------------------------------
    # A ground-truth-free sanity check on the refine stage's own output
    # (see text_compare.py), run right after every refine call, not just in
    # the eval harness. If refine's output fails either check, it's
    # discarded and the raw transcript is used instead -- see
    # pipeline.stage_refine. Thresholds are deliberately generous rather
    # than tight: real disfluent speech can legitimately shrink a lot when
    # refine correctly strips hesitations/repetitions (see prompts.py's
    # refine instructions), and the goal here is only to catch catastrophic
    # failures (truncation, runaway repetition), not flag normal cleanup.
    # Not yet calibrated against real disfluent conversational audio (the
    # FLEURS/Common Voice benchmarks are both prompted sentence-reading, not
    # spontaneous speech) -- treat these as a reasonable starting point, not
    # a tuned constant.
    refine_min_content_recall: float = 0.5
    refine_min_length_ratio: float = 0.3
    refine_max_length_ratio: float = 2.5

    # --- summarization -------------------------------------------------
    # Language of the generated title/description/topics. The transcript
    # itself is never translated -- this only picks which prompt variant
    # (see prompts.py) the summarize stage uses.
    summary_language: Literal["pt", "en"] = "pt"

    # Defense-in-depth PII safety net (see anonymize.py) applied to the
    # filename slug and docx metadata (title/subject/keywords) built from the
    # summary -- never to the transcript itself, and never to the visible
    # "Resumo" section in the document body. On by default: it's a safety
    # net, not a feature someone should have to opt into.
    anonymize_metadata: bool = True

    # --- audio -------------------------------------------------------------
    # Whisper expects 16 kHz mono. Anything not already in that shape gets
    # converted by ffmpeg. target_codec matters too, not just rate/channels --
    # a WAV can hold float32 PCM as easily as 16-bit (FLEURS' own files do),
    # and is_already_target_format() checks all three before skipping ffmpeg.
    target_sample_rate: int = 16000
    target_channels: int = 1
    target_codec: str = "pcm_s16le"
    audio_extensions: frozenset[str] = Field(
        default_factory=lambda: frozenset({
            ".m4a", ".opus", ".mp3", ".wav", ".ogg", ".flac",
            ".aac", ".wma", ".amr", ".mp4", ".webm", ".m4b", ".3gp",
        })
    )

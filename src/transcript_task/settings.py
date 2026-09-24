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

    @property
    def llm_options(self) -> dict:
        return {"temperature": self.llm_temperature, "num_ctx": self.llm_num_ctx}

    # --- audio -------------------------------------------------------------
    # Whisper expects 16 kHz mono. Anything not already in that shape gets
    # converted by ffmpeg.
    target_sample_rate: int = 16000
    target_channels: int = 1
    audio_extensions: frozenset[str] = Field(
        default_factory=lambda: frozenset({
            ".m4a", ".opus", ".mp3", ".wav", ".ogg", ".flac",
            ".aac", ".wma", ".amr", ".mp4", ".webm", ".m4b", ".3gp",
        })
    )

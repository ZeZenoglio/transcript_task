"""Shared configuration for the transcription pipeline."""

from pathlib import Path

ROOT = Path(__file__).parent

# --- Paths -------------------------------------------------------------
# The input itself (a zip of recordings, or a single audio file) is resolved
# at runtime from a CLI argument -- see resolve_input() in pipeline.py.
TMP_DIR = ROOT / "tmp"
EXTRACT_DIR = TMP_DIR / "extracted"
NORMALIZED_DIR = TMP_DIR / "normalized"
OUTPUT_DIR = ROOT / "output"
TRANSCRIPTS_JSON = OUTPUT_DIR / "transcripts.json"
DOCX_DIR = OUTPUT_DIR / "docx"

# --- Models ------------------------------------------------------------
# Whisper large-v3-turbo via MLX: best European-Portuguese accuracy we
# measured, ~15x realtime on an M4 after warmup.
ASR_MODEL = "mlx-community/whisper-large-v3-turbo"
ASR_LANGUAGE = "pt"

# Ollama SLM used to clean up the raw ASR output. Thinking is forced off.
# 9b measured better content preservation than 4b on these recordings
# (0.981 vs 0.969 mean content recall) for ~1.9x the runtime. Swap to
# "qwen3.5:4b" if throughput matters more than fidelity.
LLM_MODEL = "qwen3.5:9b"
LLM_OPTIONS = {"temperature": 0.2, "num_ctx": 16384}

# --- Audio -------------------------------------------------------------
# Whisper expects 16 kHz mono. Anything not already in that shape gets
# converted by ffmpeg.
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
AUDIO_EXTENSIONS = {
    ".m4a", ".opus", ".mp3", ".wav", ".ogg", ".flac",
    ".aac", ".wma", ".amr", ".mp4", ".webm", ".m4b", ".3gp",
}

# --- Prompt ------------------------------------------------------------
REFINE_SYSTEM = (
    "És um revisor profissional de transcrições em português. "
    "Devolves sempre apenas o texto revisto, sem comentários nem explicações."
)

REFINE_PROMPT = """Abaixo está a transcrição automática de um ficheiro de áudio em português.
A transcrição foi gerada por um modelo de reconhecimento de fala e pode conter erros.

A tua tarefa é preparar este texto para revisão humana. Regras obrigatórias:

1. CORRIGE erros evidentes de reconhecimento de fala, ortografia, acentuação e concordância.
2. PONTUA e divide em frases e parágrafos de forma natural e legível.
3. REMOVE apenas ruído de oralidade sem conteúdo (hesitações como "hum", "ãh", gaguejos e
   repetições acidentais da mesma palavra). Mantém repetições que sejam intencionais ou enfáticas.
4. NÃO inventes informação. NÃO resumas. NÃO omitas ideias. NÃO traduzas.
5. PRESERVA o registo do falante (informal, coloquial, ou mesmo grosseiro). Não suavizes o tom.
6. MANTÉM o português original do falante (europeu ou brasileiro). Não converjas para outra variante.
7. Quando um trecho for inaudível ou ambíguo e não conseguires deduzir com confiança,
   mantém a tua melhor hipótese seguida de [?].
8. Se identificares mudança de interlocutor, inicia um novo parágrafo.

Responde APENAS com o texto revisto.

TRANSCRIÇÃO ORIGINAL:
---
{transcript}
---"""

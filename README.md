# Transcript Tool

A small, local speech-to-text utility. Give it one audio recording, or a zip
of many, and it produces a reviewed Word document per recording — corrected,
punctuated, and traceable back to its source file. Everything runs on-device
via [Ollama](https://ollama.com) and [MLX](https://github.com/ml-explore/mlx) —
no audio or text leaves the machine.

```
input (one file, or a zip of many)
  → extract → normalize (ffmpeg) → transcribe (Whisper) → refine (Ollama) → .docx
```

> **Privacy note.** This pipeline was developed and benchmarked against a private
> set of recordings, which are not in this repository and are excluded by
> `.gitignore`. A couple of transcript excerpts quoted below, kept only as evidence
> for a model comparison, are redacted — personal names appear as `[NOME]`.
> Nothing sent to a model here leaves the machine, which is the point of running
> it locally at all: whatever you feed it stays private by construction.

## Quick start

```bash
brew install ffmpeg          # if not already present
uv sync
ollama serve                 # if not already running
uv run python pipeline.py --input recordings.zip     # a zip of several files
uv run python pipeline.py --input interview.m4a       # or just one file
```

Results land in:

| Path | Contents |
|---|---|
| `output/transcripts.json` | every transcript, raw + refined, with timings and segment timestamps |
| `output/docx/` | one Word document per audio file |
| `tmp/extracted/` | audio unpacked from the input |
| `tmp/normalized/` | 16 kHz mono WAVs fed to the ASR model |

## Model choices

Both were picked by benchmarking real candidates on real recordings, not on
published leaderboard numbers.

### Speech-to-text — `mlx-community/whisper-large-v3-turbo`

Whisper large-v3-turbo running through [`mlx-whisper`](https://github.com/ml-explore/mlx-examples/tree/main/whisper),
which uses Metal on Apple Silicon. ~1.6 GB, **~15x realtime on an M4** after warmup
(13:35 of test audio transcribed in 77 s).

The main alternative considered was **NVIDIA Parakeet TDT 0.6B v3** via
`parakeet-mlx`. On paper it is the more attractive option: 600M params, 25 European
languages including Portuguese, and roughly 48x Whisper throughput. It measured
**~3 s for 43 s of audio** here, genuinely faster.

It was rejected on output quality. Parakeet v3 auto-detects language with no way to
pin it, and on noisy, conversational, phone-quality recordings it drifted into
English mid-sentence and produced unusable text (personal names redacted as `[NOME]`):

> "Go target, [NOME]. I am in contact with my advice and I saw the number of process,
> where the process does entrade, in which tribunal entrant."

Whisper, with `language="pt"` forced, transcribed the same audio correctly:

> "Boa tarde novamente, [NOME]. Eu estou em contacto com o meu advogado e eu preciso
> de saber o número do processo, onde o processo deu entrada, em que tribunal deu entrada."

This tracks the published FLEURS Portuguese WER (Whisper large-v3 3.65% vs Parakeet
v3 4.76%), but the real-world gap on noisy spontaneous speech is far wider than those
clean-read-speech numbers suggest.

### Text cleanup — `qwen3.5:9b` (Ollama, thinking disabled)

First screened `qwen3.5:4b`, `gemma3:4b` and `llama3.2:3b` on a real transcript:

| Model | Time | Notes |
|---|---|---|
| `qwen3.5:4b` (default) | 276 s | emitted 27k chars of reasoning — disqualified |
| `qwen3.5:4b` (`think=False`) | 10 s | best output: correct quoting, fixed `o quê`, kept register |
| `gemma3:4b` | 9 s | safe but passive — corrupted a proper noun into an unrelated word |
| `llama3.2:3b` | 7 s | hallucinated an inserted clause |

**Qwen3.5 reasons by default in Ollama**, which is slow and pointless for a cleanup
task, so the pipeline passes `think=False` on every call. With that flag it was both
the fastest and the most accurate of the three.

Qwen3.5 9B was then benchmarked against the 4B across a small set of real
recordings, scoring *content recall* — the fraction of distinct content words in
the raw transcript that survive into the refined text. This is the failure mode
that matters here: a small model silently dropping a clause while tidying prose.

| Model | Mean content recall | Total batch time |
|---|---|---|
| `qwen3.5:4b` | 0.9688 | 140 s |
| **`qwen3.5:9b`** | **0.9809** | 265 s |

The 9B preserves more of the original and handles verb tense better. On one
recording the raw ASR produced a present-perfect with inverted clitic word order;
the 4B "corrected" it into a simple past, changing the meaning, while the 9B fixed
only the word order and left the tense intact. For accuracy-sensitive material that
distinction is worth the extra two minutes, so the 9B is the default. Switch
`LLM_MODEL` in [config.py](config.py) to `qwen3.5:4b` if throughput matters more.

Caveat on the metric: content recall counts *distinct* words, so it does not detect
a dropped repetition. In one spot-check the 9B dropped a trailing repeated word that
the 4B kept. Neither model is a substitute for the human review step this pipeline
is designed to feed.

## What the refine pass actually fixes

It is not cosmetic. On one recording Whisper fell into a repetition-loop
hallucination, emitting the same four-word phrase 28 times at the end. The LLM
collapsed it to a single instance. It also recovered a garbled proper noun to its
correct spelling using surrounding context.

The prompt in [config.py](config.py) forbids summarising, inventing content and
translating, requires the speaker's register and Portuguese variant to be preserved,
and instructs the model to mark uncertain passages with `[?]` for a human reviewer.

Because a small model editing text is inherently lossy, **every Word document embeds
the unedited ASR output as an appendix**, so a reviewer can check any correction
against what was actually heard.

## Tracing a document back to its audio

Filenames are preserved end to end: `interview-2026-03-01.m4a` becomes
`output/docx/interview-2026-03-01.docx`. Each document also carries a provenance
header naming the source file, duration, original codec and sample rate, both model
names, and the generation timestamp. The same key indexes
`output/transcripts.json`.

## Usage

```bash
uv run python pipeline.py --input recordings.zip                       # run everything (resumes from cache)
uv run python pipeline.py --input recordings.zip --force               # ignore cache, redo all stages
uv run python pipeline.py --only transcribe refine                     # run a single stage (reuses the last input)
uv run python pipeline.py --input recordings.zip --only refine docx    # re-run cleanup and regenerate documents
uv run python pipeline.py --input recordings.zip --skip-refine         # raw ASR only, no LLM pass
```

`--input` accepts a zip archive of several recordings, or a single audio file. If
omitted, the pipeline looks for exactly one `.zip` in the project root. Stages are
`extract`, `normalize`, `transcribe`, `refine`, `docx`. Progress is checkpointed to
`output/transcripts.json` after every file, so an interrupted run resumes where it
stopped. To try a different SLM, change `LLM_MODEL` in [config.py](config.py) and
run `--only refine docx --force`.

## Audio handling

Recordings commonly arrive in a mix of formats and sample rates — the normalize
stage handles that automatically. It probes each file with `ffprobe` and converts
anything that is not already 16 kHz mono PCM to that format, downmixing stereo as
needed; files already in the target format are copied rather than re-encoded.
macOS `__MACOSX/._*` resource-fork entries are skipped during zip extraction —
they share the real files' extensions but contain no audio.

Recognised extensions are listed in `AUDIO_EXTENSIONS` in [config.py](config.py);
add to that set if other formats show up.

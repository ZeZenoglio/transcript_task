# Transcript Tool

A small, local speech-to-text utility. Give it one audio recording, or a zip
of many, and it produces a reviewed Word document per recording — corrected,
punctuated, titled, and traceable back to its source file. Everything runs
on-device via [Ollama](https://ollama.com) and [MLX](https://github.com/ml-explore/mlx)
— no audio or text leaves the machine.

```
input (one file, or a zip of many)
  → extract → normalize (ffmpeg) → transcribe (Whisper) → refine (Ollama)
  → summarize (Ollama, structured JSON) → .docx
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
uv run python main.py --input recordings.zip     # a zip of several files
uv run python main.py --input interview.m4a       # or just one file
```

Results land in:

| Path | Contents |
|---|---|
| `output/transcripts.json` | every transcript — raw, refined, and its structured summary — with timings and segment timestamps |
| `output/docx/` | one Word document per audio file, named after its generated title |
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
`llm_model` in [settings.py](src/transcript_task/settings.py) to `qwen3.5:4b` if throughput matters more.

Caveat on the metric: content recall counts *distinct* words, so it does not detect
a dropped repetition. In one spot-check the 9B dropped a trailing repeated word that
the 4B kept. Neither model is a substitute for the human review step this pipeline
is designed to feed.

## What the refine pass actually fixes

It is not cosmetic. On one recording Whisper fell into a repetition-loop
hallucination, emitting the same four-word phrase 28 times at the end. The LLM
collapsed it to a single instance. It also recovered a garbled proper noun to its
correct spelling using surrounding context.

The prompt in [prompts.py](src/transcript_task/prompts.py) forbids summarising, inventing content and
translating, requires the speaker's register and Portuguese variant to be preserved,
and instructs the model to mark uncertain passages with `[?]` for a human reviewer.

Because a small model editing text is inherently lossy, **every Word document embeds
the unedited ASR output as an appendix**, so a reviewer can check any correction
against what was actually heard. It's also not risk-free in the other direction: the
evaluation harness's benchmarks found refine measurably raising WER on both FLEURS
and Common Voice test data (see "Evaluation harness" below), and a production
safety net now discards a refine attempt outright if it drifts too far from the
raw transcript — that section covers both the quantitative evidence and the guard.

## Titles and summaries

A final LLM pass reads *both* the raw and refined transcripts and returns a
structured summary: a title, a 3–6 sentence description, 3–8 topic keywords, an
estimated speaker count, the detected Portuguese variant, a `sensitivity` flag, and
the model's own `confidence`. Giving it both transcript versions lets it use the
raw text's disfluencies as evidence for speaker count and register, and flag where
cleanup may have changed the meaning.

The summary is returned as JSON constrained by Ollama's structured-output mode (a
pydantic model's schema passed as `format=`), so malformed output is rare — but it's
validated on arrival regardless, retried once with the validation error appended to
the prompt if it fails, and replaced with a low-confidence stub (never a crash) if it
fails twice. `sensitivity: medium/high` drives a warning banner at the top of the
document; `confidence: low` adds a note asking the reviewer to double-check the title
and description by hand.

Output language is a config option (`summary_language: "pt" | "en"` in
[settings.py](src/transcript_task/settings.py), or `TRANSCRIPT_SUMMARY_LANGUAGE`) —
the title/description can either match the recording or be English for easier
scanning. The transcript itself is **never** translated; this only picks which
summary prompt variant runs.

## Privacy: filenames and metadata never carry a name

A title or description is LLM-authored text — it can end up in a filename, or in a
document's File Properties, either of which can travel more casually than the full
`.docx` (a folder listing, an OS search index, an email preview). Two layers guard
against a name or other identifying detail leaking into that surface:

1. **The summarize prompt is told not to.** Both the `pt` and `en` templates in
   [prompts.py](src/transcript_task/prompts.py) explicitly instruct the model to
   refer to people by role ("a caller", "the client") rather than by name in
   `title`/`description`/`topics`, and never to touch this for any other field.
2. **A deterministic backstop, because a prompt instruction is a soft control.**
   [anonymize.py](src/transcript_task/anonymize.py) runs a small local Portuguese
   NER model ([spaCy](https://spacy.io) `pt_core_news_sm`, ~12 MB, no GPU) plus regex
   patterns for emails/phone numbers/ID numbers over the summary before it's used
   for a filename or written into docx metadata, replacing anything it catches with
   `[nome]`/`[contacto]`. It's tuned to over-redact rather than under-redact — a
   real place name occasionally getting swept up costs nothing on a short title; a
   real person's name slipping through is the failure this exists to prevent.
   Toggle with `anonymize_metadata` in settings (on by default).

**This never touches the transcript itself, or the visible summary in the document
body.** By the time someone has the `.docx` open, the full transcript with real
names is right there in the "Transcrição revista" section — redacting the abstract
above it would protect nothing and just look broken. Redaction is scoped
specifically to what leaves the document body: the filename and the OOXML core
properties (title/subject/keywords).

## Tracing a document back to its audio

Every recording gets a stable, random `transcript_id` (an 8-character hex string)
the first time it's processed, and that id never changes even if you rerun the
pipeline with `--force`. The document filename is `<transcript_id>_<slug>.docx`,
where the slug is the (anonymized) title: `interview-2026-03-01.m4a` might become
`output/docx/a1b2c3d4_weekly-status-update.docx`. The id is what guarantees two
recordings never collide, even if they summarize to the same title or a summary is
missing entirely (in which case the filename is just `<transcript_id>.docx`).

The original filename doesn't appear in the new name, but traceability doesn't
depend on that: it's in the document's own provenance header (source file,
duration, codec, sample rate, both model names, generation timestamp), and in the
`output/transcripts.json` record indexed by the same `transcript_id`. This is
deliberately the same id a future API (see [PLAN.md](PLAN.md), Phase 7) would use
as a job id and a database primary key — one id names a file, a JSON record, and
eventually a database row for the same recording.

## Usage

```bash
uv run python main.py --input recordings.zip                          # run everything (resumes from cache)
uv run python main.py --input recordings.zip --force                  # ignore cache, redo all stages
uv run python main.py --only transcribe refine                        # run a single stage (reuses the last input)
uv run python main.py --input recordings.zip --only summarize docx    # regenerate titles + documents only
uv run python main.py --input recordings.zip --skip-refine            # raw ASR only, no LLM pass
```

`--input` accepts a zip archive of several recordings, or a single audio file. If
omitted, the pipeline looks for exactly one `.zip` in the project root. Stages are
`extract`, `normalize`, `transcribe`, `refine`, `summarize`, `docx`. Progress is
checkpointed to `output/transcripts.json` after every file, so an interrupted run
resumes where it stopped. To try a different SLM, change `llm_model` in
[settings.py](src/transcript_task/settings.py) (or set `TRANSCRIPT_LLM_MODEL`) and
run `--only refine summarize docx --force`.

`main.py` is a thin entry point; `uv run python -m transcript_task.pipeline --input ...`
does exactly the same thing.

## Project layout

```
src/transcript_task/
  settings.py      # pydantic-settings model — every config value, env-overridable
  prompts.py       # LLM prompt templates, versioned by id
  audio.py         # ffprobe/ffmpeg wrappers (probe, normalize)
  asr.py           # speech-to-text behind a Transcriber protocol
  refine.py        # LLM cleanup behind a ChatModel protocol
  summarize.py     # title/description generation, schema + slugify + filename logic
  anonymize.py     # PII safety net (NER + regex) for filenames/docx metadata
  text_compare.py  # normalize_pt + content_recall/length_ratio — used both as
                   # benchmark metrics (eval/) and a live refine-quality guard (pipeline.py)
  docx_writer.py   # Word document generation
  pipeline.py      # stage orchestration + CLI, transcript_id assignment
  eval/            # evaluation harness (Phase 6) — see below
main.py            # entry point
scripts/           # fetch_dataset.py, fetch_common_voice.py, build_fixtures.py, benchmark.py
tests/             # pytest suite — see the Testing section below
```

`asr.py` and `refine.py` expose their model calls behind small `Protocol`
interfaces rather than the pipeline calling `mlx_whisper`/`ollama` directly. That
is what lets tests inject a fake model and lets the eval harness swap ASR/LLM
models without touching `pipeline.py`. `summarize.py` reuses the same `ChatModel`
protocol as `refine.py`.

## Testing

```bash
uv run pytest              # fast unit tests, no network or models required
uv run pytest -m integration   # also exercises the real local Ollama model
```

Unit tests fake the LLM client (`tests/fakes.py`) so schema validation, the
retry-then-fallback path, and filename generation all run in milliseconds with
no model calls. `anonymize.py`'s tests use the real (small, local) spaCy model
rather than a fake, since its actual entity-recognition behaviour is the thing
under test. `audio.py`'s tests run against four real, diverse, public-domain
speech clips (see below) rather than only synthetic tones, precisely because
real content has repeatedly caught bugs synthetic fixtures didn't (see the
next section). Two `integration`-marked tests call real local models and skip
themselves if Ollama isn't reachable: one exercises `summarize_transcript`
directly, the other runs a full clip through the real ASR + LLM pipeline via
the eval harness below and checks the resulting WER against ground truth.

## Benchmark dataset

For evaluation and for real-audio test fixtures, this project uses
[Google FLEURS](https://huggingface.co/datasets/google/fleurs) (`pt_br`, CC BY
4.0) rather than the private recordings it was originally developed against
(see [docs/research-stt-landscape.md](docs/research-stt-landscape.md) and
[PLAN.md](PLAN.md) for why, and its honest limitation: FLEURS is clean read
speech, not the noisy conversational audio this tool actually targets, so
scores against it are a *relative* regression signal, not an absolute quality
claim).

```bash
uv run python scripts/fetch_dataset.py          # full test split, ~277 MB, 919 clips -> data/fleurs_pt/ (gitignored)
uv run python scripts/build_fixtures.py         # picks 4 clips, re-encodes each to a different format -> tests/fixtures/
```

`data/` is gitignored and regenerated on demand; the small, diversified subset
in `tests/fixtures/` (4 clips, ~660 KB, spanning ~4s to ~37s, one each in
`.wav`/`.mp3`/`.m4a`/`.opus`) is committed so tests and CI don't need the full
download. See `tests/fixtures/NOTICE.md` for attribution.

**Two real bugs found and fixed while building this**, on top of the docx and
NER bugs from Phases 3 and 3b -- the fourth time in a row that testing against
real data, not just plausible-looking fakes, has caught something a
synthetic fixture wouldn't have:
- The dataset's `num_samples` field, used to compute duration without
  decoding audio, disagreed with the real ffprobe-measured duration on ~60%
  of a random sample -- by several seconds in some cases. Fixed by measuring
  duration from the file actually written to disk instead of trusting a
  metadata field.
- FLEURS' `id` field is a shared sentence/prompt id (multiple speakers read
  the same sentence), not a unique row id. Keying output filenames by it was
  silently overwriting one recording with another: of 919 rows, only 349 had
  a unique `id`, so 570 clips were being lost to filename collisions with no
  error raised anywhere. Fixed by keying filenames on each row's position in
  the split instead, which cannot collide by construction.

## Evaluation harness

A pragmatic, repeatable way to know whether a model or prompt change helped,
built on `scripts/benchmark.py` and `src/transcript_task/eval/`:

```bash
uv run python scripts/benchmark.py run --tier smoke --tag baseline    # 4 committed clips, no download
uv run python scripts/benchmark.py run --tier quick --tag baseline    # stratified n=30, ~5 min
uv run python scripts/benchmark.py run --tier full  --tag release-1.0 # the whole 919-clip split

# swap models via the same env vars Settings always honors
TRANSCRIPT_LLM_MODEL=qwen3.5:4b uv run python scripts/benchmark.py run --tier quick --tag candidate

uv run python scripts/benchmark.py compare baseline candidate   # exits 1 on regression -- what CI consumes
```

**Metrics**, all pure functions in `eval/metrics.py`, independent of MLflow:
- **WER/CER** (`jiwer`) after a Portuguese-aware normalizer (`text_compare.py`)
  — Whisper's bundled one is English-only. It casefolds, strips punctuation, and
  expands digit runs to number words (`12` → `doze`) via `num2words`, since
  Whisper sometimes writes digits where FLEURS' ground truth spells them out —
  a real, spurious source of WER that has nothing to do with transcription
  quality.
- **SemDist**: 1 − cosine similarity between multilingual sentence embeddings
  (`paraphrase-multilingual-MiniLM-L12-v2`), behind the same `Embedder` protocol
  pattern as `Transcriber`/`ChatModel`.
- **Refine, scored against the same ground truth, before and after**: this is
  the trick that turns an unmeasurable stage into a measurable one. If cleanup
  helps, WER drops; if the LLM paraphrases, it rises.
- **Summary stage**: deterministic only (schema-validity rate, retry/fallback
  rate, field-length and topic-count conformance, filename-slug uniqueness,
  language-variant agreement, description-vs-transcript embedding distance).
  No LLM-as-judge scoring anywhere in this harness.
- **Performance**: realtime factor, per-stage latency (p50/p95), tokens/s for
  the LLM stages, peak RSS.

**Tiers** (`eval/tiers.py`): `smoke` (the 4 committed fixtures), `quick`
(duration-stratified random n=30, fixed seed — stratified so the sample isn't
all short easy clips), `full` (the whole split). `--dataset {fleurs,common_voice}`
picks which corpus to run against (see below for why there's a second one).

**MLflow**: local file-backed tracking (`mlruns/`, `mlflow ui`), one run per
benchmark. Every run also writes `benchmarks/<tag>/results.json` + `table.md`
independent of MLflow — metric computation never depends on the tracking
sink, so ripping MLflow out would touch one file (`eval/mlflow_sink.py`).
A local LLM writes a short, non-scoring markdown **interpretation** of the
metrics table afterward (`eval/interpretation.py`) — it reads only the
numbers already computed, never the audio or transcripts, and produces no
score of its own.

**What the first real smoke run found:** across the 4 committed FLEURS clips,
raw ASR WER averaged 0.034, but WER *after* the refine stage averaged 0.108 —
refine consistently made the transcript *further* from ground truth, not
closer. This isn't a harness bug; it's a real, honest limitation the
benchmark is supposed to surface: FLEURS is clean, unambiguous read speech,
so ASR output already has almost nothing to clean up, while refine's
paraphrasing risk (rewording a correct sentence into a different correct
sentence) still applies. It's exactly the trade-off flagged in
[docs/research-stt-landscape.md](docs/research-stt-landscape.md) and PLAN.md:
FLEURS is a *relative* regression signal, not a verdict on whether refine
helps on the noisier, disfluent conversational audio the tool actually
targets, where refine has real disfluencies and hesitations to remove that
FLEURS' clips simply don't contain.

**Found and fixed while building this:** MLflow 3.x put its plain filesystem
tracking backend (`file:./mlruns`) into maintenance mode and now refuses it
unless `MLFLOW_ALLOW_FILE_STORE` is set, nudging new projects toward a
SQLite/DB-backed store. That collided with the local-file-backed tracking
decided on for this project; rather than revisit the decision for a
single-user local tool, `eval/mlflow_sink.py` sets the opt-out flag so
`mlruns/` + `mlflow ui` still work exactly as intended.

**A second, more serious bug, found while proving the regression gate works
on real data:** swapping to a much smaller model (`llama3.2:1b` in place of
`qwen3.5:9b`) to demonstrate `compare`'s non-zero exit — the plan's actual exit
criterion for this phase — hit a real runaway generation loop: one clip's
refine call generated 163,840 tokens over 51 minutes instead of stopping
naturally, because nothing capped output length. `settings.llm_options` set
`temperature` and `num_ctx` but never a token-generation ceiling, so a model
that doesn't reliably stop had nothing bounding it short of exhausting its
context. This was a real gap in the *production* pipeline, not just the eval
harness — anyone swapping `llm_model` to a different model could hit the same
thing. Fixed by adding `llm_num_predict` (default 8192) to `Settings`. With
that real data captured, `compare` shows exactly what the plan asked for:

```
$ uv run python scripts/benchmark.py compare smoke-test-1 smoke-worse-llm
| metric | baseline | candidate | delta | regressed |
|---|---|---|---|---|
| wer_raw | 0.0342 | 0.0342 | +0.0000 |  |
| wer_refined | 0.1084 | 1129.1570 | +1129.0486 | YES |
$ echo $?
1
```

`wer_raw` is unchanged (only the LLM was swapped, not the ASR model) while
every refine-stage metric is flagged — the regression gate correctly points at
which stage broke, not just that something did.

### A noisier tier: Common Voice pt

The FLEURS result above only says something about refine on *clean, studio-
quality read speech* — FLEURS was always flagged as unrepresentative of this
tool's noisy-conversational target domain (see "Benchmark dataset" above).
`scripts/fetch_common_voice.py` fetches a second dataset,
[Common Voice](https://commonvoice.mozilla.org/) pt (`CC0-1.0`), via
[`fsicoli/common_voice_17_0`](https://huggingface.co/datasets/fsicoli/common_voice_17_0)
— a community mirror, not the official `mozilla-foundation` org's repo,
because that one ships a Python loading script and `datasets` 5.x has
**removed loading-script support entirely** (confirmed by trying it, not
assumed). Contributors record themselves on whatever device they have, so
these clips carry real background noise and mic-quality variance FLEURS'
professional narration doesn't.

```bash
uv run python scripts/fetch_common_voice.py                                  # full test split, ~305 MB, 9,467 clips
uv run python scripts/benchmark.py run --dataset common_voice --tier quick --tag noisy-baseline
```

A second committed smoke set, `tests/fixtures_noisy/` (4 clips, 0.9s–10.6s,
mp3/wav/m4a/opus), was built the same way as the FLEURS one — see
`tests/fixtures_noisy/NOTICE.md` for attribution.

**What real Common Voice data found, immediately, before any tuning run:**
2 of the 4 committed smoke clips came back with `wer_raw = 1.0` — total ASR
misses, not refine problems. A 0.9s clip of the single word "apuração" was
transcribed as "Obrigada." (a known Whisper hallucination on very short/quiet
audio); a 10.6s clip came back as "Me задissinou canubar." — Cyrillic
characters mixed into Portuguese-looking fragments, reproduced identically
from the untouched source file, so not an artifact of this project's own
format conversion. FLEURS' clean narration never surfaced failures like this.

**The actual payoff — a real `--tier quick` run (n=30, stratified, seed=42):**
`wer_raw` mean 0.0852 (vs. FLEURS' 0.0342, confirming Common Voice is
genuinely harder), `wer_refined` mean 0.1360. Per clip, refine **improved**
WER on 1 of 30, **worsened** it on 7, left 22 unchanged. This is the answer
the FLEURS-only result couldn't give: refine is net-negative on WER even on
noisier, more varied real audio — the earlier FLEURS finding wasn't just an
artifact of testing on unnaturally perfect speech. It still isn't the final
word on whether refine is worth keeping (WER doesn't score punctuation or
readability, most of refine's actual job, and neither dataset contains truly
spontaneous disfluent speech), but it does rule out "FLEURS was just too
clean" as the explanation.

### A production safety net for refine

`content_recall` and `length_ratio` (ground-truth-free, originally built as
benchmark metrics above) are now also a **live check on every refine call**,
not just something the eval harness measures after the fact. They live in
`text_compare.py`, not `eval/metrics.py`, specifically so `pipeline.py` can
use them without importing jiwer/sentence-transformers/mlflow — `eval/metrics.py`
re-exports them so nothing else had to change.

If a refine call's output fails either check, `pipeline.stage_refine`
discards it and falls back to the raw transcript — which `summarize`/`docx`
already do automatically for a missing `refined_transcript`. This fails
**soft and visibly**: the rejected text is kept (not thrown away), the docx
shows a bold warning banner naming the reason and the numbers, and the
rejected attempt is included as an appendix so a reviewer can check the
guard's call rather than trust it blindly.

Verified against the real incident that motivated it, not just synthetic
tests: re-running the exact clip that caused the 51-minute/163,840-token
runaway above, through the real (now-guarded) pipeline, reproduces the slow
generation again — a real, repeatable failure mode, not a one-off — but
bounded this time to 521.6 seconds (the `llm_num_predict` cap) and correctly
discarded (26,398 characters of runaway text, `content_recall=0.273`) rather
than shipped.

### Tuning the refine prompt, measured against both benchmarks

Diagnosing the WER erosion above meant inspecting actual `raw` vs `refined`
text, not just the aggregate metric. The pattern: v1 volunteers small,
unforced rewrites of text that's already correct — `"como uma rota"` →
`"como numa rota"` (no rule justifies this), `"...voltando ao mar"` → `"que
volta ao mar"` (a stylistic rewrite of an already-grammatical sentence), and
one actively **wrong** "agreement fix" that guessed the wrong noun for an
ambiguous adjective and flipped a correct sentence into an incorrect one.

`refine-pt-v2` ([prompts.py](src/transcript_task/prompts.py)) adds an explicit
minimal-edit principle ahead of every other rule — "if it's already correct,
leave it exactly as-is, even if you can imagine a more elegant phrasing" —
and narrows the agreement-fixing rule to defer instead of guess when a
term's referent is ambiguous. `Settings.refine_prompt_id`
(`TRANSCRIPT_REFINE_PROMPT_ID`) makes prompt variants an A/B switch, not a
code change.

**Measured on identical clips (same seed) against both benchmarks:**

| | Common Voice (n=30) | FLEURS (n=30) |
|---|---|---|
| wer_refined: v1 → v2 | 0.1360 → 0.1013 (−25%) | 0.0489 → 0.0289 (−41%) |
| regressions: v1 → v2 | 7/30 → 2/30 | 13/30 → 6/30 |

`refine-pt-v2` is now the default. The improvement holds across two
independently-sourced datasets, not just one run's noise — though a genuine
noise floor did turn up in the process: `wer_raw` moved by +0.0014 between
the two FLEURS runs on the identical 30 clips, because mlx-whisper's own
output for one clip differed slightly between the two separate process
runs. That's about 15x smaller than the refine-stage improvement, so it
doesn't change the conclusion, but it's real ASR non-determinism worth
knowing about before reading too much into a small delta elsewhere.

The two regressions still remaining under v2 are a different, more specific
failure mode, not "still too aggressive": one is the model converting a
Brazilian gerund construction (`"estão trabalhando"`) into the European
periphrastic form (`"estão a trabalhar"`) — a genuine bug against the
explicit variant-preservation rule, and a concrete target for a future
`refine-pt-v3`. The other is a word-enumeration-style Common Voice prompt
("impugnar, imunidade, regalias...") that the punctuation-naturalization
rule reasonably, but wrongly, "corrected" into fluent prose — a collision
between a legitimate rule and unusual source content, not a wording defect.

Neither FLEURS nor Common Voice contains genuinely spontaneous/disfluent
speech, refine's actual target domain — this tuning is a real, verified
improvement on what these two benchmarks can measure, not a substitute for
eventually testing against real disfluent conversational audio.

## Audio handling

Recordings commonly arrive in a mix of formats and sample rates — the normalize
stage handles that automatically. It probes each file with `ffprobe` and converts
anything that is not already 16 kHz mono **16-bit PCM** (`pcm_s16le`) to that
format, downmixing stereo as needed; files already in the exact target format
are copied rather than re-encoded. That "exact" matters: `.wav` can just as
easily hold 32-bit float PCM as 16-bit (FLEURS' own files do — see the
benchmark dataset section above), and a version of this check that only
compared sample rate and channel count would silently skip re-encoding a
float32 file. Caught by testing against a real FLEURS clip, not a synthetic
one, and now covered by a regression test in `tests/test_audio.py`.
macOS `__MACOSX/._*` resource-fork entries are skipped during zip extraction —
they share the real files' extensions but contain no audio.

Recognised extensions are listed in `audio_extensions` in [settings.py](src/transcript_task/settings.py);
add to that set if other formats show up.

# Implementation Plan — transcript_task

From a working local script to an evaluated, served, tested product.

**Status:** Phases 0–5 complete (2026-09-24, on `dev`). Phases 6-12 pending. All open
decisions answered — see *Decisions made* at the end.

**Scope correction (2026-09-24):** the tool generalises to a plain speech-to-text
utility — one recording or a zip of many in, reviewed Word transcripts out. It is
not built around, or documented as, any particular recording scenario. `pipeline.py`
now takes `--input <zip-or-audio-file>` instead of a hardcoded archive name; the
README and this plan were swept for scenario-specific framing and examples.

---

## Current state

| | |
|---|---|
| Works | `extract → normalize → transcribe → refine → summarize → docx`, JSON checkpointing, stable `transcript_id` per recording; validated end-to-end on FLEURS clips (no private recordings remain on this machine as of Phase 5) |
| Models | `mlx-whisper` large-v3-turbo (ASR) · `qwen3.5:9b` via Ollama, `think=False` (refine + summarize) · `spacy` `pt_core_news_sm` (PII safety net) |
| Measured | 13:35 audio → 77 s ASR + 202 s refine + ~110s summarize on an M4 base (measured before Phase 5's deletion, against the original private recordings) |
| Tests | 80 unit tests (`uv run pytest`) + 1 integration test against real Ollama |
| Benchmark data | FLEURS pt_br test split (919 clips) fetchable via `scripts/fetch_dataset.py`; 4-clip diversified fixture committed under `tests/fixtures/` |
| Missing | linting, API, eval harness, CI, frontend, logging, persistence |
| Repo | pushed, public, `origin/master` + `origin/dev`, `gh` not authenticated locally |

As of Phase 2, the code is a proper `src/` package (`src/transcript_task/`) with a
pydantic `Settings` model passed explicitly into each stage, and the ASR/LLM calls
behind `Protocol` interfaces. See Phase 2 below for what moved where.

---

## Phase 0 — Safety and repo hygiene ✅ done

Completed 2026-09-24. The repo is public-bound, so this went further than planned.

1. **`.gitignore` rewritten**, secrets block first: `.env`/`.env.*` (with an
   `.env.example` exception), keys/pems, Streamlit secrets; all private source material
   (`Arquivo.zip`, `*.zip`, `tmp/`, `output/`, `data/`, every audio extension, `*.docx`);
   caches, `mlruns/`, `*.db`, `.DS_Store`, `.claude/settings.local.json`. Committed test
   fixtures under `tests/fixtures/` are re-included at the bottom.
2. **`.env.example`** added with key names and guidance only.
3. **Personal data scrubbed from `README.md`** — this was not in the original plan and
   turned out to matter most. The README quoted the real recordings verbatim, including
   a named adult and a child's first name, in the context of divorce proceedings. Names
   are now `[NOME]`; the name-recovery and verb-tense examples were rewritten to describe
   the linguistic phenomenon without reproducing the content. Every technical claim
   survives; none of the personal detail does.
4. **Privacy note** added to the README.
5. **Verified by dry-run**: `git add -A` would stage exactly 10 files — `.env.example`,
   `.gitignore`, `.python-version`, `PLAN.md`, `README.md`, `config.py`, `main.py`,
   `pipeline.py`, `pyproject.toml`, `uv.lock`. Every sensitive path confirmed ignored
   via `git check-ignore`, including `data/` and `mlruns/` tested with real files.

Deferred to the user by agreement: the first commit, repo creation, push, and the
branch-name decision.

**Exit:** met. No secrets, audio, transcripts or personal names are stageable.

---

## Phase 1 — Research: local vs. public STT, quality vs. speed ✅ done

Completed 2026-09-24, on the `dev` branch. Written up in
[`docs/research-stt-landscape.md`](docs/research-stt-landscape.md). No code changed.

- Cloud/API baselines to compare against: OpenAI `whisper-1` / `gpt-4o-transcribe`,
  Deepgram Nova-3, AssemblyAI Universal, Azure Speech, Google Chirp 2, ElevenLabs Scribe.
- Axes: Portuguese WER (published + caveats), realtime factor, cost per audio-hour,
  privacy/data-residency, offline capability, diarisation and timestamp support.
- Anchor everything to the numbers this repo already measured (~15x realtime on M4,
  FLEURS pt WER 3.65% for large-v3) so the comparison is like-for-like.
- Explicit section on **what local buys you here**: recordings may contain private
  or sensitive content of any kind; zero-egress is a feature, not just a cost saving.

**Findings:** the top 5 cloud providers (OpenAI, Deepgram, AssemblyAI, Azure, Google,
ElevenLabs) now sit within ~1–2 WER points of each other and of local Whisper on
published benchmarks — the competitive edge has moved to streaming latency,
diarization and domain-vocabulary biasing, not raw accuracy. Cost is negligible at
this project's actual volume (cents for the whole test set either way). The decisive
axis is privacy: audio containing speech is personal data under GDPR, and every cloud
option requires a signed DPA and a residency/transfer decision that running locally
avoids entirely. **No model swap recommended** — staying local is the right call for
this project's threat model and scale, revisit only if real-time streaming, volume
beyond one machine, or production-grade diarization become requirements.

**Deliverable:** comparison table + a recommendation paragraph on when to reach for a
cloud API instead.
**Exit:** doc reviewed; any model swap it suggests is logged as a follow-up, not done
inline.

---

## Phase 2 — Restructure into a package ✅ done

Completed 2026-09-24, on `dev`. Built as planned; `summarize.py`, `db.py` and
`logging_conf.py` are deliberately not created yet — they belong to Phases 3 and 7
and would be empty stubs today.

```
src/transcript_task/
  __init__.py
  settings.py        # pydantic-settings model; env-overridable via TRANSCRIPT_* / .env
  prompts.py         # PromptTemplate dataclass, versioned by id (refine-pt-v1 today)
  audio.py           # ffprobe/ffmpeg wrappers: probe(), convert_to_target()
  asr.py             # Transcriber protocol + MlxWhisperTranscriber
  refine.py          # ChatModel protocol + OllamaChatModel + refine_transcript()
  docx_writer.py     # write_docx() — the document-building logic, unchanged output
  pipeline.py        # stage orchestration + CLI (unchanged interface, new internals)
```

`config.py` and the root `pipeline.py` are deleted; `main.py` now imports
`transcript_task.pipeline.main`. `--input` behaviour from the earlier generalisation
pass carried over unchanged.

Key changes, as planned:
- Constants → a `Settings` pydantic model, instantiated once in `main()` and
  **passed into** every stage function, not imported as globals.
- `asr.Transcriber` and `refine.ChatModel` are `Protocol`s; `stage_transcribe`/
  `stage_refine` accept an optional instance of each, defaulting to the real
  MLX/Ollama implementation. Phase 6's eval harness and Phase 10's tests both
  depend on this — it's what makes a fake model injectable.
- `pyproject.toml` gained a `hatchling` build backend and `[tool.hatch.build.targets.wheel]`
  so `uv sync` installs the package in editable mode. `parakeet-mlx` was dropped from
  dependencies — it was only ever used for the one-off benchmark documented in the
  README, and nothing in the shipped pipeline imports it.

**Exit:** met, with one honest caveat. `output/transcripts.json` from before the
restructure loaded without changes, and a cached re-run (`--only docx`, everything
else already done) reproduced the state file exactly, field for field. A full
`--force` re-run reused identical arguments to `mlx_whisper.transcribe` and matched
on 7 of 9 files exactly; the two longest recordings (247s and 248s) came back with
minor wording differences. This traces to `mlx-whisper`/Metal itself, not the
refactor — verified by calling the new `MlxWhisperTranscriber` twice in the same
process on the same file (byte-identical), then noting the divergence only appears
*across separate process runs* on long, hard-to-transcribe audio, consistent with
Whisper's temperature-fallback decoding being sensitive to Metal's non-deterministic
kernel scheduling on marginal/low-confidence stretches. **True byte-for-byte ASR
reproduction across runs was never actually available before this refactor either**
— this just made it visible. Worth keeping in mind for Phase 6: the eval harness
should tolerate small per-run WER noise on long audio rather than expect exact
repeatability.

---

## Phase 3 — Summarisation stage (title + description) ✅ done

Completed 2026-09-24, on `dev`. Built as planned: new `summarize` stage between
`refine` and `docx`, in `src/transcript_task/summarize.py`.

New final LLM pass receiving **both** the raw and refined transcripts, emitting
structured output.

**Schema** (pydantic → JSON Schema → Ollama `format=`, which enforces it at decode time):

```python
class TranscriptSummary(BaseModel):
    title: str                    # <= 80 chars, filename-safe, no trailing period
    description: str              # 3-6 sentences, what the recording is about
    topics: list[str]             # 3-8 keywords
    speakers_detected: int        # best-effort count
    language_variant: Literal["pt-PT", "pt-BR", "unknown"]
    sensitivity: Literal["low", "medium", "high"]   # flags likely-private content
    confidence: Literal["low", "medium", "high"]    # model's own confidence
```

**Output language is configurable** (`settings.summary_language`: `pt` | `en`), so the
title and description can either match the recording or be English for easier scanning.
The transcript itself is never translated — this affects the summary fields only. Both
values get a prompt variant and a test.

- Giving the model *both* versions lets it flag where cleanup may have changed meaning,
  and gives it the raw text's disfluencies as evidence about speaker count and register.
- Ollama structured outputs are constrained decoding, so schema violations should be
  near-zero — but **validate anyway** and retry once with the validation error appended,
  then fall back to a `confidence="low"` stub rather than failing the file.
- `sensitivity` exists because any recording can turn out to contain private material;
  it drives a warning banner in the docx and the UI when it does.

**Wiring:** new `summarize` stage between `refine` and `docx`; persisted into
`transcripts.json`; surfaced in the docx as title + an abstract block; used to generate
a **human-readable filename** (`<slugified-title>__<original-stem>.docx`) so documents
are both descriptive *and* still traceable to their source audio.

**Tests:** schema validation, retry-on-invalid path, fallback path, slug collision
handling, prompt-template rendering — all against a fake LLM client, plus one
`@pytest.mark.integration` test hitting real Ollama.

**A real bug, found by running against real data rather than only fakes:**
OOXML core document properties (`subject`, `keywords`, `title`) have a hard
255-unicode-character limit that `python-docx` enforces by raising `ValueError`.
The unit tests all passed -- they don't write real `.docx` files with long LLM
descriptions -- but the first end-to-end run against actual recordings crashed on
the very first file, since a 3-6 sentence description routinely exceeds 255 chars.
Fixed with a `_truncated()` helper applied to every core-property write in
`docx_writer.py`; the full untruncated description still appears in the document
body. Added regression tests in `tests/test_docx_writer.py` that write a real
`.docx` with a >255-char description and assert it doesn't raise. Logged here as a
reminder for Phase 10: fakes-only unit tests are necessary but not sufficient --
this class of bug only shows up when a real file format's real constraints meet
real (LLM-length) content.

**Exit:** met. All 9 files got a valid summary with no fallbacks needed (Ollama's
structured-output mode held up under real use); 33 unit tests green
(`uv run pytest`), plus the integration test passing against real Ollama. Sample
titles produced: recordings were correctly identified as sensitivity=high or
=medium in 8 of 9 cases, matching what a human reviewer would flag by hand.

**Follow-up from post-Phase-3 review (2026-09-24):** the truncation helper added
for the 255-char core-property bug now cuts on a word boundary instead of
mid-word (harmless polish -- it only ever touched invisible metadata, never the
visible transcript). Two real design gaps were raised and are addressed below:
test/eval fixtures need real audio, not just fakes (folded into Phase 4 and
Phase 10), and filenames/summaries need a stable ID and a privacy pass of their
own (new Phase 3b).

---

## Phase 3b — Stable transcript IDs & PII-aware summaries ✅ done

Completed 2026-09-24, on `dev`. Built as planned, with both open calls resolved
(see "Decisions made" #8-9) and one refinement found while testing against real
recordings.

Raised in review, not in the original plan: filenames should carry a stable
identifier rather than being a slug tied only to a title, and a title/description
generated by an LLM can itself leak a name or other identifying detail even when
the underlying transcript is never redacted (nor should it be -- the reviewer's
own recording stays intact; this is only about what a *summary* echoes into a
filename or document metadata that might travel more casually than the full doc).

### Stable transcript IDs

- A `transcript_id` is assigned the first time a source file's state entry is
  created (in practice, `stage_normalize` -- `stage_extract` doesn't touch
  `state` at all, so "at extract time" from the original review discussion
  became "at first touch"), and is **stable across `--force` re-runs** via
  `dict.setdefault`: reprocessing a recording doesn't give it a new identity,
  only a new result. Stored as `state["items"][key]["transcript_id"]`.
  `load_state()` also migrates any pre-existing item missing one (legacy state
  files, or anything from before this phase), so every downstream consumer can
  rely on it being present unconditionally.
- This is deliberately the *same* identifier Phase 7's API will expose as `job_id`
  and use as the SQLite `jobs` primary key -- introducing it now avoids retrofitting
  every downstream consumer later.
- **Format, decided (#8): 8-character random hex** (`secrets.token_hex(4)`) --
  short, filesystem-friendly, effectively collision-free at this tool's personal
  scale (2^32 space). Not time-sortable by construction; Phase 7's SQLite row will
  carry a real `created_at` column for that.
- **Filename is** `<transcript_id>_<slug>.docx` (or `<transcript_id>.docx`
  with no summary yet), replacing `<slug>__<original-stem>.docx` from Phase 3.
  **Decided (#9): the original stem is dropped from the filename**, staying
  traceable through the docx's own provenance header, the `transcripts.json`
  record, and (Phase 7) the SQLite row instead.
- `docx_filename()`'s signature changed to `(transcript_id, summary, *, language,
  anonymize=True)`; `stage_docx` passes `item["transcript_id"]`. Phase 3's filename
  collision tests were rewritten against the new scheme, not dropped -- the
  underlying guarantee (two recordings never collide) still holds, now via the id
  rather than the stem.

### PII-aware summaries

Two layers, not one -- a prompt instruction alone is a soft control an LLM can
ignore under real content pressure:

1. **Prompt-level:** both summarize templates bumped to v2 (`summarize-pt-v2`/
   `summarize-en-v2` in `prompts.py`, versioned per the id convention so an eval
   run can tell v1 and v2 results apart) with an explicit instruction to avoid
   full names, phone numbers, addresses, or ID/tax numbers in `title`,
   `description`, or `topics`; refer to people by role or relationship instead.
   Scoped to those three fields only -- the transcript itself is never touched.
2. **Deterministic safety net, in `anonymize.py`:** spaCy's `pt_core_news_sm`
   (~12 MB, CPU-only, no torch) for names, plus regex for email/phone/9-digit
   ID numbers. Research spike resolved in favour of spaCy over a HF
   token-classification model specifically for the "no torch" property --
   `parakeet-mlx` had just been dropped from dependencies in Phase 2 for
   pulling torch in, and reintroducing it for NER would have undone that.
   Pinned as a real dependency via its wheel URL (`uv add "pt_core_news_sm @
   https://..."`), not the imperative `spacy download` command, which does not
   survive `uv sync` -- confirmed by testing (the model disappeared on the
   first `uv sync` after an imperative install, before being pinned properly).
3. **Config toggle:** `settings.anonymize_metadata: bool = True`, exactly as
   specified.

**Scope decision made while implementing, worth recording:** the redaction
applies to the filename slug and the docx's OOXML core properties
(title/subject/keywords) -- never to the visible "Resumo" heading/paragraph in
the document body, and never to the transcript. Reasoning: by the time someone
has the `.docx` open, the full transcript with real names is already visible a
few paragraphs below the summary. Redacting the abstract at that point would
protect nothing and would just read as inconsistent (title says `[nome]`,
transcript says the name fifteen times). The two surfaces that actually travel
independently of opening the file -- the filename and embedded metadata -- are
where redaction has real value, so that's where it's scoped.

**A real accuracy gap found and fixed by testing against real generated titles,
not just constructed examples** (same lesson as Phase 3's docx bug): spaCy's
small model frequently mislabels an unfamiliar bare first name as `LOC`/`ORG`
instead of `PER` -- an uncommon first name (changed in this writeup and in the
test suite for privacy; the underlying recording is not in this repo) came back
as both `LOC` and `ORG` in different real titles from this project's own
recordings. Fixed by treating any single-token entity of *any* label as a name
candidate, not just `PER`. A second, narrower gap: a name immediately following
a sentence-initial capitalized word merges into one multi-token entity (e.g.
"Contactar Marta" -> one `LOC` span), because the model reads the sentence's
first word as capitalized-therefore-proper-noun.
Fixed by redacting just the last token of a multi-token, non-`PER` entity that
starts at position 0 -- accepting, as a documented trade-off, that a genuine
multi-word institution name opening a title loses its last word too. Both
fixes are covered by regression tests in `tests/test_anonymize.py`.

**Tests:** 13 tests in `tests/test_anonymize.py` against the real spaCy model
(not a fake -- its actual NER behaviour is what's under test, and the model is
a pinned dependency, not a runtime download, so there's no reason to avoid it
in the fast unit-test tier); `tests/test_pipeline_ids.py` for transcript-id
assignment, stability across a save/reload cycle, and legacy-state migration;
`tests/test_docx_writer.py` gained a class asserting metadata is redacted while
the visible body is not; `tests/test_summarize.py`'s filename tests rewritten
for the id-based scheme.

**Exit:** met, and verified end-to-end against all 9 real recordings, not just
the fixture set. Every generated filename now carries a stable id with no real
name in any of the 9; docx core-properties metadata was clean (`meta_has_name =
False`) on all 9, while the visible document body correctly still shows real
names in the 4 of 9 recordings that actually discuss people by name --
confirming the scope boundary works as designed, not just in unit tests. 58
unit tests green (`uv run pytest`).

---

## Phase 4 — Public benchmark dataset ✅ done

Completed 2026-09-24, on `dev`. Built as planned: FLEURS pt_br via
`scripts/fetch_dataset.py`, a diversified fixture set via `scripts/build_fixtures.py`.

**Recommendation: Google FLEURS, `pt_br` test split** → `data/fleurs_pt/`.

| Candidate | Why / why not |
|---|---|
| **FLEURS pt** | CC-BY, small test split, *published Whisper baselines to check ourselves against* (large-v3 = 3.65% WER). Clean read speech. **Pick this.** |
| Common Voice pt | CC0, larger, more accent variety, noisier — good **second** set, needs HF auth + big download |
| MLS pt | CC-BY, audiobooks — long-form, least like our domain |

Honest caveat to record in the docs: **FLEURS is clean read Brazilian Portuguese; the
real workload was noisy conversational European Portuguese.** The benchmark will
therefore *overstate* absolute quality. Its value is as a **relative** regression
signal — "did my change make things worse" — not an absolute quality claim. Adding a
handful of Common Voice pt clips as a noisier tier is a cheap partial mitigation.

- Download via `datasets`, convert to the pipeline's native shape: audio files +
  `manifest.jsonl` of `{audio_path, ground_truth, ground_truth_normalized,
  duration, split}`. Revision pinned as a constant in `fetch_dataset.py`
  (`70bb2e84b976...`); a `manifest_sha256` + license/source recorded in
  `data/fleurs_pt/DATASET_INFO.json` (not committed -- regenerated by the
  script, which is the actual reproducibility guarantee via the pinned
  revision; `data/` stays fully gitignored).
- **Bypassing `datasets`' own audio decoding was a deliberate, necessary
  choice, not an oversight:** newer `datasets` versions require `torchcodec`
  (-> torch) to auto-decode the `audio` column, which would have silently
  undone Phase 2's removal of torch (it came in via `parakeet-mlx`, dropped
  because nothing uses it). Fixed with `Audio(decode=False)` and writing
  FLEURS' raw WAV bytes straight to disk -- this pipeline's own
  ffmpeg-based normalize stage handles whatever codec they turn out to be
  anyway.

**The committed fixture set is deliberately diversified, not just the first N
clips alphabetically** -- this is what closes the gap the Phase 3 review raised:
tests and eval should stress real audio, not only fakes. Concretely, in
`tests/fixtures/` (4 clips, ~660 KB total):
- a mix of **durations**: shortest (4.4s), longest (37.1s), and two evenly
  spaced points between them across the full 919-clip split -- long-form audio
  is exactly where Phase 2 found Whisper's repetition-loop hallucination, and
  a fixture set of only short clips would never exercise that path;
- a mix of **formats**: FLEURS itself ships one canonical format (16kHz mono
  float32 WAV), so each of the 4 picked clips is re-encoded with ffmpeg into a
  different format -- one each of `.wav` (unchanged), `.mp3`, `.m4a`, `.opus`
  -- exercising `audio.py`'s normalize logic across codecs the way the
  original personal test set (5 `.m4a` + 4 `.opus`) did by accident.
- This same set is what `tests/test_audio.py` (Phase 10 material, written
  now rather than deferred) uses for anything touching real transcript
  content -- synthetic sine-wave WAVs stay useful for pure-plumbing tests but
  can't stand in for real speech. One fixture set, already used by two
  consumers (the eval harness will be the third, in Phase 6), per the Phase 3
  review discussion. CC BY 4.0 attribution recorded in `tests/fixtures/NOTICE.md`.

**Three real bugs found and fixed by building this against real data, not
just plausible-looking fakes -- the third and fourth instances of this
pattern in a row (after Phase 3's docx bug and Phase 3b's spaCy
mislabeling):**
1. **`is_already_target_format()` never checked the actual sample format.**
   FLEURS' own WAV files are 16kHz mono -- matching the target rate and
   channel count -- but float32 PCM, not the 16-bit PCM the pipeline
   normalizes to. The check only compared rate/channels/extension, so a
   float32 WAV would have been "already normalized" and copied through
   unconverted. Harmless today (mlx-whisper's own loader tolerates float32
   fine, verified directly), but the function's whole job is to guarantee
   normalization, and it wasn't actually checking the thing that mattered.
   Fixed by adding `settings.target_codec` and checking it explicitly; caught
   only because the fixture set includes a real float32 WAV, which no
   synthetic fixture had reason to be.
2. **Trusted a metadata field instead of measuring the fact directly.**
   `fetch_dataset.py` first computed duration as `num_samples / 16000` from
   the dataset's own metadata. A spot check against real ffprobe-measured
   durations found ~60% of a random sample disagreed, by several seconds in
   some cases. Fixed by measuring duration from the file actually written to
   disk via this project's own `audio.probe()`, rather than trusting a field
   that turned out not to be reliable for this dataset revision.
3. **Non-unique key silently overwrote 570 of 919 clips.** Filenames were
   first keyed by the dataset's `id` field, assumed to be a unique row id. It
   isn't -- FLEURS records multiple speakers reading the same prompt
   sentence, and they share an `id`. Only 349 of 919 `id` values were
   actually unique, so 570 rows silently overwrote another row's audio file
   with **no error raised anywhere** -- the manifest still listed all 919
   entries, but many pointed at the wrong audio. This is the most serious of
   the three: a benchmark with silently-misaligned audio/transcript pairs
   would have produced meaningless WER numbers in Phase 6 without any visible
   failure. Fixed by keying filenames on each row's position in the split,
   which cannot collide by construction; `id` kept as informational metadata
   only. Verified after the fix: 919 unique audio paths for 919 rows, 0
   duration mismatches on a fresh random sample.

**Exit:** met. `uv run python scripts/fetch_dataset.py` produces a valid,
verified manifest (919/919 unique files, spot-checked durations correct); the
4-clip fixture subset is committed, diversified by duration and format, and
all 4 transcribe correctly end-to-end through the real pipeline (compared
against ground truth, not just "doesn't crash"). 80 unit tests green.

---

## Phase 5 — Retire the private test data ✅ done

Approved 2026-09-24 (decision #1): the source recordings are backed up externally
and are to be removed from this machine; the repo generalises to a plain
speech-to-text utility with no tie to any particular recording set. Run once Phase 4
was green and the pipeline had already been validated end-to-end on public FLEURS
data instead.

- Deleted `Arquivo.zip`, `tmp/`, `output/` from the local machine.
- Verified first, before deleting anything, that none of `Arquivo.zip`, `tmp/`,
  `output/`, or `data/` ever appeared in git history:
  `git log --all --full-history -- Arquivo.zip tmp/ output/ data/` returned nothing.
  Phase 0's `.gitignore` (written before the first commit) did its job — there was
  no history to scrub, just local files to remove.
- `data/fleurs_pt/` (the public FLEURS download used by `scripts/fetch_dataset.py`
  and `scripts/build_fixtures.py`) was kept — it's gitignored only because it's
  large derived data, not because it's private.
- Irreversible on this machine — done only because the external backup exists.

**Exit:** repo and working tree contain no personal audio or transcripts; the
pipeline was already confirmed (Phase 4) to run end-to-end on FLEURS data, so no
further functional verification was needed here.

---

## Phase 6 — Evaluation harness + MLflow

The core of the request: a pragmatic, repeatable way to know whether a model or prompt
change helped.

### Metrics

*ASR stage — has ground truth:*
- **WER / CER** via `jiwer`, primary. Needs a **Portuguese-aware normalizer**
  (casefold, strip punctuation/accents-optional, expand numerals, unify `ç`/digits).
  Whisper's bundled normalizer is English-only — writing the pt one is real work and a
  common source of bogus WER deltas.
- **Semantic distance** (SemDist) via multilingual sentence embeddings
  (`paraphrase-multilingual-MiniLM`). Recent work finds it aligns with human judgement
  better than WER; BERTScore costs far more and, per the same work, does not beat CER.

*Refine stage — no ground truth of its own, but measurable:*
- The clever part: run **WER against the FLEURS ground truth before and after refine**.
  If cleanup helps, WER drops; if the LLM is paraphrasing, it rises. This turns an
  unmeasurable stage into a measurable one, and directly tests the failure mode already
  observed (the 4B shifting tense, models dropping repetitions).
- Plus content-recall (already prototyped) and length-delta guards.

*Summary stage — no ground truth:*
- **Deterministic only, per decision #6.** Schema-validity rate, retry rate, field-length
  and topic-count conformance, filename-slug uniqueness, language-variant agreement with
  the dataset's known language, and embedding similarity between the description and the
  transcript it describes (catches a summary that drifts off-topic). No LLM scoring —
  a local model grading its own family's output shares its blind spots and skews lenient.
- ROUGE/BLEU deliberately excluded: they need reference summaries we do not have, and
  reward lexical overlap over meaning.

*Interpretation report (LLM, non-scoring):*
- After metrics are computed, a local LLM writes a short markdown **interpretation** of
  the numbers — what moved, what likely caused it, what to look at — logged as an MLflow
  artifact alongside the raw results.
- It reads **only the computed metrics table**, never the audio or transcripts, and it
  produces **no scores**. Every number in the report traces to a deterministic metric, so
  the LLM is a writing aid over trustworthy inputs, not a source of evidence.

*Performance — every stage:*
- Realtime factor (audio-sec / wall-sec), latency p50/p95, tokens in/out and tok/s for
  LLM stages, peak RSS, model load time (cold vs. warm — the 166 s first-run figure
  earlier was almost all compile).

### Tiered runs

The full FLEURS test split is far more than 10 minutes of pipeline time, so:

| Tier | Size | Target wall time | Use |
|---|---|---|---|
| `smoke` | 3–5 committed clips | < 60 s | unit/CI, no network |
| `quick` | stratified random **n=30**, fixed seed | ~5 min | pre-commit, CI on main, day-to-day |
| `full` | whole split | as long as it takes | release checks, model swaps |

Stratify by clip duration so the subsample isn't all short easy clips; fixed seed so
runs are comparable; seed overridable to check subsample stability.

### MLflow

Local file-backed tracking (`mlruns/`, `mlflow ui`) — no server to run.
- One run per benchmark; params = model IDs, prompt-template IDs/hashes, tier, seed,
  settings snapshot. Metrics = everything above. Artifacts = per-clip results table,
  the diff of worst regressions.
- `scripts/benchmark.py --tier quick --tag <name>` and a `compare` mode that diffs two
  runs and **exits non-zero past a configurable regression threshold** — that exit code
  is what CI consumes.
- Keep the metric computation **independent of MLflow** (pure functions returning a
  results object; MLflow is just a sink). If MLflow proves annoying, the fallback is
  writing the same object to `benchmarks/*.json` + a markdown table, with no rework.

**Tests:** metric functions against hand-computed fixtures (a known WER pair, identical
strings → 0, empty edge cases), normalizer unit tests, subsampling determinism, the
regression-gate exit-code logic, MLflow logging against a temp tracking dir.

**Exit:** `--tier quick` completes in ~5 min and produces an MLflow run; deliberately
swapping to a worse model shows a visible metric drop and a non-zero compare exit.

---

## Phase 7 — FastAPI service, persistence, logging

**Async model:** transcription is minutes-long, so endpoints must not block. Submit a
job → `202` + `job_id` → poll status / fetch result. Background execution via a worker
task; the ASR and Ollama calls are CPU/GPU-bound and synchronous, so they run in a
thread pool with a **concurrency limit of 1–2** — a 16GB M4 running whisper + a 9B
simultaneously will swap and crater.

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness + ffmpeg/Ollama/model reachability and versions |
| `POST /v1/jobs` | upload audio (multipart), optional per-job config override → `202` |
| `GET /v1/jobs` · `GET /v1/jobs/{id}` | list / status + stage progress |
| `GET /v1/jobs/{id}/result` · `/docx` | transcript JSON · generated document |
| `DELETE /v1/jobs/{id}` | cancel / purge |
| `GET /v1/config` · `PATCH /v1/config` | read / change models, prompts, options |
| `GET /v1/models` | what's actually available in Ollama right now |
| `POST /v1/benchmark` · `GET /v1/benchmark/{id}` | trigger an eval run, fetch results |

Config mutation notes: validate against available Ollama models on `PATCH`, reject
changes while jobs are in flight (or version the config per job — a job must record the
config it *ran under*, not the current one), and persist so restarts keep the change.

**SQLite** (`runs.db`, SQLModel/SQLAlchemy): `jobs`, `stage_timings`, `benchmark_runs`,
`config_history`. `jobs.id` **is** the `transcript_id` introduced in Phase 3b, not a
fresh autoincrement -- the same identifier that names a `.docx` file on disk is the
one this table and the API's `job_id` use, so a file, a JSON record, and a DB row
for the same recording are always the same id. Gives the frontend history, and
makes "did this get slower last Tuesday" answerable.

**Logging:** `structlog` JSON to file + readable console, request-ID middleware
propagated into stage logs and stored on the job row, per-stage timing at INFO, prompts
and transcript bodies at DEBUG only — **never log transcript content at INFO**, given
what it contains.

**Tests:** `httpx.AsyncClient` against the app with fake ASR/LLM — happy path, bad
upload type, oversized file, unknown job, config validation, concurrency cap,
cancellation. Plus one real end-to-end integration test.

---

## Phase 8 — OpenAPI/Swagger review

Treated as its own pass, not a side effect.
- Rich `summary`/`description` on every route, tags with descriptions, realistic request
  and response `examples`, documented error models (RFC 7807-style `Problem`), explicit
  status codes, enum-typed fields, upload constraints stated.
- Top-level `description` with a quickstart, the async job lifecycle explained, and a
  note that everything runs locally.
- Review by actually opening `/docs` and reading it as a newcomer would.

---

## Phase 9 — Streamlit demo

Single-screen demo: upload → run → result → reset.
- Start from a clean community template; keep it minimal and themed via
  `.streamlit/config.toml`.
- Polls the API (does **not** import the pipeline) so it exercises the real service.
- Shows: live stage progress, generated title + description, refined transcript, a
  raw/refined toggle, `.docx` download, sensitivity banner when the summary flags
  `high`, and a "Start over" reset.
- Its own log file; API errors surfaced as readable messages, not tracebacks.

**Tests:** logic extracted into pure helpers and unit-tested; a smoke test that the app
imports and renders headlessly. (Streamlit UIs resist deep automated testing — I'd keep
coverage honest and shallow here rather than fake it.)

---

## Phase 10 — Test suite and linting

```
tests/unit/          fast, no network, no models — the bulk
tests/integration/   real Ollama + real MLX, marked, opt-in
tests/e2e/           API up → upload → poll → docx out
tests/eval/          metric correctness (not model quality)
```

- `pytest` + `pytest-asyncio` + `pytest-cov`; markers `unit`/`integration`/`e2e`/`slow`;
  default `pytest` run = unit only, so it stays fast.
- Shared fixtures: the diversified real-audio set from Phase 4 (varied duration and
  format -- this is what would have caught Phase 2/3-style bugs earlier) for anything
  touching real transcript content, plus tiny generated WAVs (ffmpeg sine/silence)
  for pure-plumbing tests that don't need real speech; fake ASR + fake LLM clients
  for logic tests; temp settings, temp SQLite, temp MLflow dir.
- Coverage target ~80% on `src/`, with the honest exception of thin model-call wrappers.
- **Ruff** for lint + format (replacing black/isort), configured in `pyproject.toml`,
  plus `mypy` on `src/` if it isn't a fight. Dev deps via `uv add --dev`.

---

## Phase 11 — CI/CD

**The constraint that shapes this phase:** CI cannot run the production models. GitHub's
macOS runners are Apple Silicon (so MLX *works*), but pulling a 6.6 GB Ollama model on
every run is not viable, and Linux runners can't run MLX at all.

Tiered strategy:

| Where | What runs |
|---|---|
| `ubuntu-latest`, every PR | ruff, mypy, unit tests. Models faked. Fast, free. |
| `macos-latest`, every PR | import checks + `whisper-tiny` smoke on the committed fixture, cached via `actions/cache` |
| Push to `main` | the above **plus** `quick` benchmark |
| Self-hosted (this Mac mini), optional | full-model integration + `full` benchmark |

The `main` benchmark is **non-blocking by design**, as you asked: it compares against the
last stored baseline and, on regression past threshold, posts a warning annotation /
opens an issue rather than failing the build. Baseline stored as a committed JSON
artifact so the comparison survives across runs.

Pre-commit hooks: ruff, ruff-format, trailing whitespace, `check-added-large-files`
(keeps `data/` and audio out), and **`detect-secrets`/`gitleaks`** — cheap insurance
given the `.env` situation.

**What I need from you:** a GitHub repo must exist and be set as `origin`. `gh` is not
authenticated here. If `GITHUB_ACCESS_PAT` in `.env` has `repo` + `workflow` scope I can
create the repo, push, and watch the first Actions run; otherwise you create the repo and
I'll wire the workflows and verify by inspection. **Note that a `workflow`-scoped PAT is
a powerful credential** — I'd rather you `gh auth login` interactively, and it costs you
about thirty seconds.

---

## Phase 12 — README, commit, final verification

- Rewrite the README for the finished system: what it is, architecture diagram,
  quickstart (CLI / API / UI), model rationale carried over from the current README,
  benchmark results with the FLEURS caveat stated plainly, API reference pointer,
  eval guide ("how to tell if your change helped"), and a privacy note.
- Squash-free, readable commit history; push to `main`.
- **Final audit pass**: walk this document top to bottom and verify each phase's exit
  criteria against the built system, reporting anything unmet rather than quietly
  dropping it.

---

## Sequencing

```
0 ─→ 1 (research, parallelisable)
  └─→ 2 ─→ 3 ─→ 3b ─→ 4 ─→ [5 gate] ─→ 6 ─→ 7 ─→ 8 ─→ 9
                                              └─→ 10 ─→ 11 ─→ 12
```

Phase 1 is desk work and can run alongside 2–3. Phase 3b (ids + PII-aware summaries)
sits between 3 and 4 because Phase 4's fixture set and Phase 6's eval harness should
be built against the filename/id scheme that will actually ship, not the one Phase 3
shipped first. Everything from 6 onward depends on the Phase 2 restructure. Phase 5
is a hard gate requiring your confirmation.

---

## Risks

| Risk | Mitigation |
|---|---|
| **PAT leak on first commit** | Phase 0 before anything; secret-scanning hook |
| **Irreversible deletion of real recordings** | Phase 5 gated on explicit confirmation + external backup |
| FLEURS ≠ production domain; benchmark flatters the system | State it everywhere; treat as relative signal; add noisier Common Voice tier |
| pt normalizer bugs produce fake WER deltas | Unit-test the normalizer first, against hand-built cases |
| Local LLM judge is unreliable / self-congratulatory | Calibrate against ~20 human labels; drop it if uncorrelated |
| 16 GB RAM: whisper + 9B concurrently | Hard concurrency cap of 1–2; measure peak RSS in the benchmark |
| CI can't run real models | Tiered CI; tiny models + fakes; self-hosted runner optional |
| Scope: this is 12 phases (+3b) | Phases are independently shippable; stop anywhere and have something coherent |
| LLM ignores the PII-avoidance prompt instruction | Layered with a deterministic NER pass on the metadata surface (Phase 3b, built) — never rely on prompt compliance alone |
| Fakes-only tests miss real-format/real-content bugs | Already happened *twice* now — Phase 3's 255-char docx bug, and Phase 3b's spaCy PER/LOC mislabeling only visible on real generated titles; both fixed by testing against real recordings, not constructed examples. Phase 4/10 fixtures diversified with real audio specifically to close this |
| Local NER model has real, known blind spots (small-model recall, sentence-initial merging) | Explicitly a safety net, not the primary control (the prompt instruction is); documented trade-offs in `anonymize.py` rather than chased indefinitely |

---

## Decisions made (2026-09-24)

| # | Decision |
|---|---|
| 1 | **Phase 5 delete approved.** `Arquivo.zip` is backed up externally; the data has been sent on and is to be removed. Repo generalises to a plain speech-to-text solution. |
| 2 | **Branch:** stay on `master` unless the first push produces `main`; revisit after. |
| 3 | **GitHub:** user does the first commit, creates the repo and pushes; brings the remote back afterwards. No PAT use by Claude. |
| 4 | **Visibility: public.** Drives the stricter Phase 0 below. |
| 5 | **MLflow:** local file-backed `mlruns/` + `mlflow ui`. No server until/unless dockerised. |
| 6 | **No LLM-as-judge scoring.** Deterministic metrics only — see revised Phase 6. An LLM *does* write a human-readable interpretation report, logged as an MLflow artifact. |
| 7 | **Summary language configurable:** `pt` or `en`, settings-driven. |
| 8 | **`transcript_id` format: 8-char random hex** (`secrets.token_hex(4)`), built in Phase 3b with no objection raised. ULID remains the fallback if filename-sort-by-creation-time ever matters more than brevity. |
| 9 | **Original filename stem dropped from the new `.docx` name**, built in Phase 3b with no objection raised. Traceable via the docx's provenance header and the `transcripts.json` record instead. Reversible at low cost (`<id>_<slug>__<stem>.docx`) if quick folder browsing turns out to matter more. |

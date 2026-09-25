# Implementation Plan — transcript_task

From a working local script to an evaluated, served, tested product.

**Status:** Phases 0–9 complete (2026-09-25, on `dev`). Phases 10-12 pending. All open
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
| Tests | 284 unit tests (`uv run pytest`) + 3 integration tests against real Ollama/mlx-whisper |
| Benchmark data | FLEURS pt_br (919 clips, `scripts/fetch_dataset.py`) + Common Voice pt (9,467 clips, `scripts/fetch_common_voice.py`); diversified fixtures committed under `tests/fixtures/` and `tests/fixtures_noisy/` |
| Eval harness | `scripts/benchmark.py run/compare --dataset {fleurs,common_voice}` — WER/CER/SemDist, refine before/after delta, deterministic summary metrics, MLflow (`mlruns/`) + local `benchmarks/*.json` |
| Refine safety net | production runtime guard (`text_compare.py` + `pipeline.stage_refine`) discards a refine call that drifts too far from the raw transcript, visibly, falling back to raw |
| Refine prompt | `refine-pt-v2` (default), tuned and measured against both benchmarks: cut refine's own WER regressions 7/30→2/30 (Common Voice) and 13/30→6/30 (FLEURS) vs. `refine-pt-v1`, still available by id |
| API | FastAPI (`transcript_task.api.app`) — jobs (async, per-job `refine` toggle, always both transcripts when refine runs), config (live-patchable, validated, versioned per job), models, benchmark endpoints; SQLite (`runs.db`) persistence; `structlog` JSON+console logging; RFC 7807 errors, full OpenAPI docs at `/docs` |
| Frontend | `frontend/app.py` — single-screen Streamlit demo (upload → poll → result → reset), polls the API only, never imports the pipeline |
| Missing | linting, CI |
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

## Phase 6 — Evaluation harness + MLflow ✅ done

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

### Implementation notes (completed 2026-09-24, on `dev`)

Built as `src/transcript_task/eval/` (pure, MLflow-independent metric functions,
tiering, results/report dataclasses, comparison/regression gate, MLflow sink) plus
`scripts/benchmark.py` (`run` / `compare` CLI), following the plan closely:

- `eval/normalizer.py` — pt-aware normalizer: NFKC + casefold, `num2words`-based
  digit expansion (`12` → `doze`), punctuation stripping, optional accent-folding.
  Accents kept by default deliberately — "más"/"mas" are different words, folding
  them by default would hide real errors, not just cosmetic ones.
- `eval/metrics.py` — `word_error_rate`/`character_error_rate` (jiwer, both sides
  normalized identically), `semantic_distance` (embedding cosine distance behind
  an `Embedder` protocol — same DI pattern as `Transcriber`/`ChatModel`, so tests
  never load the real MiniLM model), `refine_delta` (WER before/after refine
  against the same ground truth — the "clever part" from the plan), `content_recall`,
  `length_ratio`, `evaluate_summary` (deterministic conformance checks against
  `TranscriptSummary`), `slug_uniqueness`.
- `eval/tiers.py` — `smoke` (the 4 committed fixtures), `quick` (duration-bucketed
  stratified sample, fixed seed, default n=30), `full` (everything).
- `eval/results.py` — `ClipResult`/`BenchmarkResult`, JSON round-trip, markdown
  table rendering (aggregate metrics + summary-stage metrics + worst-WER clips).
- `eval/compare.py` — diffs mean WER/CER/SemDist between two runs; a metric
  "regresses" if it worsens by more than an absolute threshold (default 0.02) in
  *both* runs having measured it (a metric missing from either side is skipped,
  not treated as a regression by omission).
- `eval/interpretation.py` — sends the already-computed markdown metrics table to
  the local LLM and asks for a short interpretation; never raises (an LLM outage
  degrades to a placeholder string, since the deterministic metrics it would have
  described are already safely saved).
- `eval/mlflow_sink.py` — local file-backed MLflow logging, **and** always writes
  `benchmarks/<tag>/{results.json,table.md,interpretation.md}` regardless, per the
  plan's explicit fallback design.
- `eval/runner.py` — the only module that imports the real ASR/LLM code; runs one
  clip through normalize → transcribe → refine → summarize using the same
  `Transcriber`/`ChatModel` building blocks as `pipeline.py`, not `pipeline.py`'s
  own batch/checkpoint machinery (which is oriented around `output/transcripts.json`
  and writing real `.docx` files — neither of which a benchmark run wants).

**One real, load-bearing bug found while building this, unrelated to the pipeline
itself:** MLflow 3.x put the plain filesystem tracking backend (`file:./mlruns`)
into maintenance mode — `mlflow.set_tracking_uri("file:...")` now raises outright
unless `MLFLOW_ALLOW_FILE_STORE` is set, nudging new projects toward a SQLite/DB
backend. This directly collided with decision #5 below (local file-backed
tracking, no server). Rather than revisit that decision for a single-user local
tool, `eval/mlflow_sink.py` sets the opt-out env var at import time — `mlruns/` +
`mlflow ui` behave exactly as originally decided; the DB backend was never
actually needed here, just newly gatekept.

**A non-obvious design call in the eval package: `used_retry`/`used_fallback`
detection is inferred, not reported.** `summarize_transcript` doesn't expose
whether it needed its one retry — doing so would mean changing a function every
other stage already depends on, just to serve the benchmark. Instead
`eval/runner.py` wraps the `ChatModel` passed to `summarize_transcript` in a
small local `_CountingChatModel` that counts calls (1 = no retry needed, 2 = retry
was attempted) and combines that with a structural check (does the returned
summary equal `TranscriptSummary.fallback()` exactly) to tell "retried and
succeeded" apart from "retried and still fell back." Non-invasive, and the
distinction is real: it showed up immediately in the smoke run below (both flags
correctly read `0.0` — the 9B model got every summary right on the first try).

**First real benchmark run (`smoke-test-1`, `--tier smoke`, 4 committed clips,
`mlx-whisper large-v3-turbo` + `qwen3.5:9b`)** — the harness's first real result,
not just "it runs without crashing":

| metric | mean |
|---|---|
| wer_raw | 0.0342 |
| **wer_refined** | **0.1084** |
| cer_raw | 0.0140 |
| cer_refined | 0.0488 |
| schema_valid_rate | 1.0 |
| retry_rate / fallback_rate | 0.0 / 0.0 |

**Refine made WER *worse* on every one of the 4 clips**, not better — exactly the
failure mode the plan's "clever part" (scoring refine against ground truth
before/after) exists to catch, and it caught it on the very first real run. This
isn't a harness bug: FLEURS is clean, unambiguous read speech, so ASR output
already has almost nothing to clean up, while refine's paraphrasing risk (correct
wording rewritten into different-but-still-correct wording) still fully applies
and is the only thing left for it to do. The LLM-written interpretation report
(reproduced in full in `benchmarks/smoke-test-1/interpretation.md`) independently
identified the same thing from the metrics table alone and recommended
investigating `refine-pt-v1` — without inventing a single number not already in
the table, which is exactly what it's scoped to do. This is documented as an
honest limitation in the README, not papered over: it says nothing about whether
refine helps on the noisier, disfluent conversational audio the tool actually
targets (FLEURS clips have no hesitations or disfluencies to remove), which is
precisely the caveat Phase 4 already recorded about this dataset.

**Exit, verified for real:** `--tier smoke` ran end-to-end with 0 errors, produced
a valid MLflow run (`mlflow.tracking.MlflowClient` round-trip confirmed:
`wer_refined_mean` readable back from the tracking store) and local
`benchmarks/smoke-test-1/` artifacts. 175 unit tests green, covering hand-computed
WER/CER pairs, the normalizer (casefolding, digit expansion, idempotence,
accent-folding on/off, a pathological huge-digit-run input that would otherwise
crash `num2words`), stratified-sampling determinism and duration coverage, the
regression-gate's threshold logic including the "metric missing from one side is
skipped, not flagged" edge case, and MLflow logging against a temp tracking dir.

**A second real bug, found by the exit-criterion demo itself, and fixed:** to
verify "deliberately swapping to a worse model shows a visible metric drop and a
non-zero compare exit" for real rather than just in unit tests, `--tier smoke`
was re-run with `TRANSCRIPT_LLM_MODEL=llama3.2:1b` in place of `qwen3.5:9b`. One
of the 4 clips took **3041 seconds** (~51 minutes) for a single refine call on a
~10s audio clip, generating **163,840 tokens** — a runaway repetition loop that
never stopped on its own. `wer_refined` for that clip came back as 4515 (WER has
no upper bound: it's edit distance over reference length, and the "refined"
output was thousands of times longer than the raw transcript). Root cause:
`settings.llm_options` set `temperature` and `num_ctx` but no cap on *generated*
tokens, so a model that doesn't reliably stop had nothing bounding it short of
context exhaustion. Fixed by adding `llm_num_predict` (default 8192, generous for
a single recording's refine/summary output, verified against the real 202s-refine
recording's scale) to `Settings`, threaded into `llm_options` as Ollama's
`num_predict`. This is a pipeline robustness fix, not just an eval-harness one:
the same runaway risk existed in production `pipeline.py` for anyone who
switched `llm_model` to a smaller/different model, silently, with no timeout.
Regression-tested in `tests/test_settings.py`.

With the exit-criterion demo's real (if extreme) data now captured, `compare`
against it works exactly as specified:

```
$ uv run python scripts/benchmark.py compare smoke-test-1 smoke-worse-llm

| metric | baseline | candidate | delta | regressed |
|---|---|---|---|---|
| wer_raw | 0.0342 | 0.0342 | +0.0000 |  |
| wer_refined | 0.1084 | 1129.1570 | +1129.0486 | YES |
| cer_refined | 0.0488 | 1165.9311 | +1165.8823 | YES |
| semdist_refined | 0.0289 | 0.0788 | +0.0499 | YES |

[benchmark] REGRESSION detected past threshold.
$ echo $?
1
```

`wer_raw`/`cer_raw`/`semdist_raw` correctly show **no** change (only the LLM was
swapped, not the ASR model) while every refine-stage metric is flagged — the
comparator distinguishing which stage actually regressed, not just "something
got worse," is exactly the point.

### Second follow-up (2026-09-24): a noisier tier, and a production refine-quality guard

Prompted by discussion after the above: the FLEURS finding (refine raises WER)
only says something about refine on *clean, studio-quality read speech* — the
plan's own risk table always flagged FLEURS as unrepresentative of this tool's
actual noisy-conversational target domain. Two follow-ups, both built and run
against real data rather than left as a hypothesis:

**1. A second, noisier benchmark dataset: Common Voice pt.**
- `mozilla-foundation/common_voice_17_0` (the official HF org's repo) turned
  out to be a dead end, not just an auth hassle as the risk table guessed:
  it ships a Python loading script, and `datasets` 5.x has **removed loading-
  script support entirely** (`trust_remote_code "is not supported anymore"`,
  confirmed by actually trying it). `fsicoli/common_voice_17_0` is a
  well-maintained community mirror of the same release, republished as plain
  per-language `.tar` audio shards + `.tsv` transcripts, still `CC0-1.0`
  (verified from the repo's own README, not assumed). `scripts/fetch_common_voice.py`
  fetches it directly via `huggingface_hub` (no `datasets.load_dataset`
  needed at all here, so the torchcodec concern doesn't even arise).
- 9,467 clips fetched, 0.9s–10.6s each, verified unique with no duration
  mismatches (the FLEURS lessons — never trust a metadata field, always
  measure with `probe()`, key by position not a dataset id — were applied
  from the start this time, and no equivalent bugs turned up).
- `scripts/build_fixtures.py` generalised (`--source-dir`/`--dest-dir`/
  `--source-ext`) to build a second committed smoke set, `tests/fixtures_noisy/`
  (4 clips, mp3/wav/m4a/opus, 0.9s–10.6s), from real Common Voice data. It
  needed one real generalization, not a rewrite: the "native format" to copy
  unchanged is now a parameter instead of hardcoded `wav`.
- `eval/tiers.py`/`eval/runner.py` needed **zero changes** — the manifest
  schema (`audio_path`/`ground_truth`/`duration`) was already dataset-agnostic
  by construction. `scripts/benchmark.py` gained a `--dataset {fleurs,common_voice}`
  flag and a small registry of per-dataset paths; `BenchmarkResult` gained a
  `dataset` field so a run always self-documents which corpus it's from, and
  `compare` refuses to diff runs from different datasets (their WER numbers
  aren't on the same scale).
- **A real finding from the committed smoke fixtures, before any tuning run:**
  2 of the 4 real Common Voice clips came back with `wer_raw = 1.0` — total
  ASR misses, not refine problems. One (a 0.9s clip of the single word
  "apuração") was transcribed as "Obrigada." — a well-known Whisper
  hallucination pattern on very short/quiet audio. The other (a 10.6s clip)
  came back as "Me задissinou canubar." — **Cyrillic characters mixed into
  Portuguese-looking fragments**, reproduced identically from the original
  source mp3 (not an artifact of this project's own format conversion,
  verified by transcribing the untouched source file directly). Real
  background noise and recording-device variance breaks even large-v3-turbo
  in ways FLEURS' clean narration never surfaced — exactly why this tier
  exists.
- **The actual payoff, run for real at `--tier quick` (n=30, stratified,
  seed=42):** `wer_raw` mean 0.0852 (vs. FLEURS' 0.0342 — confirms Common
  Voice really is harder), `wer_refined` mean 0.1360. Per-clip: refine
  **improved** WER on 1 of 30 clips, **worsened** it on 7, left 22 unchanged.
  This directly answers the question the FLEURS-only result couldn't: refine
  is net-negative on WER even on noisier, more varied real audio, not just
  on unnaturally clean read speech. That doesn't settle whether refine helps
  overall (it isn't scored for punctuation/readability, which is most of its
  actual job, and neither FLEURS nor Common Voice contains genuinely
  spontaneous disfluent speech, the domain refine's prompt actually targets)
  — but it does mean the earlier FLEURS result wasn't just an artifact of
  testing on abnormally perfect audio.

**2. A production runtime guard on refine, using the eval harness's own metrics.**
`content_recall` and `length_ratio` (ground-truth-free, already built for
Phase 6's benchmark) are now also a **live check on every refine call**, not
just a benchmark metric:
- Moved out of `eval/metrics.py` into a new `text_compare.py` (alongside
  `normalize_pt`), specifically so the production pipeline can import them
  without pulling in `eval/`'s jiwer/sentence-transformers/mlflow
  dependencies — a real layering fix, verified by a test
  (`test_text_compare.py`) that checks the module imports cleanly with
  `transcript_task.eval` untouched.
- `pipeline.stage_refine` now calls `refine_rejection_reason()` right after
  every refine call. If content_recall or length_ratio fails its (generous,
  documented-as-uncalibrated) threshold, the refined output is **discarded**
  — `refined_transcript` is left unset, which every downstream stage
  (summarize, docx) already falls back to the raw transcript for, with zero
  changes needed there.
- This fails **soft and visibly**, not silently: the rejected text is kept
  (`item["refine_rejected"]`, including the metrics that triggered it), the
  docx shows a bold warning banner naming the reason and the numbers, the
  provenance line distinguishes "revision never attempted" from "revision
  attempted and discarded" (previously both looked identical), and the
  rejected attempt itself is kept visible in an appendix so a reviewer can
  judge the guard's call rather than just trust it blindly.
- **Verified against the real incident, not just synthetic unit tests:**
  re-running the exact clip that caused Phase 6's original 51-minute/163,840-
  token runaway (`llama3.2:1b` on the "Correntes de retorno..." sentence)
  through the real, now-guarded pipeline reproduces the slow generation
  again — confirming it's a real, repeatable model+prompt failure mode, not
  a one-off fluke — but this time bounded by both fixes together, measured:
  521.6 seconds (~8.7 min) instead of 3041s (~51 min) -- the `llm_num_predict=8192`
  cap doesn't scale down generation time linearly with the token cap (real
  generation still ran long relative to a normal ~2s call), but it is a real,
  substantial bound, not a cosmetic one. The output was still 26,398
  characters of runaway text (length_ratio 221.8, content_recall 0.273
  against the ~120-character raw transcript); the new guard caught it on
  `content_recall too low` and correctly discarded it, falling back to the
  raw transcript. This is the honest result of the actual re-run, not a
  rounded-off estimate.
- 12 new unit tests (`test_pipeline_refine_guard.py`) cover: normal cleanup
  passing, legitimate heavy hesitation-removal shrinkage passing (the guard
  must not punish refine for doing its actual job), runaway generation and
  near-total content loss both correctly rejected, thresholds overridable
  via `Settings`, and `--force` correctly clearing a stale result in either
  direction (accepted → rejected on a worse model, rejected → accepted on a
  better one). 6 new tests (`test_docx_writer.py::TestRefineRejected`) cover
  the document rendering.

### Third follow-up (2026-09-25): tuning the refine prompt, measured

**Diagnosis before writing anything:** rather than guess, `raw` vs `refined`
text was inspected directly (not just the aggregate WER) on the 8 committed
smoke clips. This found a consistent pattern behind v1's WER erosion — the
model volunteers small, unforced rewrites of text that was already correct:
`"como uma rota"` → `"como numa rota"` (no rule justifies this at all),
`"das ondas na praia voltando ao mar"` → `"que volta ao mar"` (a stylistic
rewrite of an already-grammatical sentence), and one actively **wrong**
"agreement fix" — `"gorilas mais baratas"` → `"...baratos"` — that guessed
the wrong noun for an ambiguous adjective and flipped a correct sentence
into an incorrect one. None of this is hallucination or runaway generation
(the other two follow-ups' failure modes); it's a system prompt ("professional
reviewer") and rule set that never told the model *not* to polish text that
didn't need it.

**`refine-pt-v2`** (`prompts.py`) adds an explicit minimal-edit principle
ahead of every other rule ("if it's already correct, leave it exactly as-is,
even if you can imagine a more elegant phrasing — you are not improving the
text, only fixing genuine errors"), a new rule explicitly forbidding
synonym-substitution/restructuring of already-correct text, and narrows the
concordância (agreement) rule to defer instead of guess when a term's
referent is ambiguous. `Settings.refine_prompt_id` (default still
`refine-pt-v1` until the comparison below is complete) makes this an A/B
switch (`TRANSCRIPT_REFINE_PROMPT_ID=refine-pt-v2`) — `pipeline.py`,
`eval/runner.py`, and `scripts/benchmark.py` were updated to resolve the
prompt through it instead of a hardcoded import, exactly the plumbing
decision #14 said was already in place.

**Real, measured result — Common Voice, `quick` tier, n=30, identical clips
(same seed) as the earlier v1 baseline:**

| metric | v1 (baseline) | v2 (candidate) | delta |
|---|---|---|---|
| wer_refined (mean) | 0.1360 | 0.1013 | **−0.0346** (−25% relative) |
| cer_refined (mean) | 0.0350 | 0.0229 | **−0.0120** (−34% relative) |
| per-clip: improved / worsened / unchanged | 1 / 7 / 22 | 0 / 2 / 28 | regressions cut from 7 to 2 |

`compare noisy-quick-1 noisy-v2-candidate` reports no regression past
threshold on any metric, and every refine-stage metric moved in the better
direction. `wer_raw`/`cer_raw`/`semdist_raw` are bit-for-bit unchanged, as
expected (same ASR, same clips) — confirming the difference is attributable
entirely to the prompt, not sampling noise.

**The two remaining v2 regressions were inspected directly too, and are a
genuinely different failure mode than v1's — not solved by "make it more
conservative," and worth recording precisely rather than papering over:**
1. `"Paulo e Joana estão trabalhando no projeto"` → `"...estão a trabalhar
   no projeto"` — the model converted a Brazilian gerund construction
   (`estar + gerúndio`) into the European periphrastic form (`estar a +
   infinitivo`), directly against the explicit variant-preservation rule
   (present unchanged since v1). A genuine, specific bug, not a vague
   "over-editing" tendency — a good, concrete target for `refine-pt-v3`.
2. A word-enumeration-style Common Voice prompt (`"impugnar, imunidade,
   regalias, privilégios, outorgados"`, read as a list, not a sentence) was
   "corrected" into fluent prose (`"...regalias e privilégios outorgados"`)
   by the punctuation-naturalization rule (rule 2, present since v1)
   reasonably — but wrongly — assuming a missing conjunction. This is a
   collision between a legitimate rule and an unusual source-content style,
   not a prompt-wording defect; not clear it's fixable without weakening
   rule 2's real value elsewhere.

**FLEURS, `quick` tier, n=30, identical clips (same seed) in both runs:**

| metric | v1 (baseline) | v2 (candidate) | delta |
|---|---|---|---|
| wer_refined (mean) | 0.0489 | 0.0289 | **−0.0199** (−41% relative) |
| cer_refined (mean) | 0.0210 | 0.0103 | **−0.0107** (−51% relative) |
| per-clip: improved / worsened / unchanged | 1 / 13 / 16 | 2 / 6 / 22 | regressions cut from 13 to 6 |

An even bigger relative win than Common Voice, and consistent with the
hypothesis: FLEURS' clips are cleaner, so v1 had more perfect raw
transcripts available to accidentally break (13/30, the highest regression
rate measured anywhere in this project) — v2's minimal-edit principle stops
most of that. One genuine measurement-noise data point surfaced here too,
worth being honest about: `wer_raw` moved by +0.0014 between the two runs
despite scoring the identical 30 clips, because mlx-whisper's own output for
one clip (`fleurs_row00401`) differed very slightly between the two separate
process runs — real ASR non-determinism, not a bug in the harness. It's
about 15x smaller than the refine-stage improvement being measured, so it
doesn't change the conclusion, but it's a real noise floor worth remembering
before treating a small delta as meaningful in future comparisons.

**Decision: `refine-pt-v2` is now the default** (`Settings.refine_prompt_id`).
The improvement is large, consistent across both independent datasets, and
directly explained by inspecting real examples rather than an unexplained
metric wiggle. `refine-pt-v1` stays available by id for comparison/rollback.
Two new v1-vs-v2 unit tests and 9 prompt-content tests (`test_prompts.py`)
cover the template lookup, id stability, and that v2 didn't regress any of
v1's existing safety rules (no-invention, register/variant preservation,
`[?]` marker) while adding the minimal-edit principle.

**Still an open limitation, not solved by this:** neither FLEURS nor Common
Voice contains genuinely spontaneous/disfluent speech (refine's actual
target domain), so this tuning optimizes "reads correctly but gets reworded
less," not "actually has disfluencies to remove." A prompt that measures
well here is a real, verified improvement on what these two benchmarks can
see — not a substitute for eventually testing against real disfluent
conversational audio.

---

## Phase 7 — FastAPI service, persistence, logging ✅ done

**Async model:** transcription is minutes-long, so endpoints must not block. Submit a
job → `202` + `job_id` → poll status / fetch result. Background execution via a worker
task; the ASR and Ollama calls are CPU/GPU-bound and synchronous, so they run in a
thread pool with a **concurrency limit of 1–2** — a 16GB M4 running whisper + a 9B
simultaneously will swap and crater.

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness + ffmpeg/Ollama/model reachability and versions |
| `POST /v1/jobs` | upload audio (multipart), `refine: bool` + optional per-job config override → `202` |
| `GET /v1/jobs` · `GET /v1/jobs/{id}` | list / status + stage progress |
| `GET /v1/jobs/{id}/result` · `/docx` | transcript JSON · generated document |
| `DELETE /v1/jobs/{id}` | cancel / purge |
| `GET /v1/config` · `PATCH /v1/config` | read / change models, prompts, options |
| `GET /v1/models` | what's actually available in Ollama right now |
| `POST /v1/benchmark` · `GET /v1/benchmark/{id}` | trigger an eval run, fetch results |

Config mutation notes: validate against available Ollama models on `PATCH`, reject
changes while jobs are in flight (or version the config per job — a job must record the
config it *ran under*, not the current one), and persist so restarts keep the change.

**Requested 2026-09-25: `refine` is a user-facing, per-job choice, and both
transcripts are always in the result when it runs.** `POST /v1/jobs` takes a
`refine: bool` field (mirrors the CLI's existing `--skip-refine`, just
exposed as an explicit opt-in/out per job rather than a flag on the whole
process). Two rules, both binding on the response schema:
- `refine=false` — the refine stage never runs for that job (faster, no LLM
  cost); the result carries only `raw_transcript`.
- `refine=true` (default) — the refine stage runs, and the result **always**
  includes both `raw_transcript` and `refined_transcript` side by side, never
  just the refined one. This isn't new behavior invented for the API: the
  pipeline's state (`transcripts.json`) and every `.docx` it writes already
  carry both today (the docx's raw-transcript appendix exists for exactly
  this reason — see "What the refine pass actually fixes" in the README).
  The API's job here is to not regress that guarantee by only exposing one
  field where the pipeline already tracks two — a caller decides which
  version to show a user, the API doesn't decide for them.
- If the refine-quality guard (Phase 6 follow-up) rejects the refine
  attempt, the result should say so explicitly (`refine_rejected: {...}`,
  same shape already stored in pipeline state) rather than silently look
  identical to `refine=false` — a caller displaying "revised version
  unavailable" needs to know *why*.

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

### Implementation notes (completed 2026-09-25, on `dev`)

Built as `src/transcript_task/api/` (`app.py`, `db.py`, `job_runner.py`, `worker.py`,
`schemas.py`, `logging_config.py`), plus a shared `eval/orchestrator.py` extracted
from `scripts/benchmark.py` so the CLI and `/v1/benchmark` don't duplicate the
"run a tier end to end" logic. Every endpoint in the plan's table is real and
tested, not stubbed:

- **Job orchestration reuses `pipeline.py`'s stage functions directly**
  (`job_runner.run_job` calls `stage_normalize`/`stage_transcribe`/`stage_refine`/
  `stage_summarize`/`stage_docx` against a fresh single-item state dict), rather
  than duplicating their logic the way `eval/runner.py` does for a genuinely
  different reason (scoring against ground truth). An API job *is* a CLI run on
  one file; the only real difference is each job gets its own isolated
  `tmp`/`output` directories (`api_jobs_dir/<job_id>/`) via a per-job `Settings`
  copy, so concurrent jobs never collide the way sharing the CLI's single
  `transcripts.json` would.
- **Worker: a `ThreadPoolExecutor`, not a task queue.** `Settings.api_concurrency`
  (default 1) caps it — matches the plan's reasoning exactly (Whisper + a 9B
  model at once is already the realistic ceiling on a 16GB machine). Verified
  for real, not just asserted: a test with a tracking fake transcriber confirms
  peak concurrent execution never exceeds 1 across two simultaneously-submitted
  jobs.
- **Config: versioned per job, not rejected while jobs are in flight** — the plan
  offered both options; per-job versioning was picked because a per-job `Settings`
  copy already made it free (see above), and it's strictly more useful than
  blocking a legitimate config change for however long a job takes. `PATCH
  /v1/config` validates `refine_prompt_id` against `get_refine_template()` and
  `llm_model`/`asr_model` against `ollama.list()`, records an append-only
  `ConfigHistory` row (diff *and* full snapshot, so a restart reconstructs
  current config from the latest row without replaying history), and the
  change is visible to new jobs immediately.
- **DELETE is honest about what it can't do.** A still-queued job can be
  canceled outright (`Future.cancel()`); a currently-running one returns `409`
  rather than pretending to interrupt it — Python threads aren't preemptible
  and the ASR/LLM calls have no cancellation hook, so silently "succeeding"
  at canceling a running job would be a lie. Verified with a deterministic
  test (a blocking fake transcriber gated by a `threading.Event`), not a timing
  guess.
- **`jobs.id` is the `transcript_id`**, exactly as specified — no separate
  autoincrement anywhere.

**A real, load-bearing constraint discovered while wiring this up, not just a
test artifact:** `pipeline.py`'s stage functions store paths via
`.relative_to(PROJECT_ROOT)` (used to keep `transcripts.json`/docx paths
portable across machines). That means job/tmp directories for the API
**must** live under `PROJECT_ROOT` — confirmed the hard way when a first draft
of the test fixtures used pytest's own `tmp_path` (outside the repo) and hit
`ValueError: ... is not in the subpath of ...` immediately. `Settings.api_jobs_dir`
defaults to `PROJECT_ROOT / "data" / "api_jobs"` for exactly this reason; tests
mirror it (`PROJECT_ROOT / "data" / "test_api_jobs" / <unique>`) rather than
using `tmp_path` directly.

**A third, found by checking the actual build artifact rather than trusting
that tests passing meant the package was correctly assembled:** `api/` had
no `__init__.py` (Python's implicit namespace packages made every import
and every test pass anyway, silently). `pyproject.toml`'s hatchling build
only picks up explicit packages, so a real `uv build --wheel` produced a
wheel with the whole `api/` directory **missing** — verified by building the
wheel and checking its contents directly, not assumed from tests being
green. Fixed by adding the missing `__init__.py`; re-verified the wheel
contains all six `api/` modules afterward.

**Two real bugs found by the test suite, both fixed:**
1. `POST /v1/benchmark`/`create_benchmark` validated the response model
   (`BenchmarkRunOut.model_validate(run, ...)`) *after* its DB session block
   had already closed — a `DetachedInstanceError` on every call, caught
   immediately by the first benchmark test written. Fixed by building the
   response inside the session block, before it closes.
2. The eval harness's SemDist metric was silently loading `sentence-transformers`
   (and its sklearn/joblib/onnx dependency chain) inside a background thread
   during what should have been fast, offline API tests — `run_benchmark_tier`
   had no way to skip it. Fixed by adding a `semdist: bool = True` field to
   `POST /v1/benchmark`'s request schema, threaded through to
   `run_benchmark_tier(no_semdist=...)` — a real API capability gap this
   surfaced, not just a test-speed workaround: a caller who only wants
   WER/CER (no embedding-model load) now has a way to ask for that.

**Verified for real, beyond the test suite:** started the actual server with
`uv run uvicorn transcript_task.api.app:app` and hit `/health`, `/v1/config`,
`/v1/models` with `curl` against the real, running Ollama — confirmed the
dual JSON-file + console structlog output, and that `runs.db`/`data/logs/`
land in the right (gitignored) places. Ran the full real end-to-end
integration test (`test_real_job_end_to_end`, `-m integration`): a committed
FLEURS fixture goes in through the actual HTTP upload endpoint, comes back out
transcribed by real mlx-whisper and refined/summarized by real Ollama, as a
downloadable real `.docx` — 29.7s wall clock, matching ground truth.

**Exit:** all of the above, plus 33 new fast unit tests (fakes for ASR/LLM,
real SQLite, real background threads) and the one real integration test — 242
unit tests total, 3 integration tests total.

---

## Phase 8 — OpenAPI/Swagger review ✅ done

Treated as its own pass, not a side effect.
- Rich `summary`/`description` on every route, tags with descriptions, realistic request
  and response `examples`, documented error models (RFC 7807-style `Problem`), explicit
  status codes, enum-typed fields, upload constraints stated.
- Top-level `description` with a quickstart, the async job lifecycle explained, and a
  note that everything runs locally.
- Review by actually opening `/docs` and reading it as a newcomer would.

### Implementation notes (completed 2026-09-25, on `dev`)

Every item above is real, not partial:

- **RFC 7807 everywhere.** A new `Problem` schema (`type`/`title`/`status`/`detail`/`instance`)
  plus two global exception handlers (`HTTPException`, `RequestValidationError`) replace
  FastAPI's default `{"detail": ...}` shape with `application/problem+json` on every
  non-2xx response, app-wide — one place to get this right rather than per-route
  boilerplate. Verified for real: `GET /v1/jobs/deadbeef` and an invalid `PATCH
  /v1/config` body both checked directly, not just asserted by type.
- **Every route** got `tags`, `summary`, a `description` (initially 3 routes had only a
  `summary` — caught by actually walking the generated `/openapi.json` programmatically
  and checking every path, not by eyeballing the source), and a `responses={}` map
  documenting its real non-2xx outcomes with `Problem` as the model.
- **Realistic examples** via `json_schema_extra` on `JobCreateResponse`, `JobSummary`,
  `JobResult`, `ConfigPatch`, `BenchmarkCreateRequest`, and `Problem` — actual plausible
  values (a real-shaped `job_id`, a pt-PT summary), not `"string"`/`0` placeholders.
- **Enum-typed fields tightened further than Phase 7 left them:** `BenchmarkRun.status`
  was a plain `str` (`"running"`/`"done"`/`"failed"` set directly in code, no type backing
  them) — added a proper `BenchmarkStatus` enum, matching `JobStatus`'s existing pattern,
  so the OpenAPI schema shows the real closed set of values instead of `type: string`.
- **`JobResult.summary`/`refine_rejected` were previously untyped `dict`s** — showed in
  `/docs` as an opaque, propertyless "object". Now `RefineRejected` (new) and a new
  `SummaryOut` model give both a full, documented shape. `SummaryOut` deliberately
  mirrors `summarize.TranscriptSummary` rather than reusing it directly: that model's
  `model_json_schema()` is also what's sent to Ollama as the structured-output
  constraint (see `summarize.py`), so adding API-doc field descriptions there would
  risk changing an already-tuned production prompt path just to improve unrelated
  documentation. A short, separate, purely-additive model was the safer call.
- **Upload constraints are discoverable, not just described in prose.** `GET /v1/config`
  gained two read-only fields, `api_max_upload_mb` and `audio_extensions` — not
  patchable (excluded from `CONFIGURABLE_FIELDS`/`ConfigPatch`), just informational, so
  `POST /v1/jobs`'s docs can point at the live source of truth instead of a hardcoded
  number that would drift out of sync the next time someone changes `Settings`.
- **Top-level description** covers the quickstart (submit → poll → fetch result/docx),
  explains *why* the job endpoints are async (the same concurrency-cap reasoning from
  Phase 7, restated where a newcomer reading `/docs` cold will actually see it), states
  the local-only/no-telemetry property explicitly, and points at the `Problem` schema
  for error shape.

**The review pass, done for real, with an honest caveat:** no interactive browser tool
was available in this environment, so "opening `/docs`" was done by starting the actual
server, fetching the rendered `/docs` HTML (confirmed `200`) and the generated
`/openapi.json`, then walking every path/schema programmatically checking for missing
summaries, descriptions, tags, and examples — the same defects a human skim would catch
(three routes with a summary but no description), just found by a script instead of an
eyeball. This is a narrower form of "review" than actually reading the rendered Swagger
UI end to end, and is recorded here as a real scope limitation rather than a plain "done".

**Tests:** 3 new (`TestProblemDetails`, plus a `GET /v1/config` upload-constraints check) —
245 unit tests total, still 3 integration tests (nothing about a docs-only phase needed a
new one).

---

## Phase 9 — Streamlit demo ✅ done

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

### Implementation notes (completed 2026-09-25, on `dev`)

Built as `frontend/` (outside `src/transcript_task/` deliberately -- this is an app
you run, not library code, same reasoning as `scripts/`): `app.py` (the Streamlit
UI), `api_client.py` (a thin `httpx` wrapper, the only thing that talks to the
API), `helpers.py` (every piece of non-Streamlit logic: stage-to-progress mapping,
raw/refined selection, the sensitivity banner rule, duration formatting).

- **"Clean community template" honestly wasn't literally forked** — this
  environment has no way to fetch one. Built a small, idiomatic single-file app
  directly instead (upload → primary action → poll → result → reset), which serves
  the plan's actual intent ("keep it minimal") the same way; recorded here rather
  than silently presented as if a template had been used.
- **Never imports `transcript_task.pipeline`** — `api_client.py` is the only path
  to the backend, over HTTP, so the demo genuinely exercises the Phase 7/8 service
  rather than being a second way to invoke the library.
- **Sensitivity banner matches the plan's literal wording** ("when the summary
  flags `high`"), not `docx_writer.py`'s own banner rule (Phase 3), which also
  covers `medium`. Deliberate, and called out in `helpers.py`'s docstring rather
  than silently drifting from the spec.
- **Polling** is a plain `time.sleep(1.5); st.rerun()` loop while status isn't
  terminal — no extra polling/autorefresh package added, in keeping with "keep it
  minimal."
- **API errors never reach the user as a traceback:** every `ApiClient` call
  extracts a readable message from the API's RFC 7807 `Problem` body (Phase 8)
  and raises `ApiError`; `app.py`'s `_api_call` wrapper logs the real exception to
  `data/logs/streamlit.log` and shows only the short message via `st.error`.

**A real bug found and fixed while testing this, not a synthetic edge case:**
the sidebar's default API URL (`http://localhost:8000`) happened to collide with
an unrelated local project's own server on this development machine. That server
answered with a `200` and a JSON body — just not one shaped like this API's
`/health` response — and the sidebar crashed with a bare `KeyError: 'asr_model'`
instead of failing gracefully. The API layer's own error handling (RFC 7807
extraction) never even triggered, because from `httpx`'s perspective the request
had *succeeded*; the bug was trusting an unfamiliar 200 response's shape.
Fixed by validating the response shape before reading from it, with a warning
("doesn't look like the transcript_task API — check the URL") instead of either
a crash or a silent misread. This is exactly the kind of gap real-data testing
was meant to catch: nothing about the code review would have surfaced "what if
we hit a stranger's server," only actually running it against this machine's
real network state did.

Also caught, for the same reason, a test-fragility issue: the first version of
the headless smoke test relied on the *absence* of anything at
`localhost:8000` to exercise the "unreachable" path — true on most machines,
false on this one. Fixed by pinning tests to an explicit high port
(`127.0.0.1:59999`) instead of relying on ambient machine state, and adding a
dedicated regression test for the KeyError bug itself (a mocked `200` with an
unrelated JSON shape).

**Tests:** 25 pure-helper tests, 9 `api_client` tests (RFC 7807 extraction,
multipart submission shape, unreachable-host handling — all against real
`httpx.Response` objects, no mock transport needed), 5 headless `AppTest` smoke
tests (renders without raising, upload screen is the default view, the
Transcribe button is disabled with no file selected, the sidebar degrades
instead of crashing on both an unreachable host and an unexpected-but-successful
response). Verified beyond the test suite too: booted the real API and pointed
a real `AppTest` run at it directly — confirmed the sidebar's "API reachable",
correct model names, and the audio-extension list all come through the real
`/health`/`/v1/config` calls, not fakes; separately ran `streamlit run
frontend/app.py` as an actual server process and confirmed it serves `200`.
284 unit tests total project-wide.

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
| 10 | **MLflow file-store opt-out kept, DB backend not adopted.** MLflow 3.x refuses `file:./mlruns` unless `MLFLOW_ALLOW_FILE_STORE` is set (see Phase 6). Set programmatically in `eval/mlflow_sink.py` rather than migrating to SQLite, since decision #5's reasoning (single-user local tool, no server) still holds — the DB backend solves a multi-writer problem this project doesn't have. |
| 11 | **Regression threshold: 0.02 absolute** (2 percentage points of WER/CER/SemDist), built in Phase 6 with no objection raised. Configurable via `compare --threshold`; a metric missing from either compared run is skipped rather than flagged, so comparing runs of different tiers never fails spuriously. |
| 12 | **Second benchmark dataset: Common Voice pt via `fsicoli/common_voice_17_0`** (CC0-1.0 community mirror), not the official `mozilla-foundation` org's repo — that one requires a loading script `datasets` 5.x can no longer run at all. Requested by the user explicitly to test whether refine's WER-erosion finding was an artifact of FLEURS' unusually clean audio (it wasn't — see Phase 6's second follow-up). |
| 13 | **Refine quality guard: fail soft and visible, not silent, and not a hard abort.** A rejected refine attempt falls back to the raw transcript (reusing summarize/docx's existing fallback for a missing `refined_transcript`) but is never thrown away — kept in state, shown in the docx as a labeled appendix with the specific numbers that triggered rejection, so a reviewer can judge the guard's call. Thresholds (`refine_min_content_recall=0.5`, `refine_min/max_length_ratio=0.3/2.5`) are deliberately generous and explicitly documented as uncalibrated (no genuinely spontaneous-disfluent-speech dataset exists yet to calibrate against) — built to catch catastrophic failures (truncation, runaway generation), not to flag normal hesitation-removal shrinkage. |
| 14 | **Refine prompt tuning approved and executed** (2026-09-25), using the Phase 6 eval harness (both FLEURS and Common Voice `quick` tiers) as the objective measure rather than judgement calls — see Phase 6's third follow-up. |
| 15 | **API: `refine` is a per-job, user-facing toggle; both transcripts are always returned when it runs** (2026-09-25). `refine=false` skips the stage entirely (raw only, faster); `refine=true` (default) returns `raw_transcript` **and** `refined_transcript` together, never just one — the API must not regress the guarantee the pipeline state and docx already provide today. See Phase 7. |
| 16 | **`refine-pt-v2` is the default refine prompt** (2026-09-25), replacing v1. Measured, not assumed: cut WER regressions from 7/30→2/30 (Common Voice) and 13/30→6/30 (FLEURS), with wer_refined dropping 25-41% relative on both. `refine-pt-v1` stays available by id (`TRANSCRIPT_REFINE_PROMPT_ID=refine-pt-v1`) for comparison or rollback. |
| 17 | **Config mutation: versioned per job, not rejected while jobs are in flight** (2026-09-25, Phase 7). The plan offered both; per-job versioning was free once every job got its own `Settings` copy anyway (needed regardless, for isolated tmp/output dirs), and it's strictly more useful than blocking a legitimate config change for however long a job takes. |
| 18 | **DELETE on a running job returns 409, not a fake success** (2026-09-25, Phase 7). Python threads running the ASR/LLM calls can't be preempted and have no cancellation hook; silently "succeeding" at canceling a running job would misrepresent what actually happened. A still-queued job can be canceled outright. |
| 19 | **API job/tmp directories must live under `PROJECT_ROOT`** (2026-09-25, Phase 7) — not a preference, a hard constraint: `pipeline.py`'s stage functions store paths via `.relative_to(PROJECT_ROOT)`. `Settings.api_jobs_dir` defaults to `PROJECT_ROOT / "data" / "api_jobs"` accordingly. |
| 20 | **Frontend written from scratch, not forked from a community template** (2026-09-25, Phase 9) — this environment has no way to fetch an external template; a small, idiomatic single-file app serves the plan's actual intent ("keep it minimal") equivalently. Recorded rather than silently presented as if a template had been used. |
| 21 | **Sensitivity banner shown only for `high`, matching the plan's literal wording** (2026-09-25, Phase 9), not `docx_writer.py`'s own banner rule (Phase 3), which also covers `medium`. A deliberate, narrower scope for the demo, not a regression. |

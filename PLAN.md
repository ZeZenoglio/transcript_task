# Implementation Plan — transcript_task

From a working local script to an evaluated, served, tested product.

**Status:** Phases 0–3 complete (2026-09-24, on `dev`). Phases 4-12 pending. All open
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
| Works | `extract → normalize → transcribe → refine → summarize → docx`, 9 real recordings, JSON checkpointing |
| Models | `mlx-whisper` large-v3-turbo (ASR) · `qwen3.5:9b` via Ollama, `think=False` (refine + summarize) |
| Measured | 13:35 audio → 77 s ASR + 202 s refine + ~110s summarize on an M4 base |
| Tests | 33 unit tests (`uv run pytest`) + 1 integration test against real Ollama |
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

## Phase 3b — Stable transcript IDs & PII-aware summaries

Raised in review, not in the original plan: filenames should carry a stable
identifier rather than being a slug tied only to a title, and a title/description
generated by an LLM can itself leak a name or other identifying detail even when
the underlying transcript is never redacted (nor should it be -- the reviewer's
own recording stays intact; this is only about what a *summary* echoes into a
filename or document metadata that might travel more casually than the full doc).

### Stable transcript IDs

- A `transcript_id` is assigned once per source file, at extract time, and is
  **stable across `--force` re-runs** -- reprocessing a recording doesn't give it
  a new identity, only a new result. Stored as `state["items"][key]["transcript_id"]`.
- This is deliberately the *same* identifier Phase 7's API will expose as `job_id`
  and use as the SQLite `jobs` primary key -- introducing it now avoids retrofitting
  every downstream consumer later.
- **Proposed format:** an 8-character random hex id (`secrets.token_hex(4)`) --
  short, filesystem-friendly, effectively collision-free at this tool's personal
  scale (2^32 space). Deliberately *not* time-sortable by construction; Phase 7's
  SQLite row carries a real `created_at` column for that, so the ID itself doesn't
  need to. A ULID (26 chars, ordered, no new heavy dependency but one more small
  package) is the alternative if sorting a folder of `.docx` files by filename
  should also sort them by creation time -- open call, default to hex unless told
  otherwise.
- **Filename becomes** `<transcript_id>_<slug>.docx` (or `<transcript_id>.docx`
  with no summary yet), replacing `<slug>__<original-stem>.docx` from Phase 3.
  The original filename is **dropped from the filename itself** but stays fully
  traceable through three independent paths: the docx's own provenance header
  (already prints "Ficheiro de origem: <original file>"), the `transcripts.json`
  record keyed by the same id, and later the SQLite row. Open call: if losing the
  original stem from the filename itself is a step too far for quick folder
  browsing, it can go back in as a third segment
  (`<transcript_id>_<slug>__<stem>.docx`) at no real cost.
- `docx_filename()` and `stage_docx` change accordingly; existing tests for
  filename collision behaviour get rewritten against the new scheme rather than
  dropped, since the underlying guarantee (two recordings never collide) still
  needs to hold.

### PII-aware summaries

Two layers, not one -- a prompt instruction alone is a soft control an LLM can
ignore under real content pressure:

1. **Prompt-level:** both summarize templates (`prompts.py`) gain an explicit
   instruction to avoid embedding full names, phone numbers, addresses, or ID/tax
   numbers in `title`, `description`, or `topics`; refer to people by role or
   relationship (`"a caller"`, `"the client"`, `"a family member"`) instead. This
   applies only to the summary fields -- the transcript itself is never touched,
   matching the review comment's own scoping.
2. **Deterministic safety net:** a lightweight local NER pass over the generated
   `title`/`description`/`topics` before they reach a filename slug or a docx
   core property, redacting detected person names (and candidate phone/ID-number
   patterns via regex) to a placeholder. Applied only to that metadata surface,
   never to the transcript body -- consistent with "no content lost to the final
   reviewer" from the Phase 3 review. Needs a short research spike (same rigor as
   Phase 1) to pick the actual tool: candidates are spaCy's `pt_core_news_sm`/`lg`
   (small, fast, local, decent PERSON/LOC recall) vs. a HF token-classification
   model vs. regex-only for structured PII (phone/ID numbers) layered on top of
   whichever NER choice handles names.
3. **Config toggle:** `settings.anonymize_metadata: bool = True` (on by default,
   since it's a safety net, not a feature someone opts into).

**Tests:** unit tests with fabricated PII-laden titles/descriptions asserting
redaction; new transcript-id/filename tests covering the same collision guarantee
as Phase 3's, now under the new scheme; an `integration`-marked test if the chosen
NER approach needs a model download.

**Exit:** every generated filename carries a stable id; a title/description
containing a fabricated name in a test fixture comes back redacted in the docx
metadata and filename slug, unredacted in the transcript body.

---

## Phase 4 — Public benchmark dataset

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
  `manifest.jsonl` of `{audio_path, ground_truth, duration, split}`.
- Pin the version/revision and record a checksum for reproducibility.
- `data/` is gitignored; ship `scripts/fetch_dataset.py` + a small committed smoke
  fixture (3–5 clips) so tests run without the full download.

**The committed fixture set is deliberately diversified, not just the first N
clips alphabetically** -- this is what closes the gap the Phase 3 review raised:
tests and eval should stress real audio, not only fakes. Concretely:
- a mix of **durations** (shortest and longest available in the split, not just
  average-length clips) -- long-form audio is exactly where Phase 2 found
  Whisper's repetition-loop hallucination, and a fixture set of only short clips
  would never exercise that path;
- a mix of **formats**: FLEURS itself ships one canonical format, so 2-3 fixture
  clips get re-encoded with ffmpeg into `.mp3`/`.m4a`/`.opus` copies purely to
  exercise `audio.py`'s normalize logic across codecs, the way the original
  personal test set (5 `.m4a` + 4 `.opus`) did by accident.
- This same set is what Phase 10's unit/integration tests use for anything
  touching real transcript content (length, hallucination-proneness, format
  handling) -- synthetic sine-wave WAVs stay useful for pure-plumbing tests
  ("does normalize resample 48kHz to 16kHz") but can't stand in for real speech.
  One fixture set, two consumers (eval harness + test suite), per the Phase 3
  review discussion.

**Exit:** `uv run python scripts/fetch_dataset.py` produces a valid manifest; a
committed fixture subset exists for CI, diversified by duration and format as above.

---

## Phase 5 — Retire the private test data ✅ approved

Approved 2026-09-24 (decision #1): the source recordings are backed up externally
and are to be removed from this machine; the repo generalises to a plain
speech-to-text utility with no tie to any particular recording set. Only run this
once Phase 4 is green and the eval harness (Phase 6) can actually run on the public
data instead.

- Delete `Arquivo.zip`, `tmp/`, `output/`.
- Irreversible on this machine — proceed only because the backup exists.
- Verify nothing in git history ever contained them (it won't, if Phase 0 came first —
  which is the main reason Phase 0 comes first).

**Exit:** repo contains no personal audio or transcripts; pipeline still runs
end-to-end on FLEURS data.

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
| LLM ignores the PII-avoidance prompt instruction | Layered with a deterministic NER pass on the metadata surface (Phase 3b) — never rely on prompt compliance alone |
| Fakes-only tests miss real-format/real-content bugs | Already happened once (Phase 3's 255-char docx bug); Phase 4/10 fixtures now diversified with real audio specifically to close this |

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

**Open, from the Phase 3 review (2026-09-24) — proceeding with the stated default,
flag if you'd rather change it:**

| # | Open call | Default I'll build unless told otherwise |
|---|---|---|
| 8 | `transcript_id` format | 8-char random hex (`secrets.token_hex(4)`). Alternative: ULID, if filename-sort-by-creation-time matters more than brevity. |
| 9 | Original filename stem in the new `.docx` name | Dropped from the filename (traceable via provenance header + JSON record instead). Alternative: keep it as a third segment, `<id>_<slug>__<stem>.docx`, if quick folder browsing matters more than a shorter name. |

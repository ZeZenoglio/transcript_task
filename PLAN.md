# Implementation Plan — transcript_task

From a working local script to an evaluated, served, tested product.

**Status:** Phases 0–1 complete (2026-09-24, on `dev`). Phases 2-12 pending. All open
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
| Works | `extract → normalize → transcribe → refine → docx`, 9 real recordings, JSON checkpointing |
| Models | `mlx-whisper` large-v3-turbo (ASR) · `qwen3.5:9b` via Ollama, `think=False` (refine) |
| Measured | 13:35 audio → 77 s ASR + 202 s refine on an M4 base |
| Missing | tests, linting, packaging, API, eval harness, CI, frontend, logging, persistence |
| Repo | **no commits, no remote**, branch `master`, `gh` not authenticated |

The code is currently three flat modules (`config.py`, `pipeline.py`, `main.py`) with
module-level constants read directly by the stage functions. That is fine for a script
and blocks almost everything downstream — hence Phase 2 before the API work.

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

## Phase 2 — Restructure into a package 🔧 enabler

Required before the API can read/modify config or the eval harness can import stages.

```
src/transcript_task/
  __init__.py
  settings.py        # pydantic-settings; env-overridable; replaces config.py constants
  prompts.py         # refine + summarise templates, versioned with an ID
  audio.py           # probe / normalize  (from pipeline.py)
  asr.py             # transcribe stage, behind a Protocol
  refine.py          # LLM cleanup stage
  summarize.py       # Phase 3
  docx_writer.py     # document generation
  pipeline.py        # orchestration only
  db.py              # SQLite run log (Phase 7)
  logging_conf.py    # structured logging (Phase 7)
tests/
```

Key changes:
- Constants → a `Settings` pydantic model, instantiated once and **passed in**, not
  imported. This is what makes the config endpoints and per-run overrides possible.
- ASR and LLM behind thin `Protocol` interfaces, so tests can inject fakes and the
  eval harness can swap models without touching stage code.
- `pipeline.py` keeps its CLI; `main.py` stays the entry point. **No behaviour change.**

**Exit:** `uv run python -m transcript_task.pipeline` reproduces the current run
byte-for-byte on the same inputs; existing `output/transcripts.json` still loads.

---

## Phase 3 — Summarisation stage (title + description)

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

**Exit:** all 9 (or, post-Phase 5, all sample) files get a valid summary; unit tests green.

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

**Exit:** `uv run python scripts/fetch_dataset.py` produces a valid manifest; a
committed fixture subset exists for CI.

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
`config_history`. Gives the frontend history, and makes "did this get slower last
Tuesday" answerable.

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
- Shared fixtures: tiny generated WAVs (ffmpeg sine/silence), fake ASR + fake LLM
  clients, temp settings, temp SQLite, temp MLflow dir.
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
  └─→ 2 ─→ 3 ─→ 4 ─→ [5 gate] ─→ 6 ─→ 7 ─→ 8 ─→ 9
                                         └─→ 10 ─→ 11 ─→ 12
```

Phase 1 is desk work and can run alongside 2–3. Everything from 6 onward depends on the
Phase 2 restructure. Phase 5 is a hard gate requiring your confirmation.

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
| Scope: this is 12 phases | Phases are independently shippable; stop anywhere and have something coherent |

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

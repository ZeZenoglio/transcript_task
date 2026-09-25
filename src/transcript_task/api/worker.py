"""Background job execution: a small `ThreadPoolExecutor`, not a task
queue -- this is a single-machine local tool, and the ASR/LLM calls are
synchronous/CPU-GPU-bound, so they belong in a thread pool rather than an
asyncio task (see PLAN.md's Phase 7 async-model note). Concurrency is
capped low (`Settings.api_concurrency`, default 1) because running Whisper
and a 9B model at once is already the realistic ceiling on a 16GB machine.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from sqlmodel import Session, select

from ..eval.mlflow_sink import DEFAULT_ARTIFACTS_DIR
from ..eval.orchestrator import DatasetNotFetched, run_benchmark_tier
from ..settings import Settings
from .db import BenchmarkRun, BenchmarkStatus, Job, JobStatus, StageTiming, get_engine
from .job_runner import JobStageError, cleanup_job_files, job_settings, run_job

logger = logging.getLogger("transcript_task.api.worker")


class JobWorker:
    """Owns the thread pool and the in-flight Future for every submitted
    job (so DELETE can cancel a still-queued one -- see cancel())."""

    def __init__(self, settings: Settings, *, transcriber=None, chat_model=None) -> None:
        self.settings = settings
        # Fakes injected here flow into every job this worker runs -- used
        # by tests to exercise the full submit -> thread pool -> DB-update
        # path without touching mlx-whisper/Ollama. None (the default) lets
        # run_job's own stage_* calls construct the real ones.
        self.transcriber = transcriber
        self.chat_model = chat_model
        self._executor = ThreadPoolExecutor(
            max_workers=settings.api_concurrency, thread_name_prefix="job-worker"
        )
        self._futures: dict[str, Future] = {}
        self._benchmark_futures: dict[str, Future] = {}

    def submit(self, job_id: str, audio_path: Path, filename: str, refine: bool) -> None:
        future = self._executor.submit(self._run, job_id, audio_path, filename, refine)
        self._futures[job_id] = future

    def submit_benchmark(
        self,
        tag: str,
        *,
        dataset: str,
        tier: str,
        n: int,
        seed: int,
        skip_refine: bool,
        semdist: bool = True,
    ) -> None:
        # Shares the same executor (and its concurrency cap) as real jobs --
        # a benchmark run does the same ASR/LLM work a job does, just many
        # times over, so it competes for the same GPU/RAM headroom and must
        # not run concurrently with real jobs any more than two real jobs
        # should. See job_worker's docstring.
        future = self._executor.submit(
            self._run_benchmark, tag, dataset, tier, n, seed, skip_refine, semdist
        )
        self._benchmark_futures[tag] = future

    def wait_for_benchmark(self, tag: str, timeout: float = 60.0) -> None:
        """Test helper, mirrors wait_for() for jobs."""
        future = self._benchmark_futures.get(tag)
        if future is not None:
            future.result(timeout=timeout)

    def _run_benchmark(
        self,
        tag: str,
        dataset: str,
        tier: str,
        n: int,
        seed: int,
        skip_refine: bool,
        semdist: bool = True,
    ) -> None:
        with Session(get_engine(self.settings)) as session:
            run = session.get(BenchmarkRun, tag)
            if run is None:
                return  # purged before it started
            try:
                result = run_benchmark_tier(
                    self.settings,
                    dataset=dataset,
                    tier=tier,
                    tag=tag,
                    n=n,
                    seed=seed,
                    skip_refine=skip_refine,
                    use_mlflow=False,
                    no_interpretation=True,
                    no_semdist=not semdist,
                    transcriber=self.transcriber,
                    chat_model=self.chat_model,
                )
            except DatasetNotFetched as exc:
                run.status = BenchmarkStatus.failed
                run.error = str(exc)
                run.finished_at = datetime.now(UTC)
                session.add(run)
                session.commit()
                logger.warning("benchmark %s failed: %s", tag, exc)
                return
            except Exception as exc:  # noqa: BLE001
                run.status = BenchmarkStatus.failed
                run.error = f"unexpected error: {exc}"
                run.finished_at = datetime.now(UTC)
                session.add(run)
                session.commit()
                logger.exception("benchmark %s failed unexpectedly", tag)
                return

            run.status = BenchmarkStatus.done
            run.finished_at = datetime.now(UTC)
            run.results_path = f"{DEFAULT_ARTIFACTS_DIR}/{tag}/results.json"
            session.add(run)
            session.commit()
            logger.info(
                "benchmark %s done: %d clips, %d errors", tag, result.n_clips, result.n_errors
            )

    def cancel(self, job_id: str) -> bool:
        """True if the job was still queued and successfully canceled.
        A job already running cannot be interrupted mid-stage -- Python
        threads aren't preemptible, and the ASR/LLM calls have no
        cancellation hook. This is a real limitation, not hidden: DELETE on
        a running job purges its record once it finishes, but can't stop it
        early (see api/README notes in PLAN.md)."""
        future = self._futures.get(job_id)
        return future.cancel() if future else False

    def wait_for(self, job_id: str, timeout: float = 30.0) -> None:
        """Block until a submitted job's thread finishes. Not used by the
        API itself (which is async and polls instead) -- purely a test
        helper so tests don't need arbitrary sleeps to make a submit()
        deterministic before asserting on its result."""
        future = self._futures.get(job_id)
        if future is not None:
            future.result(timeout=timeout)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run(self, job_id: str, audio_path: Path, filename: str, refine: bool) -> None:
        with Session(get_engine(self.settings)) as session:
            job = session.get(Job, job_id)
            if job is None:
                return  # purged before it started

            # Record the config actually in effect now, not whatever was
            # current when the job was submitted -- PATCH /v1/config may
            # have changed it while this job sat queued (see PLAN.md's
            # Phase 7 note: "a job must record the config it ran under").
            from ..prompts import get_summarize_template

            job.asr_model = self.settings.asr_model
            job.llm_model = self.settings.llm_model
            job.refine_prompt_id = self.settings.refine_prompt_id
            job.summarize_prompt_id = get_summarize_template(self.settings.summary_language).id
            session.add(job)
            session.commit()

            def on_stage(status: JobStatus) -> None:
                job.status = status
                job.updated_at = datetime.now(UTC)
                session.add(job)
                session.commit()

            try:
                per_job_settings = job_settings(self.settings, job_id)
                result = run_job(
                    job_id,
                    audio_path,
                    filename,
                    per_job_settings,
                    refine=refine,
                    on_stage=on_stage,
                    transcriber=self.transcriber,
                    chat_model=self.chat_model,
                )
            except JobStageError as exc:
                job.status = JobStatus.failed
                job.error = str(exc)
                job.updated_at = datetime.now(UTC)
                session.add(job)
                session.commit()
                logger.warning("job %s failed at %s: %s", job_id, exc.stage, exc.detail)
                return
            except Exception as exc:  # noqa: BLE001 - a worker thread must never die silently
                job.status = JobStatus.failed
                job.error = f"unexpected error: {exc}"
                job.updated_at = datetime.now(UTC)
                session.add(job)
                session.commit()
                logger.exception("job %s failed unexpectedly", job_id)
                return

            job.status = JobStatus.done
            job.duration_seconds = result.get("duration_seconds")
            job.source_format = result.get("source_format")
            job.raw_transcript = result.get("raw_transcript")
            job.refined_transcript = result.get("refined_transcript")
            job.refine_rejected = result.get("refine_rejected")
            job.summary = result.get("summary")
            job.docx_path = result.get("docx")
            job.updated_at = datetime.now(UTC)
            session.add(job)

            for stage, field in (
                ("transcribe", "asr_seconds"),
                ("refine", "refine_seconds"),
                ("summarize", "summarize_seconds"),
            ):
                seconds = result.get(field)
                if seconds is not None:
                    session.add(StageTiming(job_id=job_id, stage=stage, seconds=seconds))

            session.commit()


def purge_job(settings: Settings, job_id: str) -> bool:
    """Delete a job's DB row, stage timings, and on-disk files. True if a
    row existed to delete."""
    with Session(get_engine(settings)) as session:
        job = session.get(Job, job_id)
        if job is None:
            return False
        timings = session.exec(select(StageTiming).where(StageTiming.job_id == job_id)).all()
        for t in timings:
            session.delete(t)
        session.delete(job)
        session.commit()
    cleanup_job_files(settings, job_id)
    return True

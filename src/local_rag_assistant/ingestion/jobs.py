"""In-memory ingestion job tracking for live progress consumers."""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from local_rag_assistant.ingestion.models import IngestionDocumentRecord
from local_rag_assistant.ingestion.progress import IngestionProgressEvent, IngestionProgressSink

IngestionJobStatus = Literal["running", "completed", "failed"]


class IngestionJobEvent(BaseModel):
    """Stored progress event with job-local sequencing."""

    sequence: int = Field(ge=1)
    job_id: str
    progress: IngestionProgressEvent


class IngestionJobState(BaseModel):
    """Inspectable state for a live or finished ingestion job."""

    job_id: str
    status: IngestionJobStatus
    paths: list[str] = Field(default_factory=list)
    snapshot: IngestionProgressEvent | None = None
    emitted_events: int = Field(default=0, ge=0)
    results: list[IngestionDocumentRecord] = Field(default_factory=list)
    error_message: str | None = None


@dataclass
class _StoredJob:
    paths: list[str]
    status: IngestionJobStatus = "running"
    snapshot: IngestionProgressEvent | None = None
    events: list[IngestionJobEvent] = field(default_factory=list)
    results: list[IngestionDocumentRecord] = field(default_factory=list)
    error_message: str | None = None
    condition: threading.Condition = field(default_factory=threading.Condition)


class IngestionJobNotFoundError(LookupError):
    """Raised when an ingestion job id is unknown."""


class InMemoryIngestionJobStore:
    """Track ingestion job snapshots and replayable progress events in memory."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, _StoredJob] = {}

    def create_job(self, paths: Sequence[str]) -> str:
        job_id = uuid4().hex
        with self._lock:
            self._jobs[job_id] = _StoredJob(paths=[str(path) for path in paths])
        return job_id

    def sink_for(self, job_id: str) -> IngestionProgressSink:
        self._require_job(job_id)
        return _IngestionJobProgressSink(self, job_id)

    def append_event(self, job_id: str, event: IngestionProgressEvent) -> IngestionJobEvent:
        job = self._require_job(job_id)
        with job.condition:
            stored = IngestionJobEvent(
                sequence=len(job.events) + 1,
                job_id=job_id,
                progress=event,
            )
            job.events.append(stored)
            job.snapshot = event
            if event.event == "completed":
                job.status = "completed"
                job.error_message = None
            elif event.event == "failed":
                job.status = "failed"
            job.condition.notify_all()
            return stored

    def set_results(self, job_id: str, results: Sequence[IngestionDocumentRecord]) -> None:
        job = self._require_job(job_id)
        with job.condition:
            job.results = list(results)
            if job.snapshot is not None and job.snapshot.event == "completed":
                job.status = "completed"
                job.error_message = None
            job.condition.notify_all()

    def mark_failed(self, job_id: str, error_message: str) -> None:
        job = self._require_job(job_id)
        with job.condition:
            if job.snapshot is None or job.snapshot.event not in {"failed", "completed"}:
                job.events.append(
                    IngestionJobEvent(
                        sequence=len(job.events) + 1,
                        job_id=job_id,
                        progress=_failed_snapshot(job.snapshot, error_message),
                    )
                )
                job.snapshot = job.events[-1].progress
            job.status = "failed"
            job.error_message = error_message.strip() or error_message
            job.condition.notify_all()

    def get_state(self, job_id: str) -> IngestionJobState:
        job = self._require_job(job_id)
        with job.condition:
            return IngestionJobState(
                job_id=job_id,
                status=job.status,
                paths=list(job.paths),
                snapshot=job.snapshot,
                emitted_events=len(job.events),
                results=list(job.results),
                error_message=job.error_message,
            )

    def stream_events(self, job_id: str, *, after_sequence: int = 0) -> Iterator[IngestionJobEvent]:
        job = self._require_job(job_id)
        next_sequence = max(after_sequence + 1, 1)
        while True:
            event_to_yield: IngestionJobEvent | None = None
            with job.condition:
                while len(job.events) < next_sequence and job.status == "running":
                    job.condition.wait(timeout=0.1)

                if len(job.events) >= next_sequence:
                    event_to_yield = job.events[next_sequence - 1]
                    next_sequence += 1
                elif job.status != "running":
                    break

            if event_to_yield is not None:
                yield event_to_yield

    def _require_job(self, job_id: str) -> _StoredJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise IngestionJobNotFoundError(f"Ingestion job {job_id} not found")
        return job


class _IngestionJobProgressSink:
    def __init__(self, store: InMemoryIngestionJobStore, job_id: str) -> None:
        self._store = store
        self._job_id = job_id

    def emit(self, event: IngestionProgressEvent) -> None:
        self._store.append_event(self._job_id, event)


def _failed_snapshot(snapshot: IngestionProgressEvent | None, error_message: str) -> IngestionProgressEvent:
    if snapshot is None:
        return IngestionProgressEvent(event="failed", error_message=error_message)
    return snapshot.model_copy(update={"event": "failed", "error_message": error_message})


__all__ = [
    "IngestionJobEvent",
    "IngestionJobNotFoundError",
    "IngestionJobState",
    "IngestionJobStatus",
    "InMemoryIngestionJobStore",
]

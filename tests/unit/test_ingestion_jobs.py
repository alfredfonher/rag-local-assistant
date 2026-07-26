from __future__ import annotations

import threading
import time

from local_rag_assistant.ingestion.jobs import InMemoryIngestionJobStore
from local_rag_assistant.ingestion.progress import IngestionProgressEvent


def test_job_store_tracks_snapshot_and_replays_events() -> None:
    store = InMemoryIngestionJobStore()
    job_id = store.create_job(["/tmp/library"])

    store.append_event(job_id, IngestionProgressEvent(event="started"))
    store.append_event(
        job_id,
        IngestionProgressEvent(
            event="processing",
            total_files=1,
            discovered_files=1,
            current_path="/tmp/library/guide.md",
            document_id="doc-1",
        ),
    )
    store.append_event(
        job_id,
        IngestionProgressEvent(
            event="completed",
            total_files=1,
            discovered_files=1,
            processed_files=1,
            indexed_files=1,
            percent=100.0,
        ),
    )

    state = store.get_state(job_id)
    replay = list(store.stream_events(job_id))
    resumed = list(store.stream_events(job_id, after_sequence=1))

    assert state.job_id == job_id
    assert state.status == "completed"
    assert state.snapshot is not None
    assert state.snapshot.event == "completed"
    assert state.emitted_events == 3
    assert [event.sequence for event in replay] == [1, 2, 3]
    assert [event.progress.event for event in replay] == ["started", "processing", "completed"]
    assert [event.sequence for event in resumed] == [2, 3]


def test_job_store_stream_waits_for_background_events() -> None:
    store = InMemoryIngestionJobStore()
    job_id = store.create_job(["/tmp/library"])
    received: list[str] = []

    def consume() -> None:
        for event in store.stream_events(job_id):
            received.append(event.progress.event)

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()

    time.sleep(0.05)
    store.append_event(job_id, IngestionProgressEvent(event="started"))
    store.append_event(
        job_id,
        IngestionProgressEvent(
            event="completed",
            total_files=0,
            discovered_files=0,
            processed_files=0,
            indexed_files=0,
            failed_files=0,
            percent=100.0,
        ),
    )
    consumer.join(timeout=1)

    assert received == ["started", "completed"]


def test_job_store_mark_failed_emits_terminal_failed_event() -> None:
    store = InMemoryIngestionJobStore()
    job_id = store.create_job(["/tmp/library"])

    store.append_event(job_id, IngestionProgressEvent(event="started"))
    store.mark_failed(job_id, "embedding runtime unavailable")

    state = store.get_state(job_id)
    replay = list(store.stream_events(job_id))

    assert state.status == "failed"
    assert state.snapshot is not None
    assert state.snapshot.event == "failed"
    assert state.snapshot.error_message == "embedding runtime unavailable"
    assert [event.progress.event for event in replay] == ["started", "failed"]

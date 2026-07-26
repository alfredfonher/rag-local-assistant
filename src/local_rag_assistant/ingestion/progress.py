"""Progress contracts and sinks for ingestion workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field, field_validator

IngestionProgressEventType = Literal[
    "started",
    "discovered",
    "processing",
    "indexed",
    "failed",
    "completed",
]


class IngestionProgressEvent(BaseModel):
    """Progress snapshot emitted while ingesting one or more paths."""

    event: IngestionProgressEventType
    total_files: int = Field(default=0, ge=0)
    discovered_files: int = Field(default=0, ge=0)
    processed_files: int = Field(default=0, ge=0)
    indexed_files: int = Field(default=0, ge=0)
    failed_files: int = Field(default=0, ge=0)
    percent: float = Field(default=0.0, ge=0.0, le=100.0)
    current_path: str | None = Field(default=None)
    document_id: str | None = Field(default=None)
    error_message: str | None = Field(default=None)

    @field_validator("current_path", "document_id", "error_message")
    @classmethod
    def normalize_optional_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class IngestionProgressSink(Protocol):
    """Consumer for progress events emitted by ingestion workflows."""

    def emit(self, event: IngestionProgressEvent) -> None: ...


class NoOpIngestionProgressSink:
    """Ignore progress events when no live consumer is attached."""

    def emit(self, event: IngestionProgressEvent) -> None:
        del event


class InMemoryIngestionProgressSink:
    """Collect progress events for tests and future adapters."""

    def __init__(self) -> None:
        self.events: list[IngestionProgressEvent] = []

    def emit(self, event: IngestionProgressEvent) -> None:
        self.events.append(event)


class IngestionProgressReporter:
    """Track ingestion counts and publish normalized progress snapshots."""

    def __init__(self, sink: IngestionProgressSink | None = None) -> None:
        self._sink = sink or NoOpIngestionProgressSink()
        self._discovered_files = 0
        self._processed_files = 0
        self._indexed_files = 0
        self._failed_files = 0

    def started(self) -> None:
        self._emit("started")

    def discovered(self, path: str | Path) -> None:
        self._discovered_files += 1
        self._emit("discovered", current_path=_normalize_path(path))

    def processing(self, path: str | Path, *, document_id: str) -> None:
        self._emit(
            "processing",
            current_path=_normalize_path(path),
            document_id=document_id,
        )

    def indexed(self, path: str | Path, *, document_id: str) -> None:
        self._processed_files += 1
        self._indexed_files += 1
        self._emit(
            "indexed",
            current_path=_normalize_path(path),
            document_id=document_id,
        )

    def failed(
        self,
        path: str | Path,
        *,
        error_message: str,
        document_id: str | None = None,
        counts_as_processed: bool = True,
    ) -> None:
        if counts_as_processed:
            self._processed_files += 1
        self._failed_files += 1
        self._emit(
            "failed",
            current_path=_normalize_path(path),
            document_id=document_id,
            error_message=error_message,
        )

    def completed(self) -> None:
        self._emit("completed")

    def _emit(
        self,
        event: IngestionProgressEventType,
        *,
        current_path: str | None = None,
        document_id: str | None = None,
        error_message: str | None = None,
    ) -> None:
        total_files = self._discovered_files
        percent = 100.0 if total_files == 0 and event == "completed" else _percent(self._processed_files, total_files)
        self._sink.emit(
            IngestionProgressEvent(
                event=event,
                total_files=total_files,
                discovered_files=self._discovered_files,
                processed_files=self._processed_files,
                indexed_files=self._indexed_files,
                failed_files=self._failed_files,
                percent=percent,
                current_path=current_path,
                document_id=document_id,
                error_message=error_message,
            )
        )


def _normalize_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def _percent(processed_files: int, total_files: int) -> float:
    if total_files <= 0:
        return 0.0
    return round((processed_files / total_files) * 100, 2)


__all__ = [
    "InMemoryIngestionProgressSink",
    "IngestionProgressEvent",
    "IngestionProgressEventType",
    "IngestionProgressReporter",
    "IngestionProgressSink",
    "NoOpIngestionProgressSink",
]

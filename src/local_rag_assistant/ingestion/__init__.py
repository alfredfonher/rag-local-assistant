"""Public ingestion package interface."""

from __future__ import annotations

from typing import Any

from local_rag_assistant.ingestion.extractors import (
    SUPPORTED_EXTENSIONS,
    ExtractionError,
    UnsupportedFileTypeError,
    extract_file,
    extract_markdown_file,
    extract_pdf_file,
    extract_text_file,
)
from local_rag_assistant.ingestion.models import ExtractionResult, IngestionDocumentRecord, IngestionStatus
from local_rag_assistant.ingestion.progress import (
    InMemoryIngestionProgressSink,
    IngestionProgressEvent,
    IngestionProgressEventType,
    IngestionProgressSink,
    NoOpIngestionProgressSink,
)
from local_rag_assistant.ingestion.registry import IngestionRegistry

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "ExtractionError",
    "ExtractionResult",
    "IngestionDocumentRecord",
    "IngestionProgressEvent",
    "IngestionProgressEventType",
    "IngestionProgressSink",
    "IngestionRegistry",
    "IngestionService",
    "IngestionStatus",
    "InMemoryIngestionProgressSink",
    "NoOpIngestionProgressSink",
    "UnsupportedFileTypeError",
    "extract_file",
    "extract_markdown_file",
    "extract_pdf_file",
    "extract_text_file",
]


def __getattr__(name: str) -> Any:
    if name == "IngestionService":
        from local_rag_assistant.ingestion.service import IngestionService

        return IngestionService
    raise AttributeError(name)

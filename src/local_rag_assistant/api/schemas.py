"""HTTP request and response schemas for the management API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from local_rag_assistant.ingestion.models import IngestionDocumentRecord
from local_rag_assistant.ingestion.jobs import IngestionJobState


class IngestRequest(BaseModel):
    """Programmatic ingestion request."""

    paths: list[str] = Field(min_length=1)


class IngestResponse(BaseModel):
    """Summary for ingestion operations."""

    documents: list[IngestionDocumentRecord] = Field(default_factory=list)


class IngestionJobResponse(IngestionJobState):
    """HTTP representation of an ingestion job state."""


class DocumentListResponse(BaseModel):
    """Collection of ingested document records."""

    documents: list[IngestionDocumentRecord] = Field(default_factory=list)


class DeleteDocumentResponse(BaseModel):
    """Deletion result for a document record."""

    document_id: str
    deleted: bool = True


__all__ = [
    "DeleteDocumentResponse",
    "DocumentListResponse",
    "IngestRequest",
    "IngestResponse",
    "IngestionJobResponse",
]

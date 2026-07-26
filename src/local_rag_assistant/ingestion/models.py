"""Normalized models for document ingestion."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

IngestionStatus = Literal["pending", "indexed", "failed"]


class ExtractionResult(BaseModel):
    """Canonical output produced by file extractors."""

    source_path: str = Field(description="Absolute path to the extracted source file")
    file_type: Literal["text", "markdown", "pdf"]
    title: str = Field(default="", description="Best-effort human title for the document")
    content: str = Field(description="Extracted textual content")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_path", "content")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("value must not be empty")
        return value

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        return value.strip()


class IngestionDocumentRecord(BaseModel):
    """Persisted document registry record for ingestion workflows."""

    document_id: str = Field(description="Stable document identifier")
    source_path: str = Field(description="Absolute path to the extracted source file")
    file_type: Literal["text", "markdown", "pdf"]
    title: str = Field(default="", description="Best-effort human title for the document")
    content_sha256: str = Field(description="Content hash used to detect changes")
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: IngestionStatus = Field(default="pending")
    error_message: str | None = Field(default=None)
    created_at: datetime = Field(description="UTC creation timestamp")
    updated_at: datetime = Field(description="UTC last update timestamp")

    @field_validator("document_id", "source_path", "content_sha256")
    @classmethod
    def validate_required_fields(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("value must not be empty")
        return value

    @field_validator("title")
    @classmethod
    def normalize_record_title(cls, value: str) -> str:
        return value.strip()

    @field_validator("error_message")
    @classmethod
    def normalize_error_message(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


__all__ = ["ExtractionResult", "IngestionDocumentRecord", "IngestionStatus"]

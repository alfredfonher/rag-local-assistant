"""Pydantic models shared across the application."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

MAX_DOCUMENT_BYTES = 10 * 1024 * 1024


def _new_id() -> str:
    return str(uuid4())


class ChunkMetadata(BaseModel):
    """Metadata attached to indexed chunks."""

    source: str = Field(description="Relative source path for the original document")
    title: str = Field(default="", description="Document title")
    chunk_index: int = Field(ge=0, description="Chunk position inside the document")
    total_chunks: int | None = Field(default=None, ge=1, description="Optional total chunk count")
    document_hash: str | None = Field(default=None, description="SHA256 hash of the source document")
    indexed_at: datetime | None = Field(default=None, description="When the source was indexed")
    mtime: float | None = Field(default=None, description="Source file modification time")
    extra: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("source must not be empty")
        return value.strip()


class Document(BaseModel):
    """Structured representation of a source document."""

    id: str = Field(default_factory=_new_id)
    source: str
    title: str = ""
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    hash: str = ""
    indexed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("source")
    @classmethod
    def validate_non_empty_source(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("source must not be empty")
        return value.strip()

    @field_validator("content")
    @classmethod
    def validate_content_size(cls, value: str) -> str:
        if len(value.encode("utf-8")) > MAX_DOCUMENT_BYTES:
            raise ValueError("content must not exceed 10MB")
        return value


class Chunk(BaseModel):
    """Single retrievable piece of a document."""

    id: str = Field(default_factory=_new_id)
    document_id: str
    chunk_index: int = Field(ge=0)
    content: str
    metadata: ChunkMetadata
    embedding: list[float] | None = None


class ChunkResult(BaseModel):
    """Ranked search result for an individual chunk."""

    text: str
    document_id: str
    chunk_index: int = Field(ge=0)
    score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Response returned by semantic search."""

    results: list[ChunkResult] = Field(default_factory=list)
    query: str
    total_results: int = Field(ge=0)


class ChatMessage(BaseModel):
    """Structured chat message for LLM interactions."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Chat endpoint request payload."""

    query: str
    top_k: int = Field(default=3)
    system_prompt: str | None = None

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("query must not be empty")
        return value.strip()

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, value: int) -> int:
        if not 1 <= value <= 50:
            raise ValueError("top_k must be between 1 and 50")
        return value


class ChatResponse(BaseModel):
    """Chat response with grounded source snippets."""

    answer: str
    sources: list["ChatSource"] = Field(default_factory=list)
    query: str


class ChatSource(BaseModel):
    """Citation-like source details returned with chat answers."""

    source: str
    chunk_index: int = Field(ge=0)
    excerpt: str
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    title: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexRequest(BaseModel):
    """Programmatic index request."""

    path: str
    recursive: bool = True
    extensions: list[str] = Field(default_factory=lambda: [".md", ".txt"])


class IndexResponse(BaseModel):
    """Summary for indexing operations."""

    status: str
    files_discovered: int = Field(default=0, ge=0)
    files_indexed: int = Field(ge=0)
    chunks_created: int = Field(ge=0)
    skipped: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    files_new: int = Field(default=0, ge=0)
    files_modified: int = Field(default=0, ge=0)
    files_deleted: int = Field(default=0, ge=0)
    deleted_from_index: int = Field(default=0, ge=0)
    errors: list[str] = Field(default_factory=list)


class IndexStats(BaseModel):
    """High-level index statistics."""

    documents: int = Field(ge=0)
    chunks: int = Field(ge=0)
    collection_name: str = "documents"
    last_indexed: datetime | None = None


__all__ = [
    "MAX_DOCUMENT_BYTES",
    "Chunk",
    "ChunkMetadata",
    "ChunkResult",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ChatSource",
    "Document",
    "IndexRequest",
    "IndexResponse",
    "IndexStats",
    "SearchResult",
]

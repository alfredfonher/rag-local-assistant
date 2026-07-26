from __future__ import annotations

import pytest
from pydantic import ValidationError

from local_rag_assistant.models import ChatRequest, Chunk, ChunkMetadata, Document, IndexStats


def test_document_rejects_empty_source() -> None:
    with pytest.raises(ValidationError):
        Document(source="   ", title="Title", content="content")


def test_document_rejects_content_over_10mb() -> None:
    oversized = "a" * ((10 * 1024 * 1024) + 1)
    with pytest.raises(ValidationError):
        Document(source="notes/doc.md", title="Large", content=oversized)


def test_chunk_allows_optional_embedding() -> None:
    chunk = Chunk(
        document_id="doc-1",
        chunk_index=0,
        content="hello world",
        metadata=ChunkMetadata(source="notes/doc.md", title="Doc", chunk_index=0),
    )
    assert chunk.embedding is None


@pytest.mark.parametrize("value", [0, 51])
def test_chat_request_validates_top_k_bounds(value: int) -> None:
    with pytest.raises(ValidationError):
        ChatRequest(query="What is RAG?", top_k=value)


def test_index_stats_fields_are_exposed() -> None:
    stats = IndexStats(documents=4, chunks=12, collection_name="documents")
    assert stats.documents == 4
    assert stats.chunks == 12
    assert stats.collection_name == "documents"

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_rag_assistant.models import Chunk, ChunkMetadata
from local_rag_assistant.vectorstore import VectorStore


class _FakeCollection:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, object]] = {}

    def upsert(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, object]],
        embeddings: list[list[float]],
    ) -> None:
        for record_id, document, metadata, embedding in zip(ids, documents, metadatas, embeddings, strict=False):
            self.records[record_id] = {
                "id": record_id,
                "document": document,
                "metadata": metadata,
                "embedding": embedding,
            }

    def query(self, *, query_embeddings: list[list[float]], n_results: int, where: dict[str, object] | None = None) -> dict[str, list[list[object]]]:
        del query_embeddings
        items = list(self.records.values())
        if where:
            items = [item for item in items if all(item["metadata"].get(key) == value for key, value in where.items())]
        items = items[:n_results]
        return {
            "documents": [[item["document"] for item in items]],
            "metadatas": [[item["metadata"] for item in items]],
            "distances": [[0.1 for _ in items]],
        }

    def delete(self, *, where: dict[str, object]) -> None:
        doomed = [
            record_id
            for record_id, item in self.records.items()
            if all(item["metadata"].get(key) == value for key, value in where.items())
        ]
        for record_id in doomed:
            self.records.pop(record_id, None)

    def get(self, *, where: dict[str, object] | None = None, include: list[str] | None = None) -> dict[str, list[object]]:
        del include
        items = list(self.records.values())
        if where:
            items = [item for item in items if all(item["metadata"].get(key) == value for key, value in where.items())]
        return {
            "ids": [item["id"] for item in items],
            "documents": [item["document"] for item in items],
            "metadatas": [item["metadata"] for item in items],
        }

    def count(self) -> int:
        return len(self.records)


class _FakeClient:
    collections: dict[str, _FakeCollection] = {}

    def __init__(self, path: str) -> None:
        self.path = path

    def get_or_create_collection(self, name: str, metadata: dict[str, object] | None = None) -> _FakeCollection:
        del metadata
        return self.collections.setdefault(name, _FakeCollection())

    def delete_collection(self, name: str) -> None:
        if name not in self.collections:
            raise ValueError(name)
        self.collections.pop(name)


@pytest.fixture(autouse=True)
def fake_chromadb(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeClient.collections = {}
    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(PersistentClient=_FakeClient))


def _chunk(document_id: str, chunk_index: int, content: str, *, source: str = "notes/doc.md") -> Chunk:
    return Chunk(
        id=f"{document_id}:{chunk_index}",
        document_id=document_id,
        chunk_index=chunk_index,
        content=content,
        metadata=ChunkMetadata(
            source=source,
            title="Doc",
            chunk_index=chunk_index,
            total_chunks=2,
            document_hash="hash-1",
            indexed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            mtime=123.0,
            extra={"path": source, "file_type": "markdown", "checksum": "hash-1"},
        ),
    )


def test_upsert_search_and_stats_round_trip(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "chroma")
    chunks = [_chunk("doc-1", 0, "alpha"), _chunk("doc-1", 1, "beta")]

    store.upsert(chunks, [[0.1, 0.2], [0.3, 0.4]])
    results = store.search([0.1, 0.2], top_k=2)
    stats = store.get_stats()

    assert [result.text for result in results] == ["alpha", "beta"]
    assert results[0].metadata["path"] == "notes/doc.md"
    assert stats.documents == 1
    assert stats.chunks == 2
    assert stats.collection_name == "documents"


def test_delete_by_document_id_removes_document_chunks(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "chroma")
    store.upsert([_chunk("doc-1", 0, "alpha"), _chunk("doc-2", 0, "beta")], [[0.1], [0.2]])

    store.delete_by_document_id("doc-1")

    remaining = store.get_document_chunks("doc-2")
    assert [chunk.content for chunk in remaining] == ["beta"]
    assert store.get_document_chunks("doc-1") == []

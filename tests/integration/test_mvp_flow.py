from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from local_rag_assistant.api.main import create_app
from local_rag_assistant.bootstrap import Bootstrap, build_bootstrap
from local_rag_assistant.config import Config
from local_rag_assistant.document_loader import build_document_id


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

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict[str, object] | None = None,
    ) -> dict[str, list[list[object]]]:
        query_embedding = query_embeddings[0]
        items = list(self.records.values())
        if where:
            items = [item for item in items if all(item["metadata"].get(key) == value for key, value in where.items())]
        ranked = sorted(
            items,
            key=lambda item: (
                _squared_distance(query_embedding, item["embedding"]),
                str(item["id"]),
            ),
        )[:n_results]
        return {
            "documents": [[item["document"] for item in ranked]],
            "metadatas": [[item["metadata"] for item in ranked]],
            "distances": [[_squared_distance(query_embedding, item["embedding"]) for item in ranked]],
        }

    def delete(self, *, where: dict[str, object]) -> None:
        doomed = [
            record_id
            for record_id, item in self.records.items()
            if all(item["metadata"].get(key) == value for key, value in where.items())
        ]
        for record_id in doomed:
            self.records.pop(record_id, None)

    def get(
        self,
        *,
        where: dict[str, object] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, list[object]]:
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


class _FakePersistentClient:
    collections_by_path: dict[str, dict[str, _FakeCollection]] = {}

    def __init__(self, path: str) -> None:
        self.path = path

    def get_or_create_collection(self, name: str, metadata: dict[str, object] | None = None) -> _FakeCollection:
        del metadata
        collections = self.collections_by_path.setdefault(self.path, {})
        return collections.setdefault(name, _FakeCollection())

    def delete_collection(self, name: str) -> None:
        collections = self.collections_by_path.setdefault(self.path, {})
        if name not in collections:
            raise ValueError(name)
        collections.pop(name)


class _FakeEmbeddingModel:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        del batch_size
        return [self._encode_text(text) for text in texts]

    def encode_single(self, text: str) -> list[float]:
        return self._encode_text(text)

    def encode_query(self, text: str) -> list[float]:
        return self._encode_text(text)

    def _encode_text(self, text: str) -> list[float]:
        vector = [0.0] * 16
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            bucket = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % len(vector)
            vector[bucket] += 1.0
        return vector


class _FakeLLM:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.messages: list[object] = []

    def chat(self, messages: list[object]) -> str:
        self.messages = list(messages)
        return "Stub grounded answer"


@pytest.fixture(autouse=True)
def fake_chromadb(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakePersistentClient.collections_by_path = {}
    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(PersistentClient=_FakePersistentClient))


def test_ingest_registry_search_and_chat_flow(tmp_path: Path) -> None:
    document_path = _write_document(
        tmp_path / "vault" / "alpha.md",
        "# Alpha\n\nAlpha beacon explains grounded retrieval for the MVP flow.",
    )
    bootstrap, client = _build_runtime(tmp_path)

    ingest = client.post("/api/v1/ingest", json={"paths": [str(document_path)]})
    documents = client.get("/api/v1/documents")
    search = client.get("/search", params={"query": "alpha beacon grounded retrieval", "top_k": 3})
    chat = client.post("/chat", json={"query": "Summarize the alpha beacon note", "top_k": 2})

    assert ingest.status_code == 200
    assert documents.status_code == 200
    assert search.status_code == 200
    assert chat.status_code == 200

    ingested_record = ingest.json()["documents"][0]
    assert ingested_record["document_id"] == build_document_id(document_path)
    assert ingested_record["status"] == "indexed"

    registry_record = documents.json()["documents"][0]
    assert registry_record["document_id"] == ingested_record["document_id"]
    assert registry_record["source_path"] == str(document_path)
    assert registry_record["status"] == "indexed"

    search_body = search.json()
    assert search_body["total_results"] == 1
    assert search_body["results"][0]["document_id"] == ingested_record["document_id"]
    assert search_body["results"][0]["metadata"]["source"] == str(document_path)

    chat_body = chat.json()
    assert chat_body["answer"] == "Stub grounded answer"
    assert chat_body["sources"][0]["source"] == str(document_path)
    assert any(
        "Alpha beacon explains grounded retrieval" in getattr(message, "content", "")
        for message in bootstrap.llm().messages
    )


def test_delete_document_removes_registry_and_vectorstore_visibility(tmp_path: Path) -> None:
    keep_path = _write_document(
        tmp_path / "vault" / "keep.md",
        "# Keep\n\nKeep beacon stays searchable after deletion.",
    )
    remove_path = _write_document(
        tmp_path / "vault" / "remove.md",
        "# Remove\n\nRemove beacon should disappear from the index.",
    )
    bootstrap, client = _build_runtime(tmp_path)

    ingest = client.post("/api/v1/ingest", json={"paths": [str(keep_path), str(remove_path)]})
    records = {item["source_path"]: item for item in ingest.json()["documents"]}
    remove_document_id = records[str(remove_path)]["document_id"]
    keep_document_id = records[str(keep_path)]["document_id"]

    delete = client.delete(f"/api/v1/documents/{remove_document_id}")
    documents = client.get("/api/v1/documents")
    deleted_search = client.get("/search", params={"query": "remove beacon disappear", "top_k": 3})
    kept_search = client.get("/search", params={"query": "keep beacon searchable", "top_k": 3})
    stats = client.get("/stats")

    assert delete.status_code == 200
    assert delete.json() == {"document_id": remove_document_id, "deleted": True}
    assert documents.status_code == 200
    assert [item["document_id"] for item in documents.json()["documents"]] == [keep_document_id]
    assert deleted_search.status_code == 200
    assert all(result["document_id"] != remove_document_id for result in deleted_search.json()["results"])
    assert kept_search.status_code == 200
    assert kept_search.json()["results"][0]["document_id"] == keep_document_id
    assert stats.status_code == 200
    assert stats.json()["documents"] == 1
    assert bootstrap.vectorstore().get_document_chunks(remove_document_id) == []


def test_reindex_document_replaces_indexed_state_cleanly(tmp_path: Path) -> None:
    document_path = _write_document(
        tmp_path / "vault" / "reindex.md",
        "# Reindex\n\nOriginal beacon content before reindex.",
    )
    bootstrap, client = _build_runtime(tmp_path)

    ingest = client.post("/api/v1/ingest", json={"paths": [str(document_path)]})
    initial_record = ingest.json()["documents"][0]
    document_id = initial_record["document_id"]

    _write_document(
        document_path,
        "# Reindex\n\nUpdated beacon content after reindex with fresh wording.",
    )

    reindex = client.post(f"/api/v1/documents/{document_id}/reindex")
    old_search = client.get("/search", params={"query": "original beacon before", "top_k": 3})
    new_search = client.get("/search", params={"query": "updated beacon fresh wording", "top_k": 3})

    assert reindex.status_code == 200
    assert reindex.json()["document_id"] == document_id
    assert reindex.json()["status"] == "indexed"
    assert reindex.json()["content_sha256"] != initial_record["content_sha256"]

    assert old_search.status_code == 200
    assert new_search.status_code == 200
    assert new_search.json()["results"][0]["document_id"] == document_id

    chunks = bootstrap.vectorstore().get_document_chunks(document_id)
    assert len(chunks) == 1
    assert "Updated beacon content after reindex with fresh wording." in chunks[0].content
    assert all("Original beacon content before reindex." not in chunk.content for chunk in chunks)


def _build_runtime(tmp_path: Path) -> tuple[Bootstrap, TestClient]:
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    embedding_model_path = models_dir / "embed.gguf"
    llm_model_path = models_dir / "llm.gguf"
    embedding_model_path.write_text("stub", encoding="utf-8")
    llm_model_path.write_text("stub", encoding="utf-8")

    config = Config(
        chroma_db_path=tmp_path / "data" / "chroma",
        ingestion_data_path=tmp_path / "data" / "ingestion",
        ingestion_registry_path=tmp_path / "data" / "ingestion" / "registry.sqlite3",
        ingestion_storage_path=tmp_path / "data" / "ingestion" / "storage",
        embedding_model_path=embedding_model_path,
        llm_model_path=llm_model_path,
        default_top_k=3,
    )
    bootstrap = build_bootstrap(
        config,
        embedding_factory=_FakeEmbeddingModel,
        llm_factory=_FakeLLM,
    )
    return bootstrap, TestClient(create_app(bootstrap=bootstrap))


def _write_document(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path.resolve()


def _squared_distance(left: list[float], right: object) -> float:
    right_vector = list(right)
    return sum((left_value - right_value) ** 2 for left_value, right_value in zip(left, right_vector, strict=False))

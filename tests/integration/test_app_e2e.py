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


@pytest.fixture
def runtime_client(tmp_path: Path) -> tuple[Bootstrap, TestClient]:
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


def test_main_routes_load_with_real_app_wiring(runtime_client: tuple[Bootstrap, TestClient]) -> None:
    _, client = runtime_client

    root = client.get("/", follow_redirects=False)
    overview = client.get("/overview")
    ingest = client.get("/ingest")
    documents = client.get("/documents")
    search = client.get("/search", headers={"accept": "text/html"})
    chat = client.get("/chat")
    api_documents = client.get("/api/v1/documents")

    assert root.status_code == 307
    assert root.headers["location"] == "/overview"

    assert overview.status_code == 200
    assert "Document Operations" in overview.text

    assert ingest.status_code == 200
    assert "Prepare document imports" in ingest.text

    assert documents.status_code == 200
    assert "Tracked sources" in documents.text

    assert search.status_code == 200
    assert "Search indexed content" in search.text

    assert chat.status_code == 200
    assert "Ask against the local index" in chat.text

    assert api_documents.status_code == 200
    assert api_documents.json() == {"documents": []}


def test_fixture_documents_ingest_through_ui_and_load_via_api_and_pages(
    runtime_client: tuple[Bootstrap, TestClient],
    ingestion_fixture_paths: list[Path],
) -> None:
    bootstrap, client = runtime_client
    expected_paths = [str(path.resolve()) for path in ingestion_fixture_paths]

    ingest = client.post(
        "/ingest",
        data={"paths": "\n".join(expected_paths)},
        follow_redirects=True,
    )
    api_documents = client.get("/api/v1/documents")
    search_page = client.get(
        "/search",
        params={"query": "Fixture PDF coverage"},
        headers={"accept": "text/html"},
    )
    chat_page = client.get("/chat", params={"query": "Summarize the fixture PDF coverage note"})

    assert ingest.status_code == 200
    assert "Processed 3 path(s): 3 indexed, 0 failed." in ingest.text
    assert "retrieval-handbook.txt" in ingest.text
    assert "system-architecture.md" in ingest.text
    assert "fixture-reference.pdf" in ingest.text

    assert api_documents.status_code == 200
    documents_payload = api_documents.json()["documents"]
    assert len(documents_payload) == 3

    documents_by_path = {item["source_path"]: item for item in documents_payload}
    assert set(documents_by_path) == set(expected_paths)
    assert documents_by_path[expected_paths[0]]["file_type"] == "text"
    assert documents_by_path[expected_paths[1]]["file_type"] == "markdown"
    assert documents_by_path[expected_paths[1]]["title"] == "System Architecture Primer"
    assert documents_by_path[expected_paths[2]]["file_type"] == "pdf"

    for expected_path in expected_paths:
        detail = client.get(f"/api/v1/documents/{documents_by_path[expected_path]['document_id']}")
        assert detail.status_code == 200
        assert detail.json()["source_path"] == expected_path

    assert search_page.status_code == 200
    assert "Fixture PDF coverage" in search_page.text
    assert "fixture-reference.pdf" in search_page.text

    assert chat_page.status_code == 200
    assert "Stub grounded answer" in chat_page.text
    assert "fixture-reference.pdf" in chat_page.text
    assert any(
        "Fixture PDF coverage" in getattr(message, "content", "")
        for message in bootstrap.llm().messages
    )


def _squared_distance(left: list[float], right: object) -> float:
    right_vector = list(right)
    return sum((left_value - right_value) ** 2 for left_value, right_value in zip(left, right_vector, strict=False))

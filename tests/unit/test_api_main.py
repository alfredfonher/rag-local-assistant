from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time

from fastapi.testclient import TestClient

from local_rag_assistant.api.main import create_app
from local_rag_assistant.config import RuntimeConfigurationError
from local_rag_assistant.ingestion.extractors import UnsupportedFileTypeError
from local_rag_assistant.ingestion.progress import IngestionProgressReporter, IngestionProgressSink
from local_rag_assistant.ingestion.models import IngestionDocumentRecord
from local_rag_assistant.models import ChatResponse, ChatSource, ChunkResult, IndexResponse, IndexStats, SearchResult


class _StubOrchestrator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def stats(self) -> IndexStats:
        self.calls.append(("stats", None))
        return IndexStats(documents=3, chunks=9, collection_name="documents")

    def index(self, path: str, *, recursive: bool = True, extensions: list[str] | None = None) -> IndexResponse:
        self.calls.append(("index", {"path": path, "recursive": recursive, "extensions": extensions}))
        return IndexResponse(status="success", files_indexed=1, chunks_created=2)

    def search(self, query: str, *, top_k: int | None = None) -> SearchResult:
        self.calls.append(("search", {"query": query, "top_k": top_k}))
        return SearchResult(
            query=query,
            total_results=1,
            results=[
                ChunkResult(
                    text="alpha",
                    document_id="doc-1",
                    chunk_index=0,
                    score=0.91,
                    metadata={"source": "notes/alpha.md"},
                )
            ],
        )

    def chat(self, question: str, *, top_k: int | None = None, system_prompt: str | None = None) -> ChatResponse:
        self.calls.append(("chat", {"question": question, "top_k": top_k, "system_prompt": system_prompt}))
        return ChatResponse(
            answer="grounded",
            query=question,
            sources=[
                ChatSource(
                    source="notes/alpha.md",
                    chunk_index=0,
                    excerpt="alpha",
                    score=0.91,
                    title="Alpha note",
                    metadata={"source": "notes/alpha.md"},
                )
            ],
        )


class _StubBootstrap:
    class _Config:
        chroma_collection_name = "documents"
        ingestion_registry_path = Path("/tmp/ingestion/registry.sqlite3")
        ingestion_storage_path = Path("/tmp/ingestion/storage")

    def __init__(self, orchestrator: _StubOrchestrator) -> None:
        self.config = self._Config()
        self.orchestrator = orchestrator
        self._registry = _StubRegistry()
        self._service = _StubIngestionService(self._registry)
        self._vectorstore = _StubVectorStore()
        self.flags: list[tuple[bool, bool]] = []

    def build_orchestrator(self, *, require_embedding: bool = True, require_llm: bool = True) -> _StubOrchestrator:
        self.flags.append((require_embedding, require_llm))
        return self.orchestrator

    def registry(self) -> "_StubRegistry":
        return self._registry

    def ingestion_service(self, *, progress_sink: IngestionProgressSink | None = None) -> "_StubIngestionService":
        if progress_sink is None:
            return self._service
        return _StubIngestionService(self._registry, progress_sink=progress_sink)

    def ingestion_job_store(self):
        from local_rag_assistant.ingestion.jobs import InMemoryIngestionJobStore

        if not hasattr(self, "_job_store"):
            self._job_store = InMemoryIngestionJobStore()
        return self._job_store

    def vectorstore(self) -> "_StubVectorStore":
        return self._vectorstore


class _StubRegistry:
    def __init__(self) -> None:
        self.records = {
            "doc-1": _record("doc-1", "/tmp/alpha.md", status="indexed"),
            "doc-2": _record("doc-2", "/tmp/beta.md", status="failed", error_message="parse failed"),
        }
        self.calls: list[tuple[str, object]] = []

    def list(self) -> list[IngestionDocumentRecord]:
        self.calls.append(("list", None))
        return list(self.records.values())

    def get(self, document_id: str) -> IngestionDocumentRecord | None:
        self.calls.append(("get", document_id))
        return self.records.get(document_id)

    def delete(self, document_id: str) -> bool:
        self.calls.append(("delete", document_id))
        return self.records.pop(document_id, None) is not None


class _StubIngestionService:
    def __init__(self, registry: _StubRegistry, *, progress_sink: IngestionProgressSink | None = None) -> None:
        self.registry = registry
        self.progress_sink = progress_sink
        self.calls: list[tuple[str, object]] = []

    def ingest_paths(self, paths: list[str]) -> list[IngestionDocumentRecord]:
        self.calls.append(("ingest_paths", paths))
        reporter = IngestionProgressReporter(self.progress_sink)
        reporter.started()
        records = []
        for index, path in enumerate(paths, start=1):
            document_id = f"ingested-{index}"
            reporter.discovered(path)
            reporter.processing(path, document_id=document_id)
            record = _record(document_id, path, status="indexed")
            self.registry.records[document_id] = record
            records.append(record)
            reporter.indexed(path, document_id=document_id)
        reporter.completed()
        return records

    def ingest_path(self, path: str) -> IngestionDocumentRecord:
        self.calls.append(("ingest_path", path))
        for document_id, record in list(self.registry.records.items()):
            if record.source_path == path:
                updated = record.model_copy(update={"status": "indexed", "error_message": None})
                self.registry.records[document_id] = updated
                return updated
        raise FileNotFoundError(path)


class _StubVectorStore:
    def __init__(self) -> None:
        self.deleted_document_ids: list[str] = []

    def delete_by_document_id(self, document_id: str) -> None:
        self.deleted_document_ids.append(document_id)


def _record(
    document_id: str,
    source_path: str,
    *,
    status: str,
    error_message: str | None = None,
) -> IngestionDocumentRecord:
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return IngestionDocumentRecord(
        document_id=document_id,
        source_path=source_path,
        file_type="markdown",
        title="Document",
        content_sha256="abc123",
        metadata={"source": source_path},
        status=status,
        error_message=error_message,
        created_at=timestamp,
        updated_at=timestamp,
    )


def test_api_routes_delegate_to_orchestrator() -> None:
    orchestrator = _StubOrchestrator()
    bootstrap = _StubBootstrap(orchestrator)
    client = TestClient(create_app(bootstrap=bootstrap))

    health = client.get("/health")
    stats = client.get("/stats")
    index = client.post("/index", json={"path": "/tmp/vault", "recursive": True, "extensions": [".md"]})
    search = client.get("/search", params={"query": "rag", "top_k": 2})
    chat = client.post("/chat", json={"query": "Explain rag", "top_k": 2})

    assert health.status_code == 200
    assert health.json() == {"status": "ok", "collection_name": "documents"}
    assert stats.status_code == 200
    assert index.status_code == 200
    assert search.status_code == 200
    assert chat.status_code == 200
    assert bootstrap.flags == [(False, False), (True, False), (True, False), (True, True)]
    assert orchestrator.calls == [
        ("stats", None),
        ("index", {"path": "/tmp/vault", "recursive": True, "extensions": [".md"]}),
        ("search", {"query": "rag", "top_k": 2}),
        ("chat", {"question": "Explain rag", "top_k": 2, "system_prompt": None}),
    ]


def test_api_maps_runtime_errors_to_http_status() -> None:
    class _ErrorOrchestrator:
        def search(self, query: str, *, top_k: int | None = None) -> SearchResult:
            del query, top_k
            raise RuntimeError("embedding runtime unavailable")

    class _ErrorBootstrap(_StubBootstrap):
        def __init__(self) -> None:
            self.config = self._Config()

        def build_orchestrator(self, *, require_embedding: bool = True, require_llm: bool = True) -> _ErrorOrchestrator:
            del require_embedding, require_llm
            return _ErrorOrchestrator()

    client = TestClient(create_app(bootstrap=_ErrorBootstrap()))

    response = client.get("/search", params={"query": "rag"})

    assert response.status_code == 503
    assert response.json()["detail"] == "embedding runtime unavailable"


def test_api_maps_runtime_configuration_errors_during_bootstrap_to_http_status() -> None:
    class _ErrorBootstrap(_StubBootstrap):
        def __init__(self) -> None:
            self.config = self._Config()

        def build_orchestrator(self, *, require_embedding: bool = True, require_llm: bool = True) -> _StubOrchestrator:
            del require_embedding, require_llm
            raise RuntimeConfigurationError(
                "Embedding model is not available at /models/missing.gguf. "
                "Set --embedding-model-path or LOCAL_RAG_ASSISTANT_EMBEDDING_MODEL_PATH to a readable local model file."
            )

    client = TestClient(create_app(bootstrap=_ErrorBootstrap()))

    response = client.get("/search", params={"query": "rag"})

    assert response.status_code == 503
    assert "--embedding-model-path" in response.json()["detail"]


def test_api_v1_ingest_returns_400_for_unsupported_file_type() -> None:
    class _ErrorService(_StubIngestionService):
        def ingest_paths(self, paths: list[str]) -> list[IngestionDocumentRecord]:
            del paths
            raise UnsupportedFileTypeError(
                "Unsupported file type for ingestion: .png. Supported types: .txt, .md, .pdf."
            )

    bootstrap = _StubBootstrap(_StubOrchestrator())
    bootstrap._service = _ErrorService(bootstrap._registry)
    client = TestClient(create_app(bootstrap=bootstrap))

    response = client.post("/api/v1/ingest", json={"paths": ["/tmp/image.png"]})

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported file type for ingestion: .png. Supported types: .txt, .md, .pdf."


def test_api_v1_ingestion_and_document_routes_use_registry_and_service() -> None:
    orchestrator = _StubOrchestrator()
    bootstrap = _StubBootstrap(orchestrator)
    client = TestClient(create_app(bootstrap=bootstrap))

    ingest = client.post("/api/v1/ingest", json={"paths": ["/tmp/new-a.md", "/tmp/library"]})
    documents = client.get("/api/v1/documents")
    document = client.get("/api/v1/documents/doc-1")
    reindex = client.post("/api/v1/documents/doc-2/reindex")
    delete = client.delete("/api/v1/documents/doc-1")

    assert ingest.status_code == 200
    assert ingest.json()["documents"][0]["document_id"] == "ingested-1"
    assert documents.status_code == 200
    assert len(documents.json()["documents"]) == 4
    assert document.status_code == 200
    assert document.json()["document_id"] == "doc-1"
    assert reindex.status_code == 200
    assert reindex.json()["document_id"] == "doc-2"
    assert reindex.json()["status"] == "indexed"
    assert delete.status_code == 200
    assert delete.json() == {"document_id": "doc-1", "deleted": True}
    assert bootstrap._service.calls == [
        ("ingest_paths", ["/tmp/new-a.md", "/tmp/library"]),
        ("ingest_path", "/tmp/beta.md"),
    ]
    assert bootstrap._vectorstore.deleted_document_ids == ["doc-1"]


def test_api_v1_document_routes_return_404_for_unknown_document() -> None:
    bootstrap = _StubBootstrap(_StubOrchestrator())
    client = TestClient(create_app(bootstrap=bootstrap))

    get_response = client.get("/api/v1/documents/missing")
    delete_response = client.delete("/api/v1/documents/missing")
    reindex_response = client.post("/api/v1/documents/missing/reindex")

    assert get_response.status_code == 404
    assert get_response.json()["detail"] == "Document missing not found"
    assert delete_response.status_code == 404
    assert delete_response.json()["detail"] == "Document missing not found"
    assert reindex_response.status_code == 404
    assert reindex_response.json()["detail"] == "Document missing not found"
    assert bootstrap._vectorstore.deleted_document_ids == []


def test_api_v1_ingestion_job_routes_start_track_and_stream_events() -> None:
    bootstrap = _StubBootstrap(_StubOrchestrator())
    client = TestClient(create_app(bootstrap=bootstrap))

    create = client.post("/api/v1/ingestion-jobs", json={"paths": ["/tmp/new-a.md", "/tmp/library"]})

    assert create.status_code == 202
    body = create.json()
    assert body["status"] in {"running", "completed"}
    job_id = body["job_id"]

    state = None
    for _ in range(20):
        state = client.get(f"/api/v1/ingestion-jobs/{job_id}")
        assert state.status_code == 200
        if state.json()["status"] == "completed":
            break
        time.sleep(0.01)

    assert state is not None
    assert state.json()["status"] == "completed"
    assert state.json()["emitted_events"] == 8
    assert len(state.json()["results"]) == 2
    assert state.json()["snapshot"]["event"] == "completed"

    with client.stream("GET", f"/api/v1/ingestion-jobs/{job_id}/events") as response:
        payload = response.read().decode()

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "retry: 1000" in payload
    assert f'"job_id": "{job_id}"' in payload
    assert "event: started" in payload
    assert "event: discovered" in payload
    assert "event: processing" in payload
    assert "event: indexed" in payload
    assert "event: completed" in payload

    with client.stream("GET", f"/api/v1/ingestion-jobs/{job_id}/events", params={"last_event_id": 7}) as response:
        resumed_payload = response.read().decode()

    assert "event: completed" in resumed_payload
    assert "event: started" not in resumed_payload


def test_api_v1_ingestion_job_events_resume_from_last_event_id_header() -> None:
    bootstrap = _StubBootstrap(_StubOrchestrator())
    client = TestClient(create_app(bootstrap=bootstrap))

    create = client.post("/api/v1/ingestion-jobs", json={"paths": ["/tmp/new-a.md"]})
    job_id = create.json()["job_id"]

    for _ in range(20):
        state = client.get(f"/api/v1/ingestion-jobs/{job_id}")
        if state.json()["status"] == "completed":
            break
        time.sleep(0.01)

    with client.stream(
        "GET",
        f"/api/v1/ingestion-jobs/{job_id}/events",
        headers={"Last-Event-ID": "3"},
    ) as response:
        payload = response.read().decode()

    assert response.status_code == 200
    assert "event: started" not in payload
    assert "event: indexed" in payload
    assert "event: completed" in payload


def test_api_v1_ingestion_job_routes_return_404_for_unknown_job() -> None:
    bootstrap = _StubBootstrap(_StubOrchestrator())
    client = TestClient(create_app(bootstrap=bootstrap))

    state = client.get("/api/v1/ingestion-jobs/missing")
    with client.stream("GET", "/api/v1/ingestion-jobs/missing/events") as stream:
        events_status = stream.status_code
        events_body = stream.read().decode()

    assert state.status_code == 404
    assert state.json()["detail"] == "Ingestion job missing not found"
    assert events_status == 404
    assert "Ingestion job missing not found" in events_body


def test_ui_routes_render_runtime_data() -> None:
    bootstrap = _StubBootstrap(_StubOrchestrator())
    client = TestClient(create_app(bootstrap=bootstrap))

    overview = client.get("/overview")
    ingest = client.get("/ingest")
    documents = client.get("/documents")
    search = client.get("/search", headers={"accept": "text/html"})
    chat = client.get("/chat")

    assert overview.status_code == 200
    assert "Document Operations" in overview.text
    assert "Searchable documents" in overview.text
    assert ">3<" in overview.text
    assert "doc-1" not in overview.text
    assert "indexed" in overview.text

    assert ingest.status_code == 200
    assert "Prepare document imports" in ingest.text
    assert "/api/v1/ingest" in ingest.text
    assert "/api/v1/ingestion-jobs" in ingest.text
    assert str(bootstrap.config.ingestion_registry_path) in ingest.text
    assert "Ingest paths" in ingest.text
    assert "Live progress is streamed from the in-memory ingestion job store" in ingest.text
    assert "data-ingest-form" in ingest.text

    assert documents.status_code == 200
    assert "Tracked sources" in documents.text
    assert "/tmp/alpha.md" in documents.text
    assert "/tmp/beta.md" in documents.text
    assert "failed" in documents.text
    assert "/documents/doc-1" in documents.text

    assert search.status_code == 200
    assert "Search indexed content" in search.text
    assert "Run a query to inspect retrieval results." in search.text

    assert chat.status_code == 200
    assert "Ask against the local index" in chat.text
    assert "Run a prompt to inspect a grounded answer." in chat.text


def test_ui_ingest_submission_posts_paths_and_redirects_with_feedback() -> None:
    bootstrap = _StubBootstrap(_StubOrchestrator())
    client = TestClient(create_app(bootstrap=bootstrap))

    response = client.post(
        "/ingest",
        data={"paths": "/tmp/new-a.md\n\n/tmp/library\n"},
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert "Processed 2 path(s): 2 indexed, 0 failed." in response.text
    assert "ingested-1" in response.text
    assert bootstrap._service.calls == [("ingest_paths", ["/tmp/new-a.md", "/tmp/library"])]


def test_ui_ingest_submission_renders_unsupported_file_type_error() -> None:
    class _ErrorService(_StubIngestionService):
        def ingest_paths(self, paths: list[str]) -> list[IngestionDocumentRecord]:
            self.calls.append(("ingest_paths", paths))
            raise UnsupportedFileTypeError(
                "Unsupported file type for ingestion: .png. Supported types: .txt, .md, .pdf."
            )

    bootstrap = _StubBootstrap(_StubOrchestrator())
    bootstrap._service = _ErrorService(bootstrap._registry)
    client = TestClient(create_app(bootstrap=bootstrap))

    response = client.post("/ingest", data={"paths": "/tmp/image.png"})

    assert response.status_code == 400
    assert "Unsupported file type for ingestion: .png. Supported types: .txt, .md, .pdf." in response.text


def test_ui_ingest_submission_requires_at_least_one_path() -> None:
    bootstrap = _StubBootstrap(_StubOrchestrator())
    client = TestClient(create_app(bootstrap=bootstrap))

    response = client.post("/ingest", data={"paths": "  \n  "})

    assert response.status_code == 400
    assert "Enter at least one absolute or resolvable path." in response.text
    assert bootstrap._service.calls == []


def test_ui_document_detail_renders_record_and_not_found_cleanly() -> None:
    bootstrap = _StubBootstrap(_StubOrchestrator())
    client = TestClient(create_app(bootstrap=bootstrap))

    detail = client.get("/documents/doc-2")
    missing = client.get("/documents/missing")

    assert detail.status_code == 200
    assert "Document detail" in detail.text
    assert "doc-2" in detail.text
    assert "parse failed" in detail.text
    assert "/tmp/beta.md" in detail.text

    assert missing.status_code == 404
    assert "Document unavailable" in missing.text
    assert "missing" in missing.text


def test_ui_document_actions_wire_reindex_and_delete() -> None:
    bootstrap = _StubBootstrap(_StubOrchestrator())
    client = TestClient(create_app(bootstrap=bootstrap))

    reindex = client.post("/documents/doc-2/reindex", follow_redirects=True)
    delete = client.post("/documents/doc-1/delete", follow_redirects=True)

    assert reindex.status_code == 200
    assert "Document reindexed." in reindex.text
    assert "doc-2" in reindex.text
    assert bootstrap._service.calls == [("ingest_path", "/tmp/beta.md")]

    assert delete.status_code == 200
    assert "Deleted Document." in delete.text
    assert "doc-1" not in delete.text
    assert bootstrap._vectorstore.deleted_document_ids == ["doc-1"]


def test_ui_document_actions_return_not_found_for_missing_record() -> None:
    bootstrap = _StubBootstrap(_StubOrchestrator())
    client = TestClient(create_app(bootstrap=bootstrap))

    reindex = client.post("/documents/missing/reindex")
    delete = client.post("/documents/missing/delete")

    assert reindex.status_code == 404
    assert "Document unavailable" in reindex.text
    assert delete.status_code == 404
    assert "Document unavailable" in delete.text
    assert bootstrap._service.calls == []
    assert bootstrap._vectorstore.deleted_document_ids == []


def test_ui_search_page_renders_results_via_existing_runtime() -> None:
    orchestrator = _StubOrchestrator()
    bootstrap = _StubBootstrap(orchestrator)
    client = TestClient(create_app(bootstrap=bootstrap))

    response = client.get("/search", params={"query": "rag"}, headers={"accept": "text/html"})

    assert response.status_code == 200
    assert "1 result(s) for" in response.text
    assert "alpha" in response.text
    assert "notes/alpha.md" in response.text
    assert bootstrap.flags == [(True, False)]
    assert orchestrator.calls == [("search", {"query": "rag", "top_k": None})]


def test_ui_chat_page_renders_answer_and_sources_via_existing_runtime() -> None:
    orchestrator = _StubOrchestrator()
    bootstrap = _StubBootstrap(orchestrator)
    client = TestClient(create_app(bootstrap=bootstrap))

    response = client.get("/chat", params={"query": "Summarize rag"})

    assert response.status_code == 200
    assert "grounded" in response.text
    assert "Alpha note" in response.text
    assert "notes/alpha.md" in response.text
    assert bootstrap.flags == [(True, True)]
    assert orchestrator.calls == [("chat", {"question": "Summarize rag", "top_k": None, "system_prompt": None})]


def test_ui_search_and_chat_pages_render_runtime_errors_cleanly() -> None:
    class _ErrorOrchestrator:
        def search(self, query: str, *, top_k: int | None = None) -> SearchResult:
            del query, top_k
            raise RuntimeError("embedding runtime unavailable")

        def chat(self, question: str, *, top_k: int | None = None, system_prompt: str | None = None) -> ChatResponse:
            del question, top_k, system_prompt
            raise RuntimeError("llm runtime unavailable")

    class _ErrorBootstrap(_StubBootstrap):
        def __init__(self) -> None:
            self.config = self._Config()
            self.flags: list[tuple[bool, bool]] = []

        def build_orchestrator(self, *, require_embedding: bool = True, require_llm: bool = True) -> _ErrorOrchestrator:
            self.flags.append((require_embedding, require_llm))
            return _ErrorOrchestrator()

    client = TestClient(create_app(bootstrap=_ErrorBootstrap()))

    search = client.get("/search", params={"query": "rag"}, headers={"accept": "text/html"})
    chat = client.get("/chat", params={"query": "Explain rag"})

    assert search.status_code == 503
    assert "embedding runtime unavailable" in search.text
    assert chat.status_code == 503
    assert "llm runtime unavailable" in chat.text


def test_ui_static_stylesheet_is_served() -> None:
    bootstrap = _StubBootstrap(_StubOrchestrator())
    client = TestClient(create_app(bootstrap=bootstrap))

    response = client.get("/static/styles.css")

    assert response.status_code == 200
    assert "text/css" in response.headers["content-type"]
    assert ".shell" in response.text

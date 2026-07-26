from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from click.testing import CliRunner

from local_rag_assistant.cli.commands import cli, main
from local_rag_assistant.cli.tui import render_diagnostics, render_search_result
from local_rag_assistant.config import Config, RuntimeConfigurationError
from local_rag_assistant.ingestion.models import IngestionDocumentRecord
from local_rag_assistant.models import ChatResponse, ChunkResult, IndexResponse, IndexStats, SearchResult


class _StubOrchestrator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

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
                    text="retrieved context",
                    document_id="doc-1",
                    chunk_index=0,
                    score=0.9,
                    metadata={"source": "notes/rag.md"},
                )
            ],
        )

    def chat(self, question: str, *, top_k: int | None = None, system_prompt: str | None = None) -> ChatResponse:
        self.calls.append(("chat", {"question": question, "top_k": top_k, "system_prompt": system_prompt}))
        return ChatResponse(answer="grounded answer", query=question)

    def stats(self) -> IndexStats:
        self.calls.append(("stats", None))
        return IndexStats(documents=2, chunks=5)


class _StubBootstrap:
    def __init__(self, orchestrator: _StubOrchestrator) -> None:
        self.orchestrator = orchestrator
        self.flags: list[tuple[bool, bool]] = []
        self.config = Config(
            chroma_db_path=Path("/tmp/chroma"),
            ingestion_registry_path=Path("/tmp/registry.sqlite3"),
            embedding_model_path=Path("/tmp/embed.gguf"),
            llm_model_path=Path("/tmp/llm.gguf"),
        )

    def build_orchestrator(self, *, require_embedding: bool = True, require_llm: bool = True) -> _StubOrchestrator:
        self.flags.append((require_embedding, require_llm))
        return self.orchestrator

    def registry(self) -> object:
        class _StubRegistry:
            @staticmethod
            def list() -> list[IngestionDocumentRecord]:
                return [
                    IngestionDocumentRecord(
                        document_id="doc-1",
                        source_path="/vault/notes/rag.md",
                        file_type="markdown",
                        title="RAG Notes",
                        content_sha256="abc",
                        metadata={"topic": "rag"},
                        status="indexed",
                        created_at=datetime(2026, 7, 16, tzinfo=timezone.utc),
                        updated_at=datetime(2026, 7, 17, tzinfo=timezone.utc),
                    )
                ]

            @staticmethod
            def get(document_id: str) -> IngestionDocumentRecord | None:
                if document_id != "doc-1":
                    return None
                return _StubRegistry.list()[0]

        return _StubRegistry()

    def vectorstore(self) -> object:
        class _StubVectorStore:
            @staticmethod
            def get_document_chunks(document_id: str):
                assert document_id == "doc-1"
                return []

        return _StubVectorStore()


def test_cli_search_command_uses_bootstrap(monkeypatch) -> None:
    orchestrator = _StubOrchestrator()
    bootstrap = _StubBootstrap(orchestrator)
    runner = CliRunner()

    monkeypatch.setattr("local_rag_assistant.cli.commands.build_bootstrap", lambda config: bootstrap)

    result = runner.invoke(cli, ["search", "what is rag", "--top-k", "2"])

    assert result.exit_code == 0
    assert '"query": "what is rag"' in result.output
    assert bootstrap.flags == [(True, False)]
    assert orchestrator.calls == [("search", {"query": "what is rag", "top_k": 2})]


def test_cli_search_reports_runtime_configuration_error_from_bootstrap(monkeypatch) -> None:
    runner = CliRunner()

    class _ErrorBootstrap(_StubBootstrap):
        def __init__(self) -> None:
            super().__init__(_StubOrchestrator())

        def build_orchestrator(self, *, require_embedding: bool = True, require_llm: bool = True) -> _StubOrchestrator:
            del require_embedding, require_llm
            raise RuntimeConfigurationError(
                "Embedding model is not available at /models/missing.gguf. "
                "Set --embedding-model-path or LOCAL_RAG_ASSISTANT_EMBEDDING_MODEL_PATH to a readable local model file."
            )

    monkeypatch.setattr("local_rag_assistant.cli.commands.build_bootstrap", lambda config: _ErrorBootstrap())

    result = runner.invoke(cli, ["search", "what is rag"])

    assert result.exit_code == 1
    assert "--embedding-model-path" in result.output


def test_cli_tui_status_command_renders_runtime_summary(monkeypatch) -> None:
    orchestrator = _StubOrchestrator()
    bootstrap = _StubBootstrap(orchestrator)
    runner = CliRunner()

    monkeypatch.setattr("local_rag_assistant.cli.commands.build_bootstrap", lambda config: bootstrap)

    result = runner.invoke(cli, ["tui", "status"])

    assert result.exit_code == 0
    assert "Runtime Status" in result.output
    assert "Documents in index: 2" in result.output
    assert bootstrap.flags == [(False, False)]


def test_cli_tui_inspect_command_renders_document_summary(monkeypatch) -> None:
    orchestrator = _StubOrchestrator()
    bootstrap = _StubBootstrap(orchestrator)
    runner = CliRunner()

    monkeypatch.setattr("local_rag_assistant.cli.commands.build_bootstrap", lambda config: bootstrap)

    result = runner.invoke(cli, ["tui", "inspect", "doc-1"])

    assert result.exit_code == 0
    assert "Document doc-1" in result.output
    assert "Source: /vault/notes/rag.md" in result.output
    assert "Chunk count: 0" in result.output


def test_cli_serve_command_starts_uvicorn(monkeypatch) -> None:
    calls: list[object] = []
    runner = CliRunner()

    class _StubUvicorn:
        @staticmethod
        def run(app, *, host: str, port: int, reload: bool) -> None:
            calls.append((app, host, port, reload))

    monkeypatch.setattr("local_rag_assistant.cli.commands.create_app", lambda config: "app")
    monkeypatch.setitem(__import__("sys").modules, "uvicorn", _StubUvicorn)

    result = runner.invoke(cli, ["serve", "--host", "0.0.0.0", "--port", "9000", "--reload"])

    assert result.exit_code == 0
    assert calls == [("app", "0.0.0.0", 9000, True)]


def test_main_returns_click_exit_code(monkeypatch) -> None:
    monkeypatch.setattr("local_rag_assistant.cli.commands.cli.main", lambda **_: (_ for _ in ()).throw(SystemExit(3)))

    assert main([]) == 3


def test_render_search_result_formats_ranked_matches() -> None:
    result = SearchResult(
        query="rag",
        total_results=1,
        results=[
            ChunkResult(
                text="retrieval augmented generation uses context to answer questions",
                document_id="doc-1",
                chunk_index=2,
                score=0.92,
                metadata={"source": "notes/rag.md", "title": "RAG"},
            )
        ],
    )

    rendered = render_search_result(result)

    assert "1. [0.92] notes/rag.md#chunk-2" in rendered
    assert "title: RAG" in rendered


def test_render_diagnostics_reports_missing_paths(tmp_path: Path) -> None:
    config = Config(
        chroma_db_path=tmp_path / "chroma",
        ingestion_registry_path=tmp_path / "registry.sqlite3",
        embedding_model_path=tmp_path / "missing-embed.gguf",
        llm_model_path=tmp_path / "missing-llm.gguf",
    )
    config.chroma_db_path.mkdir(parents=True)
    stats = IndexStats(documents=1, chunks=3, collection_name="documents")
    documents = [
        IngestionDocumentRecord(
            document_id="doc-1",
            source_path="/vault/notes/rag.md",
            file_type="markdown",
            title="RAG Notes",
            content_sha256="abc",
            metadata={},
            status="indexed",
            created_at=datetime(2026, 7, 16, tzinfo=timezone.utc),
            updated_at=datetime(2026, 7, 17, tzinfo=timezone.utc),
        )
    ]

    rendered = render_diagnostics(config, stats, documents)

    assert f"Chroma DB: {config.chroma_db_path} (ok)" in rendered
    assert f"Embedding model: {config.embedding_model_path} (missing)" in rendered

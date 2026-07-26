from __future__ import annotations

from pathlib import Path

from local_rag_assistant.config import Config
from local_rag_assistant.models import ChunkResult, IndexStats
from local_rag_assistant.orchestrator import Orchestrator


class _StubVectorStore:
    def __init__(self, results: list[ChunkResult], stats: IndexStats) -> None:
        self.results = results
        self._stats = stats
        self.search_calls: list[tuple[list[float], int]] = []

    def search(self, query_embedding: list[float], *, top_k: int = 5, filter_metadata: dict[str, object] | None = None) -> list[ChunkResult]:
        del filter_metadata
        self.search_calls.append((query_embedding, top_k))
        return self.results[:top_k]

    def get_stats(self) -> IndexStats:
        return self._stats


class _StubIndexer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool, list[str] | None]] = []

    def index(self, path: str, *, recursive: bool = True, extensions: list[str] | None = None) -> object:
        self.calls.append((path, recursive, extensions))
        return {"status": "ok"}


def _config(tmp_path: Path) -> Config:
    return Config(
        chroma_db_path=tmp_path / "chroma",
        embedding_model_path=tmp_path / "embed.gguf",
        llm_model_path=tmp_path / "llm.gguf",
        default_top_k=4,
    )


def test_orchestrator_search_uses_query_embeddings(tmp_path: Path, mock_embedding: object, mock_llm: object) -> None:
    results = [
        ChunkResult(
            text="RAG is retrieval-augmented generation.",
            document_id="doc-1",
            chunk_index=0,
            score=0.9,
            metadata={"source": "notes/rag.md", "title": "RAG"},
        )
    ]
    vectorstore = _StubVectorStore(results, IndexStats(documents=1, chunks=1))
    orchestrator = Orchestrator(_config(tmp_path), vectorstore, mock_embedding, mock_llm)

    response = orchestrator.search("What is RAG?", top_k=1)

    assert response.total_results == 1
    assert response.results[0].metadata["source"] == "notes/rag.md"
    assert vectorstore.search_calls == [([13.0, 13.0, 13.0, 13.0], 1)]


def test_orchestrator_chat_returns_answer_and_sources(tmp_path: Path, mock_embedding: object, mock_llm: object) -> None:
    results = [
        ChunkResult(
            text="RAG combines retrieval and generation to answer with context.",
            document_id="doc-1",
            chunk_index=2,
            score=0.88,
            metadata={"source": "notes/rag.md", "title": "RAG", "path": "notes/rag.md"},
        )
    ]
    vectorstore = _StubVectorStore(results, IndexStats(documents=1, chunks=3))
    orchestrator = Orchestrator(_config(tmp_path), vectorstore, mock_embedding, mock_llm)

    response = orchestrator.chat("Explain RAG", top_k=1)

    assert response.answer == "Mock grounded answer"
    assert response.sources[0].source == "notes/rag.md"
    assert response.sources[0].chunk_index == 2
    assert response.sources[0].excerpt == "RAG combines retrieval and generation to answer with context."


def test_orchestrator_chat_short_circuits_when_index_is_empty(tmp_path: Path, mock_embedding: object, mock_llm: object) -> None:
    vectorstore = _StubVectorStore([], IndexStats(documents=0, chunks=0))
    orchestrator = Orchestrator(_config(tmp_path), vectorstore, mock_embedding, mock_llm)

    response = orchestrator.chat("Explain RAG")

    assert response.answer == "No documents have been indexed yet. Please run indexing first."
    assert response.sources == []
    assert vectorstore.search_calls == []


def test_orchestrator_index_delegates_to_indexer(tmp_path: Path, mock_embedding: object, mock_llm: object) -> None:
    vectorstore = _StubVectorStore([], IndexStats(documents=0, chunks=0))
    indexer = _StubIndexer()
    orchestrator = Orchestrator(_config(tmp_path), vectorstore, mock_embedding, mock_llm, indexer=indexer)

    response = orchestrator.index("/tmp/docs", recursive=False, extensions=[".md"])

    assert response == {"status": "ok"}
    assert indexer.calls == [("/tmp/docs", False, [".md"])]

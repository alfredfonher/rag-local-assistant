from __future__ import annotations

from pathlib import Path

from local_rag_assistant.bootstrap import build_bootstrap, build_orchestrator
from local_rag_assistant.config import Config
from local_rag_assistant.orchestrator import Orchestrator


class _StubVectorStore:
    def __init__(self, persist_dir: Path, collection_name: str = "documents") -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name


def _config(tmp_path: Path) -> Config:
    embedding = tmp_path / "embed.gguf"
    llm = tmp_path / "llm.gguf"
    embedding.write_text("embed", encoding="utf-8")
    llm.write_text("llm", encoding="utf-8")
    return Config(
        chroma_db_path=tmp_path / "chroma",
        chroma_collection_name="test-documents",
        embedding_model_path=embedding,
        llm_model_path=llm,
    )


def test_bootstrap_builds_lazy_orchestrator(tmp_path: Path) -> None:
    calls: list[tuple[str, Path]] = []

    def embedding_factory(model_path: Path) -> object:
        calls.append(("embedding", model_path))
        return object()

    def llm_factory(model_path: Path) -> object:
        calls.append(("llm", model_path))
        return object()

    bootstrap = build_bootstrap(
        _config(tmp_path),
        vectorstore_factory=_StubVectorStore,
        embedding_factory=embedding_factory,
        llm_factory=llm_factory,
    )

    orchestrator = bootstrap.build_orchestrator(require_embedding=True, require_llm=False)

    assert isinstance(orchestrator, Orchestrator)
    assert calls == [("embedding", tmp_path / "embed.gguf")]


def test_build_orchestrator_wires_all_runtime_dependencies(tmp_path: Path) -> None:
    orchestrator = build_orchestrator(
        _config(tmp_path),
        require_embedding=False,
        require_llm=False,
    )

    assert isinstance(orchestrator, Orchestrator)


def test_bootstrap_registry_initializes_schema(tmp_path: Path) -> None:
    bootstrap = build_bootstrap(
        _config(tmp_path),
        vectorstore_factory=_StubVectorStore,
        embedding_factory=lambda _: object(),
        llm_factory=lambda _: object(),
    )

    registry = bootstrap.registry()

    assert registry.database_path.is_file()

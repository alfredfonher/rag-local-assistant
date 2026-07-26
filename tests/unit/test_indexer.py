from __future__ import annotations

from pathlib import Path

from local_rag_assistant.config import Config
from local_rag_assistant.indexer import Indexer
from local_rag_assistant.models import Document
from local_rag_assistant.utils.change_tracker import ChangeTracker
from local_rag_assistant.utils.text_splitter import RecursiveCharacterTextSplitter


class _RecordingVectorStore:
    def __init__(self) -> None:
        self.deleted_document_ids: list[str] = []
        self.upserts: list[tuple[list[object], list[list[float]]]] = []

    def upsert(self, chunks: list[object], embeddings: list[list[float]]) -> None:
        self.upserts.append((chunks, embeddings))

    def delete_by_document_id(self, document_id: str) -> None:
        self.deleted_document_ids.append(document_id)


class _RecordingEmbeddingModel:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int]] = []

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        self.calls.append((texts, batch_size))
        return [[float(index)] * 3 for index, _ in enumerate(texts, start=1)]


def test_indexer_indexes_new_documents_and_updates_tracker(temp_vault: Path, tmp_path: Path) -> None:
    state_file = tmp_path / "change.log"
    vectorstore = _RecordingVectorStore()
    embedding_model = _RecordingEmbeddingModel()
    indexer = Indexer(
        Config(chroma_db_path=tmp_path / "chroma", embedding_model_path=tmp_path / "embed.gguf", llm_model_path=tmp_path / "llm.gguf"),
        vectorstore,
        embedding_model,
        change_tracker=ChangeTracker(state_file),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=5),
    )

    report = indexer.index(str(temp_vault))
    state = ChangeTracker(state_file).get_state()

    assert report.status == "success"
    assert report.files_discovered == 5
    assert report.files_new == 5
    assert report.files_indexed == 5
    assert report.failed == 0
    assert report.chunks_created >= 5
    assert len(vectorstore.upserts) == 5
    assert len(state) == 5
    assert embedding_model.calls[0][1] == 32


def test_indexer_reindexes_modified_and_deleted_documents(temp_vault: Path, tmp_path: Path) -> None:
    state_file = tmp_path / "change.log"
    tracker = ChangeTracker(state_file)
    initial_files = sorted(temp_vault.rglob("*.md"))
    tracker.update_state(initial_files, hashes={str(path): tracker.calculate_hash(path) for path in initial_files})

    changed = temp_vault / "alpha.md"
    changed.write_text("# Alpha\n\nUpdated content for indexing.", encoding="utf-8")
    deleted = temp_vault / "gamma.md"
    deleted.unlink()

    vectorstore = _RecordingVectorStore()
    indexer = Indexer(
        Config(chroma_db_path=tmp_path / "chroma", embedding_model_path=tmp_path / "embed.gguf", llm_model_path=tmp_path / "llm.gguf"),
        vectorstore,
        _RecordingEmbeddingModel(),
        change_tracker=tracker,
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=5),
    )

    report = indexer.index(str(temp_vault))

    assert report.status == "success"
    assert report.files_new == 0
    assert report.files_modified == 1
    assert report.files_deleted == 1
    assert report.deleted_from_index == 1
    assert report.files_indexed == 1
    assert len(vectorstore.deleted_document_ids) == 2


def test_indexer_reports_document_loading_failures(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    broken = vault / "broken.md"
    broken.write_bytes(b"\xff\xfe\x00\x00")

    indexer = Indexer(
        Config(chroma_db_path=tmp_path / "chroma", embedding_model_path=tmp_path / "embed.gguf", llm_model_path=tmp_path / "llm.gguf"),
        _RecordingVectorStore(),
        _RecordingEmbeddingModel(),
        change_tracker=ChangeTracker(tmp_path / "change.log"),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=5),
    )

    report = indexer.index(str(vault))

    assert report.status == "failed"
    assert report.failed == 1
    assert report.files_indexed == 0
    assert len(report.errors) == 1


def test_indexer_can_index_a_preconstructed_document(tmp_path: Path) -> None:
    vectorstore = _RecordingVectorStore()
    embedding_model = _RecordingEmbeddingModel()
    indexer = Indexer(
        Config(chroma_db_path=tmp_path / "chroma", embedding_model_path=tmp_path / "embed.gguf", llm_model_path=tmp_path / "llm.gguf"),
        vectorstore,
        embedding_model,
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=5),
    )

    report = indexer.index_document(
        Document(
            id="doc-1",
            source="/tmp/doc-1.md",
            title="Doc 1",
            content="One chunk of text for indexing.",
            metadata={
                "path": "/tmp/doc-1.md",
                "file_type": "markdown",
                "modified_time": 123.0,
                "checksum": "hash-1",
            },
            hash="hash-1",
        )
    )

    assert report.status == "success"
    assert report.files_discovered == 1
    assert report.files_indexed == 1
    assert report.failed == 0
    assert vectorstore.deleted_document_ids == ["doc-1"]
    assert len(vectorstore.upserts) == 1
    assert embedding_model.calls[0][1] == 32


def test_default_change_tracker_state_is_scoped_to_chroma_path(temp_vault: Path, tmp_path: Path) -> None:
    first_db = tmp_path / "chroma-a"
    second_db = tmp_path / "chroma-b"

    first = Indexer(
        Config(chroma_db_path=first_db, embedding_model_path=tmp_path / "embed.gguf", llm_model_path=tmp_path / "llm.gguf"),
        _RecordingVectorStore(),
        _RecordingEmbeddingModel(),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=5),
    )
    second = Indexer(
        Config(chroma_db_path=second_db, embedding_model_path=tmp_path / "embed.gguf", llm_model_path=tmp_path / "llm.gguf"),
        _RecordingVectorStore(),
        _RecordingEmbeddingModel(),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=5),
    )

    first_report = first.index(str(temp_vault))
    second_report = second.index(str(temp_vault))

    assert first_report.files_indexed == 5
    assert second_report.files_indexed == 5
    assert (first_db / "change-tracker.json").is_file()
    assert (second_db / "change-tracker.json").is_file()

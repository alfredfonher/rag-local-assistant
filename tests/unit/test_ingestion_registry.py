from __future__ import annotations

from pathlib import Path

from local_rag_assistant.ingestion.models import ExtractionResult
from local_rag_assistant.ingestion.registry import IngestionRegistry


def _registry(tmp_path: Path) -> IngestionRegistry:
    registry = IngestionRegistry(tmp_path / "ingestion" / "registry.sqlite3")
    registry.init_schema()
    return registry


def _extraction_result(*, source_path: str, title: str = "Doc", content: str = "Hello") -> ExtractionResult:
    return ExtractionResult(
        source_path=source_path,
        file_type="markdown",
        title=title,
        content=content,
        metadata={"source": "vault"},
    )


def test_initialize_schema_creates_sqlite_database(tmp_path: Path) -> None:
    database_path = tmp_path / "ingestion" / "registry.sqlite3"

    registry = IngestionRegistry(database_path)
    registry.init_schema()

    assert database_path.is_file()


def test_upsert_document_inserts_and_gets_record(tmp_path: Path) -> None:
    registry = _registry(tmp_path)

    created = registry.upsert(
        "doc-1",
        _extraction_result(source_path="/vault/doc-1.md"),
        content_sha256="hash-1",
    )

    loaded = registry.get("doc-1")

    assert loaded == created
    assert loaded is not None
    assert loaded.document_id == "doc-1"
    assert loaded.status == "pending"
    assert loaded.metadata == {"source": "vault"}


def test_upsert_document_updates_existing_record_without_resetting_created_at(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    original = registry.upsert(
        "doc-1",
        _extraction_result(source_path="/vault/doc-1.md", title="Original"),
        content_sha256="hash-1",
    )

    updated = registry.upsert(
        "doc-1",
        _extraction_result(source_path="/vault/doc-1.md", title="Updated"),
        content_sha256="hash-2",
        status="indexed",
    )

    assert updated.title == "Updated"
    assert updated.content_sha256 == "hash-2"
    assert updated.status == "indexed"
    assert updated.created_at == original.created_at
    assert updated.updated_at >= original.updated_at


def test_list_documents_returns_all_records(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.upsert("doc-b", _extraction_result(source_path="/vault/doc-b.md"), content_sha256="hash-b")
    registry.upsert("doc-a", _extraction_result(source_path="/vault/doc-a.md"), content_sha256="hash-a")

    documents = registry.list()

    assert [document.document_id for document in documents] == ["doc-a", "doc-b"]


def test_delete_document_removes_existing_record(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.upsert("doc-1", _extraction_result(source_path="/vault/doc-1.md"), content_sha256="hash-1")

    deleted = registry.delete("doc-1")

    assert deleted is True
    assert registry.get("doc-1") is None


def test_update_document_status_persists_status_and_error(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.upsert("doc-1", _extraction_result(source_path="/vault/doc-1.md"), content_sha256="hash-1")

    updated = registry.update_status(
        "doc-1",
        status="failed",
        error_message="Extractor timeout",
    )

    assert updated is not None
    assert updated.status == "failed"
    assert updated.error_message == "Extractor timeout"


def test_update_status_can_clear_previous_error_message(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.upsert(
        "doc-1",
        _extraction_result(source_path="/vault/doc-1.md"),
        content_sha256="hash-1",
        status="failed",
        error_message="temporary issue",
    )

    updated = registry.update_status("doc-1", status="indexed", error_message="   ")

    assert updated is not None
    assert updated.status == "indexed"
    assert updated.error_message is None


def test_registry_returns_none_or_false_for_missing_document(tmp_path: Path) -> None:
    registry = _registry(tmp_path)

    assert registry.get("missing") is None
    assert registry.update_status("missing", status="failed", error_message="boom") is None
    assert registry.delete("missing") is False


def test_backward_compatible_registry_method_names_delegate_to_canonical_api(tmp_path: Path) -> None:
    registry = _registry(tmp_path)

    created = registry.upsert_document(
        "doc-legacy",
        _extraction_result(source_path="/vault/doc-legacy.md"),
        content_sha256="hash-legacy",
    )

    assert registry.get_document("doc-legacy") == created
    assert [record.document_id for record in registry.list_documents()] == ["doc-legacy"]
    assert registry.update_document_status("doc-legacy", status="indexed") is not None
    assert registry.delete_document("doc-legacy") is True

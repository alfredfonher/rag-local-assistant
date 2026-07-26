from __future__ import annotations

from pathlib import Path

from local_rag_assistant.document_loader import build_document_id
from local_rag_assistant.ingestion.extractors import EmptyExtractedContentError
from local_rag_assistant.ingestion.models import ExtractionResult
from local_rag_assistant.ingestion.progress import InMemoryIngestionProgressSink
from local_rag_assistant.ingestion.registry import IngestionRegistry
from local_rag_assistant.ingestion.service import IngestionService
from local_rag_assistant.models import Document, IndexResponse


class _StubIndexer:
    def __init__(self, reports: list[IndexResponse] | None = None, *, error: Exception | None = None) -> None:
        self._reports = reports or [IndexResponse(status="success", files_discovered=1, files_indexed=1, chunks_created=1)]
        self._error = error
        self.documents: list[Document] = []

    def index_document(self, document: Document) -> IndexResponse:
        self.documents.append(document)
        if self._error is not None:
            raise self._error
        return self._reports.pop(0)


def _registry(tmp_path: Path) -> IngestionRegistry:
    return IngestionRegistry(tmp_path / "ingestion" / "registry.sqlite3")


def test_ingest_path_indexes_extracted_document_and_marks_registry_indexed(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_text("# Guide\n\nBody", encoding="utf-8")
    indexer = _StubIndexer()
    service = IngestionService(_registry(tmp_path), indexer, extractor=_extractor_for({path: "Body"}))

    record = service.ingest_path(path)

    assert record.document_id == build_document_id(path)
    assert record.status == "indexed"
    assert record.error_message is None
    assert len(indexer.documents) == 1
    assert indexer.documents[0].id == build_document_id(path)
    assert indexer.documents[0].metadata["path"] == str(path.resolve())
    assert indexer.documents[0].metadata["file_type"] == "markdown"
    assert indexer.documents[0].metadata["checksum"] == record.content_sha256


def test_ingest_paths_records_index_failures_without_stopping_other_documents(tmp_path: Path) -> None:
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("# First", encoding="utf-8")
    second.write_text("# Second", encoding="utf-8")
    indexer = _StubIndexer(
        reports=[
            IndexResponse(status="success", files_discovered=1, files_indexed=1, chunks_created=1),
            IndexResponse(
                status="failed",
                files_discovered=1,
                files_indexed=0,
                chunks_created=0,
                failed=1,
                errors=["embedding backend unavailable"],
            ),
        ]
    )
    service = IngestionService(
        _registry(tmp_path),
        indexer,
        extractor=_extractor_for({first: "First body", second: "Second body"}),
    )

    records = service.ingest_paths([first, second])

    assert [record.status for record in records] == ["indexed", "failed"]
    assert records[1].error_message == "embedding backend unavailable"
    assert len(indexer.documents) == 2


def test_ingest_path_records_extraction_failure_for_supported_file_types(tmp_path: Path) -> None:
    path = tmp_path / "broken.md"
    path.write_text("# Broken", encoding="utf-8")
    service = IngestionService(_registry(tmp_path), _StubIndexer(), extractor=_raising_extractor(RuntimeError("parse failed")))

    record = service.ingest_path(path)

    assert record.document_id == build_document_id(path)
    assert record.status == "failed"
    assert record.error_message == "parse failed"
    assert record.file_type == "markdown"


def test_ingest_path_records_empty_extracted_content_failure_for_supported_file_types(tmp_path: Path) -> None:
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"%PDF")
    service = IngestionService(
        _registry(tmp_path),
        _StubIndexer(),
        extractor=_raising_extractor(
            EmptyExtractedContentError("Extracted pdf content is empty or unusable for ingestion: /tmp/empty.pdf")
        ),
    )

    record = service.ingest_path(path)

    assert record.document_id == build_document_id(path)
    assert record.status == "failed"
    assert record.file_type == "pdf"
    assert "empty or unusable" in record.error_message


def test_successful_reingestion_clears_previous_failure_state(tmp_path: Path) -> None:
    path = tmp_path / "retry.md"
    path.write_text("# Retry", encoding="utf-8")
    registry = _registry(tmp_path)
    failing_service = IngestionService(
        registry,
        _StubIndexer(
            reports=[
                IndexResponse(
                    status="failed",
                    files_discovered=1,
                    files_indexed=0,
                    chunks_created=0,
                    failed=1,
                    errors=["temporary index error"],
                )
            ]
        ),
        extractor=_extractor_for({path: "Retry body"}),
    )

    first = failing_service.ingest_path(path)

    succeeding_service = IngestionService(
        registry,
        _StubIndexer(),
        extractor=_extractor_for({path: "Retry body"}),
    )
    second = succeeding_service.ingest_path(path)

    assert first.status == "failed"
    assert first.error_message == "temporary index error"
    assert second.status == "indexed"
    assert second.error_message is None


def test_ingest_paths_recursively_processes_nested_markdown_files(tmp_path: Path) -> None:
    root = tmp_path / "library"
    nested = root / "guides" / "chapter"
    nested.mkdir(parents=True)
    document = nested / "intro.md"
    document.write_text("# Intro\n\nNested body", encoding="utf-8")
    indexer = _StubIndexer()
    service = IngestionService(_registry(tmp_path), indexer, extractor=_extractor_for({document: "Nested body"}))

    records = service.ingest_paths([root])

    assert [record.document_id for record in records] == [build_document_id(document)]
    assert [record.status for record in records] == ["indexed"]
    assert [indexed.source for indexed in indexer.documents] == [str(document.resolve())]


def test_ingest_paths_directory_supports_pdf_txt_and_markdown_files(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    markdown = root / "guide.md"
    text = root / "notes.txt"
    pdf = root / "manual.pdf"
    ignored = root / "image.png"
    markdown.write_text("# Guide", encoding="utf-8")
    text.write_text("Notes", encoding="utf-8")
    pdf.write_bytes(b"%PDF-1.4")
    ignored.write_bytes(b"png")
    indexer = _StubIndexer(
        reports=[
            IndexResponse(status="success", files_discovered=1, files_indexed=1, chunks_created=1),
            IndexResponse(status="success", files_discovered=1, files_indexed=1, chunks_created=1),
            IndexResponse(status="success", files_discovered=1, files_indexed=1, chunks_created=1),
        ]
    )
    service = IngestionService(
        _registry(tmp_path),
        indexer,
        extractor=_extractor_for({markdown: "Guide", text: "Notes", pdf: "Manual"}),
    )

    records = service.ingest_paths([root])

    assert [Path(record.source_path).name for record in records] == ["guide.md", "manual.pdf", "notes.txt"]
    assert [record.file_type for record in records] == ["markdown", "pdf", "text"]
    assert all(record.status == "indexed" for record in records)


def test_ingest_paths_accepts_mixed_file_and_directory_inputs(tmp_path: Path) -> None:
    direct = tmp_path / "direct.md"
    direct.write_text("# Direct", encoding="utf-8")
    folder = tmp_path / "folder"
    folder.mkdir()
    nested = folder / "nested.txt"
    nested.write_text("Nested", encoding="utf-8")
    indexer = _StubIndexer(
        reports=[
            IndexResponse(status="success", files_discovered=1, files_indexed=1, chunks_created=1),
            IndexResponse(status="success", files_discovered=1, files_indexed=1, chunks_created=1),
        ]
    )
    service = IngestionService(
        _registry(tmp_path),
        indexer,
        extractor=_extractor_for({direct: "Direct", nested: "Nested"}),
    )

    records = service.ingest_paths([direct, folder])

    assert [Path(record.source_path).name for record in records] == ["direct.md", "nested.txt"]
    assert [record.status for record in records] == ["indexed", "indexed"]


def test_ingest_paths_rejects_directories_without_supported_files(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "ignore.png").write_bytes(b"png")
    service = IngestionService(_registry(tmp_path), _StubIndexer(), extractor=_extractor_for({}))

    try:
        service.ingest_paths([empty])
    except ValueError as error:
        assert str(error) == (
            f"No supported files found for ingestion in directory: {empty.resolve()}. "
            "Supported types: .txt, .md, .pdf."
        )
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected ValueError for a directory without supported files")


def test_ingest_paths_emits_progress_events_for_discovery_and_completion(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    first = root / "first.md"
    second = root / "second.txt"
    first.write_text("# First", encoding="utf-8")
    second.write_text("Second", encoding="utf-8")
    progress_sink = InMemoryIngestionProgressSink()
    service = IngestionService(
        _registry(tmp_path),
        _StubIndexer(
            reports=[
                IndexResponse(status="success", files_discovered=1, files_indexed=1, chunks_created=1),
                IndexResponse(status="success", files_discovered=1, files_indexed=1, chunks_created=1),
            ]
        ),
        extractor=_extractor_for({first: "First body", second: "Second body"}),
        progress_sink=progress_sink,
    )

    records = service.ingest_paths([root])

    assert [record.status for record in records] == ["indexed", "indexed"]
    assert [event.event for event in progress_sink.events] == [
        "started",
        "discovered",
        "discovered",
        "processing",
        "indexed",
        "processing",
        "indexed",
        "completed",
    ]
    assert [event.current_path for event in progress_sink.events if event.event == "discovered"] == [
        str(first.resolve()),
        str(second.resolve()),
    ]
    assert progress_sink.events[0].total_files == 0
    assert progress_sink.events[2].total_files == 2
    assert progress_sink.events[4].processed_files == 1
    assert progress_sink.events[4].percent == 50.0
    assert progress_sink.events[-1].processed_files == 2
    assert progress_sink.events[-1].indexed_files == 2
    assert progress_sink.events[-1].failed_files == 0
    assert progress_sink.events[-1].percent == 100.0


def test_ingest_paths_emits_failed_progress_event_when_document_indexing_fails(tmp_path: Path) -> None:
    path = tmp_path / "broken.md"
    path.write_text("# Broken", encoding="utf-8")
    progress_sink = InMemoryIngestionProgressSink()
    service = IngestionService(
        _registry(tmp_path),
        _StubIndexer(
            reports=[
                IndexResponse(
                    status="failed",
                    files_discovered=1,
                    files_indexed=0,
                    chunks_created=0,
                    failed=1,
                    errors=["embedding backend unavailable"],
                )
            ]
        ),
        extractor=_extractor_for({path: "Broken body"}),
        progress_sink=progress_sink,
    )

    record = service.ingest_path(path)

    assert record.status == "failed"
    assert [event.event for event in progress_sink.events] == [
        "started",
        "discovered",
        "processing",
        "failed",
        "completed",
    ]
    failed_event = progress_sink.events[-2]
    assert failed_event.current_path == str(path.resolve())
    assert failed_event.error_message == "embedding backend unavailable"
    assert failed_event.processed_files == 1
    assert failed_event.failed_files == 1
    assert failed_event.percent == 100.0


def _extractor_for(contents: dict[Path, str]):
    def _extract(path: str | Path) -> ExtractionResult:
        resolved = Path(path).resolve()
        file_type = {
            ".md": "markdown",
            ".txt": "text",
            ".pdf": "pdf",
        }[resolved.suffix.lower()]
        return ExtractionResult(
            source_path=str(resolved),
            file_type=file_type,
            title=resolved.stem,
            content=contents[resolved],
            metadata={"origin": "test"},
        )

    return _extract


def _raising_extractor(error: Exception):
    def _extract(path: str | Path) -> ExtractionResult:
        raise error

    return _extract

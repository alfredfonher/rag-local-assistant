"""Application service for extraction-driven document ingestion."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from local_rag_assistant.document_loader import DocumentLoader, build_document_id
from local_rag_assistant.indexer import Indexer
from local_rag_assistant.ingestion.extractors import SUPPORTED_EXTENSIONS, extract_file
from local_rag_assistant.ingestion.models import ExtractionResult, IngestionDocumentRecord
from local_rag_assistant.ingestion.progress import IngestionProgressReporter, IngestionProgressSink
from local_rag_assistant.ingestion.registry import IngestionRegistry
from local_rag_assistant.models import Document, IndexResponse


class IngestionService:
    """Compose extraction, registry persistence, and indexing."""

    def __init__(
        self,
        registry: IngestionRegistry,
        indexer: Indexer,
        *,
        extractor: Callable[[str | Path], ExtractionResult] = extract_file,
        document_loader: DocumentLoader | None = None,
        progress_sink: IngestionProgressSink | None = None,
    ) -> None:
        self._registry = registry
        self._indexer = indexer
        self._extractor = extractor
        self._document_loader = document_loader or DocumentLoader()
        self._progress_sink = progress_sink
        self._registry.init_schema()

    def ingest_path(self, path: str | Path) -> IngestionDocumentRecord:
        return self.ingest_paths([path])[0]

    def ingest_paths(self, paths: Sequence[str | Path]) -> list[IngestionDocumentRecord]:
        reporter = IngestionProgressReporter(self._progress_sink)
        reporter.started()

        unique_paths: list[Path] = []
        seen_paths: set[Path] = set()
        for path in paths:
            try:
                expanded = self._expand_path(path)
            except Exception as exc:
                reporter.failed(path, error_message=str(exc), counts_as_processed=False)
                raise
            for expanded_path in expanded:
                if expanded_path in seen_paths:
                    continue
                seen_paths.add(expanded_path)
                unique_paths.append(expanded_path)
                reporter.discovered(expanded_path)

        records = [self._ingest_path(path, reporter=reporter) for path in unique_paths]
        reporter.completed()
        return records

    def _expand_path(self, path: str | Path) -> list[Path]:
        source_path = Path(path).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Document path does not exist: {source_path}")
        if source_path.is_dir():
            files = self._document_loader.discover_files(
                source_path,
                recursive=True,
                extensions=SUPPORTED_EXTENSIONS,
            )
            if files:
                return files
            supported = ", ".join(SUPPORTED_EXTENSIONS)
            raise ValueError(
                f"No supported files found for ingestion in directory: {source_path}. Supported types: {supported}."
            )
        return [source_path]

    def _ingest_path(
        self,
        path: str | Path,
        *,
        reporter: IngestionProgressReporter | None = None,
    ) -> IngestionDocumentRecord:
        source_path = Path(path).expanduser().resolve()
        document_id = build_document_id(source_path)
        if reporter is not None:
            reporter.processing(source_path, document_id=document_id)

        try:
            extraction = self._extractor(source_path)
        except Exception as exc:
            if reporter is not None:
                reporter.failed(source_path, document_id=document_id, error_message=str(exc))
            return self._record_extraction_failure(source_path, document_id, exc)

        content_sha256 = _sha256_text(extraction.content)
        self._registry.upsert(
            document_id,
            extraction,
            content_sha256=content_sha256,
            status="pending",
            error_message=None,
        )

        try:
            report = self._indexer.index_document(
                self._build_document(extraction, content_sha256=content_sha256)
            )
        except Exception as exc:
            if reporter is not None:
                reporter.failed(source_path, document_id=document_id, error_message=str(exc))
            return self._require_record(
                self._registry.update_status(document_id, status="failed", error_message=str(exc)),
                document_id,
            )

        if report.status == "success":
            if reporter is not None:
                reporter.indexed(source_path, document_id=document_id)
            return self._require_record(
                self._registry.update_status(document_id, status="indexed", error_message=None),
                document_id,
            )

        error_message = "; ".join(report.errors) or "Document indexing failed"
        if reporter is not None:
            reporter.failed(source_path, document_id=document_id, error_message=error_message)
        return self._require_record(
            self._registry.update_status(document_id, status="failed", error_message=error_message),
            document_id,
        )

    def _record_extraction_failure(
        self,
        source_path: Path,
        document_id: str,
        error: Exception,
    ) -> IngestionDocumentRecord:
        existing = self._registry.get(document_id)
        if existing is not None:
            return self._require_record(
                self._registry.update_status(document_id, status="failed", error_message=str(error)),
                document_id,
            )

        file_type = _file_type_from_path(source_path)
        if file_type is None:
            raise error

        placeholder = ExtractionResult(
            source_path=str(source_path),
            file_type=file_type,
            title=source_path.stem,
            content="extraction failed",
            metadata={"extension": source_path.suffix.lower()},
        )
        return self._registry.upsert(
            document_id,
            placeholder,
            content_sha256=_sha256_text(""),
            status="failed",
            error_message=str(error),
        )

    def _build_document(self, extraction: ExtractionResult, *, content_sha256: str) -> Document:
        source_path = Path(extraction.source_path).expanduser().resolve()
        modified_time = (
            source_path.stat().st_mtime
            if source_path.exists()
            else datetime.now(timezone.utc).timestamp()
        )
        metadata = {
            **extraction.metadata,
            "path": str(source_path),
            "file_type": extraction.file_type,
            "modified_time": modified_time,
            "checksum": content_sha256,
        }
        return Document(
            id=build_document_id(source_path),
            source=str(source_path),
            title=extraction.title,
            content=extraction.content,
            metadata=metadata,
            hash=content_sha256,
        )

    def _require_record(
        self,
        record: IngestionDocumentRecord | None,
        document_id: str,
    ) -> IngestionDocumentRecord:
        if record is None:  # pragma: no cover - defensive guard
            raise RuntimeError(f"Document {document_id} was not persisted.")
        return record


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_type_from_path(path: Path) -> str | None:
    mapping = {
        ".txt": "text",
        ".md": "markdown",
        ".pdf": "pdf",
    }
    return mapping.get(path.suffix.lower())


__all__ = ["IngestionService"]

"""SQLite-backed registry for ingestion document state."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from local_rag_assistant.ingestion.models import ExtractionResult, IngestionDocumentRecord, IngestionStatus


class IngestionRegistry:
    """Persist document ingestion state in a local SQLite database."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)

    def init_schema(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_documents (
                    document_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    content_sha256 TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_ingestion_documents_updated_at "
                "ON ingestion_documents(updated_at DESC, document_id ASC)"
            )

    def initialize_schema(self) -> None:
        self.init_schema()

    def upsert(
        self,
        document_id: str,
        extraction: ExtractionResult,
        *,
        content_sha256: str,
        status: IngestionStatus = "pending",
        error_message: str | None = None,
    ) -> IngestionDocumentRecord:
        timestamp = _utcnow()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ingestion_documents (
                    document_id,
                    source_path,
                    file_type,
                    title,
                    content_sha256,
                    metadata_json,
                    status,
                    error_message,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    source_path = excluded.source_path,
                    file_type = excluded.file_type,
                    title = excluded.title,
                    content_sha256 = excluded.content_sha256,
                    metadata_json = excluded.metadata_json,
                    status = excluded.status,
                    error_message = excluded.error_message,
                    updated_at = excluded.updated_at
                """,
                (
                    document_id,
                    extraction.source_path,
                    extraction.file_type,
                    extraction.title,
                    content_sha256,
                    json.dumps(extraction.metadata, sort_keys=True),
                    status,
                    error_message,
                    timestamp,
                    timestamp,
                ),
            )

        record = self.get_document(document_id)
        if record is None:  # pragma: no cover - defensive guard
            raise RuntimeError(f"Document {document_id} was not persisted.")
        return record

    def upsert_document(
        self,
        document_id: str,
        extraction: ExtractionResult,
        *,
        content_sha256: str,
        status: IngestionStatus = "pending",
        error_message: str | None = None,
    ) -> IngestionDocumentRecord:
        return self.upsert(
            document_id,
            extraction,
            content_sha256=content_sha256,
            status=status,
            error_message=error_message,
        )

    def get(self, document_id: str) -> IngestionDocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM ingestion_documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return self._row_to_record(row) if row is not None else None

    def get_document(self, document_id: str) -> IngestionDocumentRecord | None:
        return self.get(document_id)

    def list(self) -> list[IngestionDocumentRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM ingestion_documents ORDER BY updated_at DESC, document_id ASC"
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def list_documents(self) -> list[IngestionDocumentRecord]:
        return self.list()

    def delete(self, document_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM ingestion_documents WHERE document_id = ?",
                (document_id,),
            )
        return cursor.rowcount > 0

    def delete_document(self, document_id: str) -> bool:
        return self.delete(document_id)

    def update_status(
        self,
        document_id: str,
        *,
        status: IngestionStatus,
        error_message: str | None = None,
    ) -> IngestionDocumentRecord | None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE ingestion_documents
                SET status = ?, error_message = ?, updated_at = ?
                WHERE document_id = ?
                """,
                (status, error_message, _utcnow(), document_id),
            )
            if cursor.rowcount == 0:
                return None
            row = connection.execute(
                "SELECT * FROM ingestion_documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return self._row_to_record(row)

    def update_document_status(
        self,
        document_id: str,
        *,
        status: IngestionStatus,
        error_message: str | None = None,
    ) -> IngestionDocumentRecord | None:
        return self.update_status(document_id, status=status, error_message=error_message)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _row_to_record(self, row: sqlite3.Row) -> IngestionDocumentRecord:
        return IngestionDocumentRecord(
            document_id=row["document_id"],
            source_path=row["source_path"],
            file_type=row["file_type"],
            title=row["title"],
            content_sha256=row["content_sha256"],
            metadata=json.loads(row["metadata_json"]),
            status=row["status"],
            error_message=row["error_message"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["IngestionRegistry"]

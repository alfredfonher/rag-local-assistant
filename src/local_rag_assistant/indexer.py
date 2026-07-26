"""Incremental indexing pipeline wiring loaders, splitting, embeddings, and storage."""

from __future__ import annotations

import gc
from datetime import datetime, timezone
from pathlib import Path

from local_rag_assistant.config import Config
from local_rag_assistant.document_loader import DocumentLoader, build_document_id
from local_rag_assistant.models import Chunk, ChunkMetadata, Document, IndexResponse
from local_rag_assistant.utils.change_tracker import ChangeTracker
from local_rag_assistant.utils.text_splitter import get_text_splitter
from local_rag_assistant.vectorstore import VectorStore


class Indexer:
    """Coordinate incremental document indexing into the vector store."""

    def __init__(
        self,
        config: Config,
        vectorstore: VectorStore,
        embedding_model: object,
        *,
        document_loader: DocumentLoader | None = None,
        change_tracker: ChangeTracker | None = None,
        text_splitter: object | None = None,
    ) -> None:
        self._config = config
        self._vectorstore = vectorstore
        self._embedding_model = embedding_model
        self._document_loader = document_loader or DocumentLoader()
        self._change_tracker = change_tracker or ChangeTracker(config.chroma_db_path / "change-tracker.json")
        self._text_splitter = text_splitter or get_text_splitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    def index(
        self,
        path: str,
        *,
        recursive: bool = True,
        extensions: list[str] | None = None,
    ) -> IndexResponse:
        root = Path(path).expanduser().resolve()
        files = self._document_loader.discover_files(root, recursive=recursive, extensions=extensions)
        new_files, modified_files, deleted_files = self._change_tracker.detect_changes(files)
        modified_set = set(modified_files)

        report = IndexResponse(
            status="success",
            files_discovered=len(files),
            files_indexed=0,
            chunks_created=0,
            skipped=max(0, len(files) - len(new_files) - len(modified_files)),
            failed=0,
            files_new=len(new_files),
            files_modified=len(modified_files),
            files_deleted=len(deleted_files),
            deleted_from_index=0,
            errors=[],
        )

        if deleted_files:
            for deleted_path in deleted_files:
                self._vectorstore.delete_by_document_id(build_document_id(deleted_path))
            self._change_tracker.purge_state(deleted_files)
            report.deleted_from_index = len(deleted_files)

        successful_files: list[Path] = []
        successful_hashes: dict[str, str] = {}
        for file_path in [Path(item) for item in [*new_files, *modified_files]]:
            try:
                document = self._document_loader.load_document(root, file_path)
                if str(file_path) in modified_set:
                    self._vectorstore.delete_by_document_id(document.id)

                chunks = self._chunk_document(document)
                embeddings = self._embed_chunks(chunks)
                self._vectorstore.upsert(chunks, embeddings)

                successful_files.append(file_path)
                successful_hashes[str(file_path)] = document.hash
                report.files_indexed += 1
                report.chunks_created += len(chunks)
            except Exception as exc:
                report.failed += 1
                report.errors.append(f"{file_path}: {exc}")
            finally:
                gc.collect()

        if successful_files:
            self._change_tracker.update_state(successful_files, hashes=successful_hashes)

        if report.failed and (report.files_indexed or report.deleted_from_index):
            report.status = "partial_success"
        elif report.failed:
            report.status = "failed"

        return report

    def index_document(self, document: Document) -> IndexResponse:
        return self.index_documents([document])

    def index_documents(self, documents: list[Document]) -> IndexResponse:
        report = IndexResponse(
            status="success",
            files_discovered=len(documents),
            files_indexed=0,
            chunks_created=0,
            skipped=0,
            failed=0,
            files_new=len(documents),
            files_modified=0,
            files_deleted=0,
            deleted_from_index=0,
            errors=[],
        )

        for document in documents:
            try:
                self._vectorstore.delete_by_document_id(document.id)
                chunks = self._chunk_document(document)
                embeddings = self._embed_chunks(chunks)
                self._vectorstore.upsert(chunks, embeddings)

                report.files_indexed += 1
                report.chunks_created += len(chunks)
            except Exception as exc:
                report.failed += 1
                report.errors.append(f"{document.source}: {exc}")
            finally:
                gc.collect()

        if report.failed and report.files_indexed:
            report.status = "partial_success"
        elif report.failed:
            report.status = "failed"

        return report

    def _chunk_document(self, document: Document) -> list[Chunk]:
        parts = [chunk.strip() for chunk in self._text_splitter.split_text(document.content) if chunk.strip()]
        if not parts:
            return []

        indexed_at = datetime.now(timezone.utc)
        total_chunks = len(parts)
        chunks: list[Chunk] = []
        for index, content in enumerate(parts):
            chunks.append(
                Chunk(
                    id=f"{document.id}:{index}",
                    document_id=document.id,
                    chunk_index=index,
                    content=content,
                    metadata=ChunkMetadata(
                        source=document.source,
                        title=document.title,
                        chunk_index=index,
                        total_chunks=total_chunks,
                        document_hash=document.hash,
                        indexed_at=indexed_at,
                        mtime=float(document.metadata["modified_time"]),
                        extra={
                            "path": document.metadata["path"],
                            "file_type": document.metadata["file_type"],
                            "checksum": document.metadata["checksum"],
                        },
                    ),
                )
            )
        return chunks

    def _embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        if not chunks:
            return []
        return self._embedding_model.encode(
            [chunk.content for chunk in chunks],
            batch_size=self._config.embedding_batch_size,
        )


__all__ = ["Indexer"]

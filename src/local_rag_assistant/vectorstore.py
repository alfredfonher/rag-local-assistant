"""Thin ChromaDB wrapper for chunk persistence and retrieval."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from local_rag_assistant.models import Chunk, ChunkMetadata, ChunkResult, IndexStats


class VectorStore:
    """Persist chunk embeddings in a local ChromaDB collection."""

    def __init__(self, persist_dir: str | Path, collection_name: str = "documents") -> None:
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.persist_dir.mkdir(parents=True, exist_ok=True)

    def init_collection(self) -> Any:
        client = self._create_client()
        return client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return

        collection = self.init_collection()
        collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            metadatas=[self._serialize_chunk_metadata(chunk) for chunk in chunks],
            embeddings=embeddings,
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self.upsert(chunks, embeddings)

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[ChunkResult]:
        collection = self.init_collection()
        response = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata,
        )

        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        results: list[ChunkResult] = []
        for text, metadata, distance in zip(documents, metadatas, distances, strict=False):
            results.append(
                ChunkResult(
                    text=text,
                    document_id=str(metadata.get("document_id", "")),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    score=max(0.0, min(1.0, 1.0 - float(distance))),
                    metadata=self._build_result_metadata(metadata),
                )
            )
        return results

    def query(
        self,
        embedding: list[float],
        n: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[ChunkResult]:
        return self.search(embedding, top_k=n, filter_metadata=filter_metadata)

    def delete_by_document_id(self, document_id: str) -> None:
        collection = self.init_collection()
        collection.delete(where={"document_id": document_id})

    def get_document_chunks(self, document_id: str) -> list[Chunk]:
        collection = self.init_collection()
        response = collection.get(where={"document_id": document_id}, include=["documents", "metadatas"])

        documents = response.get("documents", [])
        metadatas = response.get("metadatas", [])
        ids = response.get("ids", [])

        chunks: list[Chunk] = []
        for chunk_id, text, metadata in zip(ids, documents, metadatas, strict=False):
            chunk_metadata = ChunkMetadata.model_validate(
                {
                    "source": metadata.get("source", ""),
                    "title": metadata.get("title", ""),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "total_chunks": metadata.get("total_chunks"),
                    "document_hash": metadata.get("document_hash"),
                    "indexed_at": metadata.get("indexed_at"),
                    "mtime": metadata.get("mtime"),
                    "extra": self._deserialize_metadata(metadata),
                }
            )
            chunks.append(
                Chunk(
                    id=str(chunk_id),
                    document_id=document_id,
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    content=text,
                    metadata=chunk_metadata,
                )
            )

        return sorted(chunks, key=lambda chunk: chunk.chunk_index)

    def get_stats(self) -> IndexStats:
        collection = self.init_collection()
        chunk_count = collection.count()
        response = collection.get(include=["metadatas"])
        metadatas = response.get("metadatas", [])
        document_ids = {str(metadata.get("document_id", "")) for metadata in metadatas if metadata.get("document_id")}

        last_indexed: datetime | None = None
        indexed_values = [metadata.get("indexed_at") for metadata in metadatas if metadata.get("indexed_at")]
        if indexed_values:
            last_indexed = max(datetime.fromisoformat(str(value)) for value in indexed_values)

        return IndexStats(
            documents=len(document_ids),
            chunks=chunk_count,
            collection_name=self.collection_name,
            last_indexed=last_indexed,
        )

    def clear(self) -> None:
        client = self._create_client()
        try:
            client.delete_collection(self.collection_name)
        except ValueError:
            return

    def _create_client(self) -> Any:
        try:
            import chromadb
        except ModuleNotFoundError as exc:
            raise RuntimeError("chromadb is required to use VectorStore") from exc

        return chromadb.PersistentClient(path=str(self.persist_dir))

    def _serialize_chunk_metadata(self, chunk: Chunk) -> dict[str, str | int | float | bool]:
        metadata = {
            "document_id": chunk.document_id,
            "source": chunk.metadata.source,
            "title": chunk.metadata.title,
            "chunk_index": chunk.chunk_index,
        }
        if chunk.metadata.total_chunks is not None:
            metadata["total_chunks"] = chunk.metadata.total_chunks
        if chunk.metadata.document_hash is not None:
            metadata["document_hash"] = chunk.metadata.document_hash
        if chunk.metadata.indexed_at is not None:
            metadata["indexed_at"] = chunk.metadata.indexed_at.isoformat()
        if chunk.metadata.mtime is not None:
            metadata["mtime"] = chunk.metadata.mtime
        for key, value in chunk.metadata.extra.items():
            if value is None:
                continue
            metadata[key] = value if isinstance(value, (str, int, float, bool)) else json.dumps(value, sort_keys=True)
        return metadata

    def _deserialize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        ignored = {
            "document_id",
            "source",
            "title",
            "chunk_index",
            "total_chunks",
            "document_hash",
            "indexed_at",
            "mtime",
        }
        return {key: value for key, value in metadata.items() if key not in ignored}

    def _build_result_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        result_metadata = self._deserialize_metadata(metadata)
        for key in ("source", "title", "total_chunks", "document_hash", "indexed_at", "mtime"):
            if metadata.get(key) is not None:
                result_metadata[key] = metadata[key]
        return result_metadata


__all__ = ["VectorStore"]

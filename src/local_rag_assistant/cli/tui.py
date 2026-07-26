"""Human-readable terminal renderers for operator-focused CLI commands."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from textwrap import shorten
from typing import Any

from local_rag_assistant.config import Config
from local_rag_assistant.ingestion.models import IngestionDocumentRecord
from local_rag_assistant.models import ChatResponse, Chunk, IndexStats, SearchResult


def render_status(stats: IndexStats, documents: list[IngestionDocumentRecord], config: Config) -> str:
    counts = _status_counts(documents)
    lines = [
        "Runtime Status",
        f"Collection: {stats.collection_name}",
        f"Documents in index: {stats.documents}",
        f"Chunks in index: {stats.chunks}",
        f"Registry records: {len(documents)}",
        f"Indexed: {counts['indexed']}  Pending: {counts['pending']}  Failed: {counts['failed']}",
        f"Last indexed: {_format_datetime(stats.last_indexed)}",
        f"Chroma DB: {config.chroma_db_path}",
        f"Registry DB: {config.ingestion_registry_path}",
    ]
    return "\n".join(lines)


def render_document_list(documents: list[IngestionDocumentRecord], *, total_documents: int) -> str:
    lines = [f"Documents ({len(documents)}/{total_documents})"]
    if not documents:
        lines.append("No documents recorded.")
        return "\n".join(lines)

    for index, document in enumerate(documents, start=1):
        title = document.title or Path(document.source_path).name
        lines.extend(
            [
                f"{index}. [{document.status}] {title} ({document.document_id})",
                f"   source: {document.source_path}",
                f"   updated: {_format_datetime(document.updated_at)}",
            ]
        )
    return "\n".join(lines)


def render_document_detail(record: IngestionDocumentRecord, chunks: list[Chunk], *, chunk_limit: int = 3) -> str:
    lines = [
        f"Document {record.document_id}",
        f"Status: {record.status}",
        f"Title: {record.title or '-'}",
        f"Source: {record.source_path}",
        f"File type: {record.file_type}",
        f"Chunk count: {len(chunks)}",
        f"Created: {_format_datetime(record.created_at)}",
        f"Updated: {_format_datetime(record.updated_at)}",
    ]

    if record.error_message:
        lines.append(f"Error: {record.error_message}")

    if record.metadata:
        lines.append("Metadata:")
        for key, value in sorted(record.metadata.items()):
            lines.append(f"  {key}: {_format_value(value)}")

    lines.append("Chunks:")
    if not chunks:
        lines.append("  No stored chunks found.")
        return "\n".join(lines)

    for chunk in chunks[:chunk_limit]:
        lines.append(f"  [{chunk.chunk_index}] {shorten(chunk.content.replace(chr(10), ' '), width=120, placeholder='...')}")
    remaining = len(chunks) - min(len(chunks), chunk_limit)
    if remaining > 0:
        lines.append(f"  ... {remaining} more chunk(s)")
    return "\n".join(lines)


def render_search_result(result: SearchResult) -> str:
    lines = [f"Search: {result.query}", f"Results: {result.total_results}"]
    if not result.results:
        lines.append("No matches found.")
        return "\n".join(lines)

    for index, item in enumerate(result.results, start=1):
        source = item.metadata.get("source") or item.metadata.get("path") or item.document_id
        title = item.metadata.get("title") or "-"
        excerpt = shorten(item.text.replace("\n", " "), width=140, placeholder="...")
        lines.extend(
            [
                f"{index}. [{item.score:.2f}] {source}#chunk-{item.chunk_index}",
                f"   title: {title}",
                f"   {excerpt}",
            ]
        )
    return "\n".join(lines)


def render_chat_response(response: ChatResponse) -> str:
    lines = [f"Question: {response.query}", "Answer:", response.answer]
    lines.append("Sources:")
    if not response.sources:
        lines.append("No supporting sources returned.")
        return "\n".join(lines)

    for index, source in enumerate(response.sources, start=1):
        score = f"[{source.score:.2f}] " if source.score is not None else ""
        excerpt = shorten(source.excerpt.replace("\n", " "), width=140, placeholder="...")
        lines.extend(
            [
                f"{index}. {score}{source.source}#chunk-{source.chunk_index}",
                f"   title: {source.title or '-'}",
                f"   {excerpt}",
            ]
        )
    return "\n".join(lines)


def render_diagnostics(config: Config, stats: IndexStats, documents: list[IngestionDocumentRecord]) -> str:
    counts = _status_counts(documents)
    lines = [
        "Diagnostics",
        f"Collection: {stats.collection_name}",
        f"Documents: {stats.documents}",
        f"Chunks: {stats.chunks}",
        f"Last indexed: {_format_datetime(stats.last_indexed)}",
        f"Chroma DB: {_path_status(config.chroma_db_path, expect='dir')}",
        f"Registry DB: {_path_status(config.ingestion_registry_path, expect='file')}",
        f"Embedding model: {_path_status(config.embedding_model_path, expect='file')}",
        f"LLM model: {_path_status(config.llm_model_path, expect='file')}",
        f"Registry summary: indexed={counts['indexed']} pending={counts['pending']} failed={counts['failed']}",
    ]
    return "\n".join(lines)


def _status_counts(documents: Iterable[IngestionDocumentRecord]) -> dict[str, int]:
    counts = {"pending": 0, "indexed": 0, "failed": 0}
    for document in documents:
        counts[document.status] += 1
    return counts


def _format_datetime(value: datetime | None) -> str:
    return value.isoformat(timespec="seconds") if value is not None else "never"


def _format_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _path_status(path: Path, *, expect: str) -> str:
    exists = path.is_dir() if expect == "dir" else path.is_file()
    state = "ok" if exists else "missing"
    return f"{path} ({state})"


__all__ = [
    "render_chat_response",
    "render_diagnostics",
    "render_document_detail",
    "render_document_list",
    "render_search_result",
    "render_status",
]

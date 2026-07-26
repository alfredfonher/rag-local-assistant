"""Public package interface for local_rag_assistant."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatSource,
    Chunk,
    ChunkMetadata,
    ChunkResult,
    Document,
    IndexRequest,
    IndexResponse,
    IndexStats,
    SearchResult,
)

__version__ = "1.0.0"

_LAZY_EXPORTS = {
    "Orchestrator": "local_rag_assistant.orchestrator",
    "Bootstrap": "local_rag_assistant.bootstrap",
    "DocumentLoader": "local_rag_assistant.document_loader",
    "VectorStore": "local_rag_assistant.vectorstore",
    "EmbeddingModel": "local_rag_assistant.embedding",
    "LLM": "local_rag_assistant.llm",
    "Indexer": "local_rag_assistant.indexer",
}

__all__ = [
    "__version__",
    "Bootstrap",
    "Orchestrator",
    "DocumentLoader",
    "VectorStore",
    "EmbeddingModel",
    "LLM",
    "Indexer",
    "Chunk",
    "ChunkMetadata",
    "Document",
    "SearchResult",
    "ChunkResult",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ChatSource",
    "IndexRequest",
    "IndexResponse",
    "IndexStats",
]


def __getattr__(name: str) -> Any:
    """Lazily expose components that land in later PR slices."""
    module_name = _LAZY_EXPORTS.get(name)
    if not module_name:
        raise AttributeError(f"module 'local_rag_assistant' has no attribute {name!r}")

    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value

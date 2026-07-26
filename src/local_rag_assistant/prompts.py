"""Prompt building helpers for grounded chat flows."""

from __future__ import annotations

from collections.abc import Sequence

from local_rag_assistant.models import ChatMessage, ChunkResult

DEFAULT_CHAT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer questions using ONLY the provided context. "
    "If the answer is not supported by the context, say that you do not have enough "
    "information from the indexed documents. Cite sources with [Source: filename] when applicable."
)
NO_DOCUMENTS_MESSAGE = "No documents have been indexed yet. Please run indexing first."
NO_RELEVANT_CONTEXT_MESSAGE = "I couldn't find relevant indexed context to answer that question."


def build_grounded_messages(
    question: str,
    results: Sequence[ChunkResult],
    *,
    system_prompt: str | None = None,
) -> list[ChatMessage]:
    """Build chat messages for a context-grounded answer."""
    prompt = system_prompt or DEFAULT_CHAT_SYSTEM_PROMPT
    context_blocks = "\n---\n".join(_format_context_block(result) for result in results)
    user_content = f"Context:\n{context_blocks}\n\nQuestion: {question.strip()}"
    return [
        ChatMessage(role="system", content=prompt),
        ChatMessage(role="user", content=user_content),
    ]


def fallback_answer(*, has_indexed_documents: bool) -> str:
    """Return the appropriate non-LLM fallback answer."""
    if has_indexed_documents:
        return NO_RELEVANT_CONTEXT_MESSAGE
    return NO_DOCUMENTS_MESSAGE


def _format_context_block(result: ChunkResult) -> str:
    source = str(result.metadata.get("source") or "unknown")
    return f"[Source: {source}]\n{result.text.strip()}"


__all__ = [
    "DEFAULT_CHAT_SYSTEM_PROMPT",
    "NO_DOCUMENTS_MESSAGE",
    "NO_RELEVANT_CONTEXT_MESSAGE",
    "build_grounded_messages",
    "fallback_answer",
]

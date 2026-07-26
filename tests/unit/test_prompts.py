from __future__ import annotations

from local_rag_assistant.models import ChunkResult
from local_rag_assistant.prompts import (
    DEFAULT_CHAT_SYSTEM_PROMPT,
    NO_DOCUMENTS_MESSAGE,
    NO_RELEVANT_CONTEXT_MESSAGE,
    build_grounded_messages,
    fallback_answer,
)


def test_build_grounded_messages_includes_system_context_and_question() -> None:
    results = [
        ChunkResult(
            text="RAG combines retrieval and generation.",
            document_id="doc-1",
            chunk_index=0,
            score=0.9,
            metadata={"source": "notes/rag.md", "title": "RAG"},
        )
    ]

    messages = build_grounded_messages("What is RAG?", results)

    assert messages[0].role == "system"
    assert messages[0].content == DEFAULT_CHAT_SYSTEM_PROMPT
    assert "[Source: notes/rag.md]" in messages[1].content
    assert "RAG combines retrieval and generation." in messages[1].content
    assert messages[1].content.endswith("Question: What is RAG?")


def test_fallback_answer_distinguishes_empty_index_from_empty_results() -> None:
    assert fallback_answer(has_indexed_documents=False) == NO_DOCUMENTS_MESSAGE
    assert fallback_answer(has_indexed_documents=True) == NO_RELEVANT_CONTEXT_MESSAGE

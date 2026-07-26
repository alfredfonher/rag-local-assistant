"""High-level coordination for indexing, retrieval, and grounded chat."""

from __future__ import annotations

from collections.abc import Sequence

from local_rag_assistant.config import Config
from local_rag_assistant.indexer import Indexer
from local_rag_assistant.models import ChatResponse, ChatSource, ChunkResult, IndexResponse, IndexStats, SearchResult
from local_rag_assistant.prompts import build_grounded_messages, fallback_answer
from local_rag_assistant.vectorstore import VectorStore


class Orchestrator:
    """Coordinate indexing, semantic search, and grounded chat."""

    def __init__(
        self,
        config: Config,
        vectorstore: VectorStore,
        embedding_model: object | None,
        llm: object | None,
        *,
        indexer: Indexer | None = None,
    ) -> None:
        self._config = config
        self._vectorstore = vectorstore
        self._embedding_model = embedding_model
        self._llm = llm
        self._indexer = indexer or (Indexer(config, vectorstore, embedding_model) if embedding_model is not None else None)

    def index(
        self,
        path: str,
        *,
        recursive: bool = True,
        extensions: list[str] | None = None,
    ) -> IndexResponse:
        if self._indexer is None:
            raise RuntimeError("Indexing requires an initialized embedding model.")
        return self._indexer.index(path, recursive=recursive, extensions=extensions)

    def search(self, query: str, *, top_k: int | None = None) -> SearchResult:
        clean_query = _clean_query(query)
        results = self._retrieve(clean_query, top_k=top_k or self._config.default_top_k)
        return SearchResult(results=results, query=clean_query, total_results=len(results))

    def rag_search(self, query: str, top_k: int = 5) -> SearchResult:
        return self.search(query, top_k=top_k)

    def chat(
        self,
        question: str,
        *,
        top_k: int | None = None,
        system_prompt: str | None = None,
    ) -> ChatResponse:
        clean_question = _clean_query(question)
        stats = self.stats()
        if stats.chunks == 0:
            return ChatResponse(
                answer=fallback_answer(has_indexed_documents=False),
                sources=[],
                query=clean_question,
            )

        results = self._retrieve(clean_question, top_k=top_k or min(3, self._config.default_top_k))
        if not results:
            return ChatResponse(
                answer=fallback_answer(has_indexed_documents=True),
                sources=[],
                query=clean_question,
            )

        messages = build_grounded_messages(clean_question, results, system_prompt=system_prompt)
        answer = self._generate_answer(messages)
        return ChatResponse(
            answer=answer,
            sources=[self._build_source(result) for result in results],
            query=clean_question,
        )

    def rag_chat(self, question: str, top_k: int = 3, system_prompt: str | None = None) -> ChatResponse:
        return self.chat(question, top_k=top_k, system_prompt=system_prompt)

    def stats(self) -> IndexStats:
        return self._vectorstore.get_stats()

    def _retrieve(self, query: str, *, top_k: int) -> list[ChunkResult]:
        embedding_model = self._require_embedding_model()
        if hasattr(embedding_model, "encode_query"):
            query_embedding = embedding_model.encode_query(query)
        else:
            query_embedding = embedding_model.encode_single(query)
        return self._vectorstore.search(query_embedding, top_k=top_k)

    def _generate_answer(self, messages: Sequence[object]) -> str:
        llm = self._require_llm()

        if hasattr(llm, "chat"):
            return llm.chat(messages)

        if hasattr(llm, "generate"):
            system_prompt = None
            prompt = ""
            if len(messages) >= 2:
                first = messages[0]
                second = messages[1]
                system_prompt = getattr(first, "content", None) if getattr(first, "role", None) == "system" else None
                prompt = getattr(second, "content", "")
            return llm.generate(prompt, system_prompt=system_prompt)

        raise RuntimeError("LLM adapter must expose chat() or generate().")

    def _require_embedding_model(self) -> object:
        if self._embedding_model is None:
            raise RuntimeError("Search requires an initialized embedding model.")
        return self._embedding_model

    def _require_llm(self) -> object:
        if self._llm is None:
            raise RuntimeError("Chat requires an initialized LLM runtime.")
        return self._llm

    def _build_source(self, result: ChunkResult) -> ChatSource:
        source = str(result.metadata.get("source") or result.metadata.get("path") or "unknown")
        title = str(result.metadata.get("title") or "")
        return ChatSource(
            source=source,
            chunk_index=result.chunk_index,
            excerpt=result.text[:200],
            score=result.score,
            title=title,
            metadata=result.metadata,
        )


def _clean_query(query: str) -> str:
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("query must not be empty")
    return clean_query


__all__ = ["Orchestrator"]

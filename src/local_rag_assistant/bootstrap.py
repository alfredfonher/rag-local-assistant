"""Central runtime wiring for CLI and API adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from local_rag_assistant.config import Config
from local_rag_assistant.embedding import EmbeddingModel
from local_rag_assistant.ingestion.jobs import InMemoryIngestionJobStore
from local_rag_assistant.indexer import Indexer
from local_rag_assistant.ingestion.progress import IngestionProgressSink
from local_rag_assistant.ingestion.registry import IngestionRegistry
from local_rag_assistant.ingestion.service import IngestionService
from local_rag_assistant.llm import LLM
from local_rag_assistant.orchestrator import Orchestrator
from local_rag_assistant.vectorstore import VectorStore


class Bootstrap:
    """Lazy dependency container for application entrypoints."""

    def __init__(
        self,
        config: Config,
        *,
        vectorstore_factory: Callable[..., VectorStore] = VectorStore,
        embedding_factory: Callable[..., object] = EmbeddingModel,
        llm_factory: Callable[..., object] = LLM,
    ) -> None:
        self._config = config
        self._vectorstore_factory = vectorstore_factory
        self._embedding_factory = embedding_factory
        self._llm_factory = llm_factory
        self._vectorstore: VectorStore | None = None
        self._embedding_model: object | None = None
        self._llm: object | None = None
        self._registry: IngestionRegistry | None = None
        self._indexer: Indexer | None = None
        self._ingestion_service: IngestionService | None = None
        self._ingestion_job_store: InMemoryIngestionJobStore | None = None

    @property
    def config(self) -> Config:
        return self._config

    def vectorstore(self) -> VectorStore:
        self._config.validate_for_runtime(require_embedding_model=False, require_llm_model=False)
        if self._vectorstore is None:
            self._vectorstore = self._vectorstore_factory(
                self._config.chroma_db_path,
                collection_name=self._config.chroma_collection_name,
            )
        return self._vectorstore

    def embedding_model(self) -> object:
        self._config.validate_for_runtime(require_embedding_model=True, require_llm_model=False)
        if self._embedding_model is None:
            self._embedding_model = self._embedding_factory(self._config.embedding_model_path)
        return self._embedding_model

    def llm(self) -> object:
        self._config.validate_for_runtime(require_embedding_model=False, require_llm_model=True)
        if self._llm is None:
            self._llm = self._llm_factory(self._config.llm_model_path)
        return self._llm

    def registry(self) -> IngestionRegistry:
        self._config.validate_for_runtime(require_embedding_model=False, require_llm_model=False)
        if self._registry is None:
            self._registry = IngestionRegistry(self._config.ingestion_registry_path)
            self._registry.init_schema()
        return self._registry

    def indexer(self) -> Indexer:
        if self._indexer is None:
            self._indexer = Indexer(
                self._config,
                self.vectorstore(),
                self.embedding_model(),
            )
        return self._indexer

    def ingestion_service(self, *, progress_sink: IngestionProgressSink | None = None) -> IngestionService:
        if progress_sink is not None:
            return IngestionService(self.registry(), self.indexer(), progress_sink=progress_sink)
        if self._ingestion_service is None:
            self._ingestion_service = IngestionService(self.registry(), self.indexer())
        return self._ingestion_service

    def ingestion_job_store(self) -> InMemoryIngestionJobStore:
        if self._ingestion_job_store is None:
            self._ingestion_job_store = InMemoryIngestionJobStore()
        return self._ingestion_job_store

    def build_orchestrator(
        self,
        *,
        require_embedding: bool = True,
        require_llm: bool = True,
    ) -> Orchestrator:
        return Orchestrator(
            self._config,
            self.vectorstore(),
            self.embedding_model() if require_embedding else None,
            self.llm() if require_llm else None,
        )


def build_bootstrap(
    config: Config,
    *,
    vectorstore_factory: Callable[..., VectorStore] = VectorStore,
    embedding_factory: Callable[..., object] = EmbeddingModel,
    llm_factory: Callable[..., object] = LLM,
) -> Bootstrap:
    return Bootstrap(
        config,
        vectorstore_factory=vectorstore_factory,
        embedding_factory=embedding_factory,
        llm_factory=llm_factory,
    )


def build_orchestrator(config: Config, *, require_embedding: bool = True, require_llm: bool = True) -> Orchestrator:
    return build_bootstrap(config).build_orchestrator(
        require_embedding=require_embedding,
        require_llm=require_llm,
    )


__all__ = ["Bootstrap", "build_bootstrap", "build_orchestrator"]

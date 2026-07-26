"""Embedding model wrapper using llama-cpp-python."""

from __future__ import annotations

import gc
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)

RAM_WARNING_THRESHOLD = 1.5 * 1024**3  # 1.5 GB


class EmbeddingModel:
    """Embedding model backed by llama-cpp-python (nomic-embed-text)."""

    def __init__(self, model_path: Path, n_ctx: int = 2048, n_threads: int = 1) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Embedding model not found: {model_path}")

        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_threads = n_threads
        self._llama = None
        self._load()

    def _load(self) -> None:
        try:
            import llama_cpp
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is required. Install with: pip install llama-cpp-python"
            ) from exc

        LOGGER.info("Loading embedding model: %s", self._model_path)
        self._llama = llama_cpp.Llama(
            model_path=str(self._model_path),
            n_ctx=self._n_ctx,
            n_threads=self._n_threads,
            embedding=True,
            verbose=False,
        )
        LOGGER.info("Embedding model loaded successfully")

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Encode a list of texts into embedding vectors.

        Uses the ``search_document: `` prefix required by nomic-embed-text.
        """
        if not texts:
            return []

        return self._encode_prefixed(texts, prefix="search_document: ", batch_size=batch_size)

    def encode_single(self, text: str) -> list[float]:
        """Encode a single text into an embedding vector."""
        results = self.encode([text])
        return results[0]

    def encode_query(self, text: str) -> list[float]:
        """Encode a single query with the retrieval-specific prefix."""
        results = self._encode_prefixed([text], prefix="search_query: ", batch_size=1)
        return results[0]

    def _encode_prefixed(self, texts: list[str], *, prefix: str, batch_size: int) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            prefixed = [f"{prefix}{text}" for text in batch]
            result = self._llama.embed(prefixed)
            all_embeddings.extend(result)
            gc.collect()

            try:
                import psutil

                available = psutil.virtual_memory().available
                if available < RAM_WARNING_THRESHOLD:
                    LOGGER.warning(
                        "Low RAM detected: %.2f GiB available. Consider reducing batch_size.",
                        available / 1024**3,
                    )
            except ImportError:
                pass

        return all_embeddings

    def __del__(self) -> None:
        self._llama = None
        gc.collect()

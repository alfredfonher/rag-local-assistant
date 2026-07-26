"""Shared pytest fixtures for foundation tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from local_rag_assistant.models import Chunk, ChunkMetadata


@pytest.fixture
def temp_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    contents = {
        "alpha.md": "# Alpha\n\nThis is the first sample document.",
        "beta.md": "---\ntitle: Beta\n---\n\nBeta content for indexing.",
        "gamma.md": "# Gamma\n\nParagraph one.\n\nParagraph two.",
        "notes/todo.md": "# Todo\n\n- task one\n- task two",
        "notes/ideas.md": "# Ideas\n\nRAG notes and experiments.",
    }
    for relative_path, content in contents.items():
        path = vault / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return vault


@pytest.fixture
def ingestion_fixture_dir() -> Path:
    return ROOT / "tests" / "fixtures" / "ingestion"


@pytest.fixture
def ingestion_fixture_paths(ingestion_fixture_dir: Path) -> list[Path]:
    return [
        ingestion_fixture_dir / "retrieval-handbook.txt",
        ingestion_fixture_dir / "system-architecture.md",
        ingestion_fixture_dir / "fixture-reference.pdf",
    ]


@pytest.fixture
def mock_embedding() -> object:
    class MockEmbedding:
        def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
            del batch_size
            return [[float(index)] * 4 for index, _ in enumerate(texts, start=1)]

        def encode_single(self, text: str) -> list[float]:
            return [float(len(text))] * 4

        def encode_query(self, text: str) -> list[float]:
            return [float(len(text) + 1)] * 4

    return MockEmbedding()


@pytest.fixture
def mock_llm() -> object:
    class MockLLM:
        def __init__(self) -> None:
            self.messages: list[object] = []

        def chat(self, messages: list[object], **_: object) -> str:
            self.messages = messages
            return "Mock grounded answer"

        def generate(self, prompt: str, **_: object) -> str:
            return f"Mock answer for: {prompt[:30]}"

    return MockLLM()


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            document_id="doc-1",
            chunk_index=index,
            content=f"Chunk {index}",
            metadata=ChunkMetadata(source="notes/doc.md", title="Doc", chunk_index=index),
        )
        for index in range(3)
    ]


@pytest.fixture
def real_models_skip() -> bool:
    embedding = ROOT / "models" / "nomic-embed-text-v1.5.f32.gguf"
    llm = ROOT / "models" / "gemma-3-1b-it-Q4_K_M.gguf"
    if not (embedding.exists() and llm.exists()):
        pytest.skip("Real GGUF models are not available in ./models")
    return True

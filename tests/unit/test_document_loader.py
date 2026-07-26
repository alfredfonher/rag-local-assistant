from __future__ import annotations

import hashlib
from pathlib import Path

from local_rag_assistant.document_loader import DocumentLoader, build_document_id


def test_discover_files_filters_supported_extensions(temp_vault: Path) -> None:
    (temp_vault / "ignore.png").write_bytes(b"png")

    loader = DocumentLoader()
    files = loader.discover_files(temp_vault)

    assert [path.relative_to(temp_vault).as_posix() for path in files] == [
        "alpha.md",
        "beta.md",
        "gamma.md",
        "notes/ideas.md",
        "notes/todo.md",
    ]


def test_load_document_normalizes_metadata_and_markdown_title(temp_vault: Path) -> None:
    loader = DocumentLoader()

    document = loader.load_document(temp_vault, temp_vault / "alpha.md")

    assert document.source == "alpha.md"
    assert document.title == "Alpha"
    assert document.metadata["path"] == "alpha.md"
    assert document.metadata["file_type"] == "markdown"
    assert isinstance(document.metadata["modified_time"], float)
    assert document.hash == document.metadata["checksum"]


def test_load_document_uses_frontmatter_title(temp_vault: Path) -> None:
    loader = DocumentLoader()

    document = loader.load_document(temp_vault, temp_vault / "beta.md")

    assert document.title == "Beta"
    assert "Beta content for indexing." in document.content


def test_build_document_id_is_deterministic(temp_vault: Path) -> None:
    path = temp_vault / "alpha.md"

    first = build_document_id(path)
    second = build_document_id(path)

    assert first == second
    assert first == hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()

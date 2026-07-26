from __future__ import annotations

from local_rag_assistant.utils.text_splitter import RecursiveCharacterTextSplitter, get_text_splitter


def _split(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def test_chunking_three_paragraphs_produces_multiple_chunks() -> None:
    text = "Paragraph A.\n\nParagraph B with more text.\n\nParagraph C concludes."
    chunks = _split(text, chunk_size=30, chunk_overlap=10)
    assert len(chunks) >= 2
    assert any("Paragraph A" in chunk for chunk in chunks)
    assert any("Paragraph C" in chunk for chunk in chunks)


def test_overlap_preserves_context() -> None:
    text = "Alpha beta gamma delta epsilon zeta eta theta iota kappa"
    chunks = _split(text, chunk_size=20, chunk_overlap=5)
    assert len(chunks) >= 2
    assert chunks[1][:5].strip() in chunks[0]


def test_empty_text_returns_empty_list() -> None:
    assert _split("", chunk_size=20, chunk_overlap=5) == []


def test_very_long_text_is_split() -> None:
    text = "A" * 300
    chunks = _split(text, chunk_size=64, chunk_overlap=8)
    assert len(chunks) > 1
    assert all(chunk for chunk in chunks)


def test_fallback_splitter_validates_arguments() -> None:
    try:
        RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=10)
    except ValueError:
        assert True
    else:
        raise AssertionError("Expected invalid overlap to raise ValueError")

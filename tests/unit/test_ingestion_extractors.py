from __future__ import annotations

from pathlib import Path

import pytest

from local_rag_assistant.ingestion.extractors import EmptyExtractedContentError, UnsupportedFileTypeError, extract_file


def test_extract_txt_file_returns_normalized_result(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("Plain text content\n", encoding="utf-8")

    result = extract_file(path)

    assert result.file_type == "text"
    assert result.title == "notes"
    assert result.content == "Plain text content"
    assert result.source_path == str(path.resolve())


def test_extract_markdown_file_uses_frontmatter_title_and_preserves_body(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_text(
        "---\ntitle: Getting Started\n---\n\n# Ignored Heading\n\nMarkdown body.",
        encoding="utf-8",
    )

    result = extract_file(path)

    assert result.file_type == "markdown"
    assert result.title == "Getting Started"
    assert result.content == "# Ignored Heading\n\nMarkdown body."


def test_extract_pdf_file_reads_text_and_page_count(tmp_path: Path) -> None:
    path = tmp_path / "sample.pdf"
    path.write_bytes(_build_pdf_bytes("Hello PDF"))

    result = extract_file(path)

    assert result.file_type == "pdf"
    assert result.title == "sample"
    assert "Hello PDF" in result.content
    assert result.metadata["pages"] == 1


def test_extract_file_rejects_unsupported_types(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(b"png")

    with pytest.raises(UnsupportedFileTypeError, match=r"Unsupported file type for ingestion: \.png"):
        extract_file(path)


def test_extract_pdf_file_rejects_empty_text(tmp_path: Path) -> None:
    path = tmp_path / "empty.pdf"
    path.write_bytes(_build_pdf_bytes(""))

    with pytest.raises(EmptyExtractedContentError, match=r"Extracted pdf content is empty or unusable"):
        extract_file(path)


def _build_pdf_bytes(text: str) -> bytes:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1")

    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        (
            b"3 0 obj\n"
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\n"
            b"endobj\n"
        ),
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        (
            b"5 0 obj\n"
            + f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1")
            + stream
            + b"\nendstream\nendobj\n"
        ),
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj)

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))

    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF"
        ).encode("latin-1")
    )
    return bytes(pdf)

"""File extractors for supported ingestion formats."""

from __future__ import annotations

import re
from pathlib import Path

from local_rag_assistant.ingestion.models import ExtractionResult

SUPPORTED_EXTENSIONS = (".txt", ".md", ".pdf")
_MARKDOWN_HEADING_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


class ExtractionError(ValueError):
    """Base exception for extraction failures."""


class UnsupportedFileTypeError(ExtractionError):
    """Raised when no extractor is available for a file type."""


class EmptyExtractedContentError(ExtractionError):
    """Raised when extracted content is empty or unusable."""


def extract_file(file_path: str | Path) -> ExtractionResult:
    """Extract a supported file into the normalized ingestion contract."""

    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Document file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".txt":
        return extract_text_file(path)
    if suffix == ".md":
        return extract_markdown_file(path)
    if suffix == ".pdf":
        return extract_pdf_file(path)

    supported = ", ".join(SUPPORTED_EXTENSIONS)
    raise UnsupportedFileTypeError(
        f"Unsupported file type for ingestion: {suffix or '<none>'}. Supported types: {supported}."
    )


def extract_text_file(file_path: str | Path) -> ExtractionResult:
    path = Path(file_path).expanduser().resolve()
    content = _require_extracted_content(path.read_text(encoding="utf-8"), path=path, file_type="text")
    return ExtractionResult(
        source_path=str(path),
        file_type="text",
        title=path.stem,
        content=content,
        metadata={"extension": path.suffix.lower()},
    )


def extract_markdown_file(file_path: str | Path) -> ExtractionResult:
    path = Path(file_path).expanduser().resolve()
    raw_text = path.read_text(encoding="utf-8")
    title, content = _parse_markdown(raw_text, fallback_title=path.stem)
    content = _require_extracted_content(content, path=path, file_type="markdown")
    return ExtractionResult(
        source_path=str(path),
        file_type="markdown",
        title=title,
        content=content,
        metadata={"extension": path.suffix.lower()},
    )


def extract_pdf_file(file_path: str | Path) -> ExtractionResult:
    path = Path(file_path).expanduser().resolve()

    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise ExtractionError(
            "PDF ingestion requires the 'pypdf' package. Install project dependencies before ingesting PDF files."
        ) from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    content = _require_extracted_content(
        "\n\n".join(page.strip() for page in pages if page.strip()),
        path=path,
        file_type="pdf",
    )

    metadata_title = getattr(reader.metadata, "title", None) if reader.metadata else None
    title = str(metadata_title).strip() if metadata_title else path.stem

    return ExtractionResult(
        source_path=str(path),
        file_type="pdf",
        title=title,
        content=content,
        metadata={
            "extension": path.suffix.lower(),
            "pages": len(reader.pages),
        },
    )


def _parse_markdown(raw_text: str, *, fallback_title: str) -> tuple[str, str]:
    try:
        import frontmatter
    except ModuleNotFoundError:
        frontmatter = None

    if frontmatter is not None:
        parsed = frontmatter.loads(raw_text)
        title = str(parsed.metadata.get("title") or "").strip() or _extract_markdown_title(
            parsed.content,
            fallback_title=fallback_title,
        )
        return title, parsed.content.strip()

    title = _extract_markdown_title(raw_text, fallback_title=fallback_title)
    return title, raw_text.strip()


def _extract_markdown_title(content: str, *, fallback_title: str) -> str:
    match = _MARKDOWN_HEADING_RE.search(content)
    if not match:
        return fallback_title
    return match.group(1).strip()


def _require_extracted_content(raw_content: str, *, path: Path, file_type: str) -> str:
    content = raw_content.strip()
    if content:
        return content
    raise EmptyExtractedContentError(
        f"Extracted {file_type} content is empty or unusable for ingestion: {path}"
    )


__all__ = [
    "SUPPORTED_EXTENSIONS",
    "ExtractionError",
    "EmptyExtractedContentError",
    "UnsupportedFileTypeError",
    "extract_file",
    "extract_markdown_file",
    "extract_pdf_file",
    "extract_text_file",
]

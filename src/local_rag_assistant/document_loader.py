"""Document discovery and loading helpers for local text sources."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from local_rag_assistant.models import Document

SUPPORTED_EXTENSIONS = (".md", ".txt")
_MARKDOWN_HEADING_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def build_document_id(path: str | Path) -> str:
    """Build a deterministic document id from an absolute file path."""
    normalized_path = str(Path(path).resolve())
    return hashlib.sha256(normalized_path.encode("utf-8")).hexdigest()


class DocumentLoader:
    """Discover and parse supported documents from a root directory."""

    def discover_files(
        self,
        root_path: str | Path,
        *,
        recursive: bool = True,
        extensions: list[str] | tuple[str, ...] | None = None,
    ) -> list[Path]:
        root = Path(root_path).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Document root does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Document root must be a directory: {root}")

        normalized_extensions = {
            extension.lower() if extension.startswith(".") else f".{extension.lower()}"
            for extension in (extensions or SUPPORTED_EXTENSIONS)
        }
        iterator = root.rglob("*") if recursive else root.glob("*")
        files = [path for path in iterator if path.is_file() and path.suffix.lower() in normalized_extensions]
        return sorted(files)

    def load_document(self, root_path: str | Path, file_path: str | Path) -> Document:
        root = Path(root_path).expanduser().resolve()
        path = Path(file_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Document file does not exist: {path}")

        raw_text = path.read_text(encoding="utf-8")
        file_type = self._normalize_file_type(path.suffix)
        title = path.stem
        content = raw_text

        if path.suffix.lower() == ".md":
            title, content = self._parse_markdown(raw_text, fallback_title=title)

        metadata = {
            "path": path.relative_to(root).as_posix(),
            "file_type": file_type,
            "modified_time": path.stat().st_mtime,
            "checksum": self._checksum(path),
        }

        return Document(
            id=build_document_id(path),
            source=metadata["path"],
            title=title,
            content=content,
            metadata=metadata,
            hash=metadata["checksum"],
        )

    def load_documents(
        self,
        root_path: str | Path,
        *,
        recursive: bool = True,
        extensions: list[str] | tuple[str, ...] | None = None,
    ) -> list[Document]:
        return [
            self.load_document(root_path, file_path)
            for file_path in self.discover_files(root_path, recursive=recursive, extensions=extensions)
        ]

    def _parse_markdown(self, raw_text: str, *, fallback_title: str) -> tuple[str, str]:
        try:
            import frontmatter
        except ModuleNotFoundError:
            frontmatter = None

        if frontmatter is not None:
            parsed = frontmatter.loads(raw_text)
            title = str(parsed.metadata.get("title") or "").strip() or self._extract_markdown_title(
                parsed.content,
                fallback_title=fallback_title,
            )
            return title, parsed.content.strip()

        title = self._extract_markdown_title(raw_text, fallback_title=fallback_title)
        return title, raw_text.strip()

    def _extract_markdown_title(self, content: str, *, fallback_title: str) -> str:
        match = _MARKDOWN_HEADING_RE.search(content)
        if not match:
            return fallback_title
        return match.group(1).strip()

    def _normalize_file_type(self, suffix: str) -> str:
        mapping = {".md": "markdown", ".txt": "text"}
        return mapping.get(suffix.lower(), suffix.lower().lstrip("."))

    def _checksum(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()


__all__ = ["SUPPORTED_EXTENSIONS", "DocumentLoader", "build_document_id"]

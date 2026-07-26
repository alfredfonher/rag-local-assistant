"""Text splitting helpers used by the indexing pipeline."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter as _LangChainSplitter
except ModuleNotFoundError:  # pragma: no cover - fallback is covered instead
    _LangChainSplitter = None


@dataclass
class RecursiveCharacterTextSplitter:
    """Small compatible fallback for recursive character splitting."""

    chunk_size: int = 512
    chunk_overlap: int = 64
    separators: list[str] | None = None

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be zero or positive")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.separators = list(self.separators or DEFAULT_SEPARATORS)

    def split_text(self, text: str) -> list[str]:
        if not text:
            return []

        pieces = [segment.strip() for segment in self._split_recursive(text, self.separators)]
        pieces = [piece for piece in pieces if piece]
        if not pieces:
            return []

        chunks: list[str] = []
        current = ""

        for piece in pieces:
            candidate = piece if not current else f"{current}\n\n{piece}".strip()
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current)
            current = piece

            while len(current) > self.chunk_size:
                chunks.append(current[: self.chunk_size])
                current = current[self.chunk_size - self.chunk_overlap :]

        if current:
            chunks.append(current)

        return self._apply_overlap(chunks)

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size or not separators:
            return [text]

        separator = separators[0]
        if separator == "":
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        parts = text.split(separator)
        if len(parts) == 1:
            return self._split_recursive(text, separators[1:])

        collected: list[str] = []
        for part in parts:
            stripped = part.strip()
            if not stripped:
                continue
            if len(stripped) <= self.chunk_size:
                collected.append(stripped)
            else:
                collected.extend(self._split_recursive(stripped, separators[1:]))
        return collected

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        if not chunks:
            return []
        if self.chunk_overlap == 0:
            return chunks

        overlapped = [chunks[0]]
        for chunk in chunks[1:]:
            previous = overlapped[-1]
            prefix = previous[-self.chunk_overlap :].strip()
            candidate = f"{prefix} {chunk}".strip() if prefix else chunk
            overlapped.append(candidate if len(candidate) <= self.chunk_size + self.chunk_overlap else candidate[-(self.chunk_size + self.chunk_overlap) :])
        return overlapped


def get_text_splitter(chunk_size: int = 512, chunk_overlap: int = 64) -> RecursiveCharacterTextSplitter | object:
    """Build the project's default recursive text splitter."""
    if _LangChainSplitter is not None:
        return _LangChainSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=DEFAULT_SEPARATORS,
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=DEFAULT_SEPARATORS,
    )


__all__ = ["DEFAULT_SEPARATORS", "RecursiveCharacterTextSplitter", "get_text_splitter"]

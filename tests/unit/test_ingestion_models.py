from __future__ import annotations

import pytest
from pydantic import ValidationError

from local_rag_assistant.ingestion.models import ExtractionResult


def test_extraction_result_rejects_empty_content() -> None:
    with pytest.raises(ValidationError):
        ExtractionResult(source_path="/tmp/doc.txt", file_type="text", title="Doc", content="   ")


def test_extraction_result_normalizes_title() -> None:
    result = ExtractionResult(
        source_path="/tmp/doc.txt",
        file_type="text",
        title="  Notes  ",
        content="Hello",
    )

    assert result.title == "Notes"

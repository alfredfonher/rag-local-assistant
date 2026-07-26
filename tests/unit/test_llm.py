from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from local_rag_assistant.llm import DEFAULT_LLM_DOWNLOAD_URL, LLM
from local_rag_assistant.models import ChatMessage


class _FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create_chat_completion(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": "Grounded answer"}}]}


def test_llm_loads_runtime_lazily_with_injected_factory(tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_text("stub", encoding="utf-8")
    runtime = _FakeRuntime()
    factory_calls: list[dict[str, object]] = []

    def factory(**kwargs: object) -> _FakeRuntime:
        factory_calls.append(kwargs)
        return runtime

    llm = LLM(model_path, runtime_factory=factory)

    assert factory_calls == []
    answer = llm.chat([ChatMessage(role="user", content="Hello")])

    assert answer == "Grounded answer"
    assert factory_calls[0]["model_path"] == str(model_path)
    assert runtime.calls[0]["messages"] == [{"role": "user", "content": "Hello"}]


def test_llm_reports_missing_model_path() -> None:
    llm = LLM("/missing/model.gguf")

    with pytest.raises(RuntimeError, match="LLM model not found") as exc_info:
        llm.chat([ChatMessage(role="user", content="Hello")])

    assert DEFAULT_LLM_DOWNLOAD_URL in str(exc_info.value)


def test_llm_reports_missing_llama_cpp_dependency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_text("stub", encoding="utf-8")
    original = sys.modules.get("llama_cpp")
    monkeypatch.delitem(sys.modules, "llama_cpp", raising=False)

    real_import = __import__

    def raising_import(name: str, *args: object, **kwargs: object) -> ModuleType:
        if name == "llama_cpp":
            raise ModuleNotFoundError("No module named 'llama_cpp'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", raising_import)

    llm = LLM(model_path)

    with pytest.raises(RuntimeError, match="llama-cpp-python is required"):
        llm.chat([ChatMessage(role="user", content="Hello")])

    if original is not None:
        sys.modules["llama_cpp"] = original

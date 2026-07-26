"""Local LLM wrapper with lazy runtime loading."""

from __future__ import annotations

import gc
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from local_rag_assistant.models import ChatMessage

DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.3
DEFAULT_TOP_P = 0.9
DEFAULT_REPEAT_PENALTY = 1.1
DEFAULT_STOP_SEQUENCES = ["<|im_end|>", "<|endoftext|>"]
DEFAULT_LLM_DOWNLOAD_URL = "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF"


class LLM:
    """Thin adapter around llama.cpp chat generation."""

    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        *,
        runtime: Any | None = None,
        runtime_factory: Callable[..., Any] | None = None,
        model_url: str = DEFAULT_LLM_DOWNLOAD_URL,
    ) -> None:
        self._model_path = Path(model_path)
        self._n_ctx = n_ctx
        self._n_threads = n_threads
        self._runtime = runtime
        self._runtime_factory = runtime_factory
        self._model_url = model_url

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
        stop: Sequence[str] | None = None,
    ) -> str:
        """Generate a completion from a user prompt."""
        messages = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=prompt))
        return self.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop=stop,
        )

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
        stop: Sequence[str] | None = None,
    ) -> str:
        """Generate a full non-streaming chat response."""
        runtime = self._ensure_runtime()
        payload = [message.model_dump() if isinstance(message, ChatMessage) else dict(message) for message in messages]
        stop_sequences = list(stop or DEFAULT_STOP_SEQUENCES)

        try:
            if hasattr(runtime, "create_chat_completion"):
                response = runtime.create_chat_completion(
                    messages=payload,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    stop=stop_sequences,
                    stream=False,
                )
                return _extract_chat_content(response)

            if hasattr(runtime, "create_completion"):
                prompt = _messages_to_prompt(payload)
                response = runtime.create_completion(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    stop=stop_sequences,
                    stream=False,
                )
                return _extract_completion_text(response)
        except Exception as exc:
            raise RuntimeError(f"LLM generation failed: {exc}") from exc

        raise RuntimeError("LLM runtime does not support chat or completion generation.")

    def unload(self) -> None:
        self._runtime = None
        gc.collect()

    def _ensure_runtime(self) -> Any:
        if self._runtime is not None:
            return self._runtime

        if self._runtime_factory is not None:
            try:
                self._runtime = self._runtime_factory(
                    model_path=str(self._model_path),
                    n_ctx=self._n_ctx,
                    n_threads=self._n_threads,
                    verbose=False,
                )
            except Exception as exc:
                raise RuntimeError(f"Failed to load LLM runtime: {exc}") from exc
            return self._runtime

        self._validate_model_path()

        try:
            from llama_cpp import Llama
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "llama-cpp-python is required to use the local LLM runtime. "
                "Install it or inject a test runtime."
            ) from exc

        try:
            self._runtime = Llama(
                model_path=str(self._model_path),
                n_ctx=self._n_ctx,
                n_threads=self._n_threads,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load LLM model from {self._model_path}: {exc}") from exc

        return self._runtime

    def _validate_model_path(self) -> None:
        if self._model_path.is_file():
            return
        raise RuntimeError(
            f"LLM model not found at {self._model_path}. Download from: {self._model_url}"
        )

    def __del__(self) -> None:
        self.unload()


def _messages_to_prompt(messages: Sequence[dict[str, str]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        lines.append(f"<|{role}|>\n{content}")
    lines.append("<|assistant|>")
    return "\n".join(lines)


def _extract_chat_content(response: Any) -> str:
    choices = response.get("choices") if isinstance(response, dict) else None
    if not choices:
        raise RuntimeError("LLM chat response did not include choices.")
    message = choices[0].get("message", {})
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("LLM chat response did not include message content.")
    return content.strip()


def _extract_completion_text(response: Any) -> str:
    choices = response.get("choices") if isinstance(response, dict) else None
    if not choices:
        raise RuntimeError("LLM completion response did not include choices.")
    text = choices[0].get("text")
    if not isinstance(text, str):
        raise RuntimeError("LLM completion response did not include text.")
    return text.strip()


__all__ = [
    "DEFAULT_LLM_DOWNLOAD_URL",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_REPEAT_PENALTY",
    "DEFAULT_STOP_SEQUENCES",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_P",
    "LLM",
]

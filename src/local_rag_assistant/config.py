"""Application configuration and runtime tuning helpers."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - dependency declared in pyproject
    yaml = None

try:
    import psutil
except ModuleNotFoundError:  # pragma: no cover - dependency declared in pyproject
    psutil = None

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ModuleNotFoundError:  # pragma: no cover - compatibility shim
    from pydantic import BaseModel as BaseSettings  # type: ignore[misc]

    class SettingsConfigDict(dict):
        """Fallback placeholder when pydantic-settings is unavailable."""

from pydantic import Field, model_validator

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_FILE = "local-rag-assistant.yaml"
LOW_RAM_THRESHOLD_BYTES = 8 * 1024**3


class RuntimeConfigurationError(RuntimeError):
    """Raised when local runtime assets are missing or unreadable."""


def _load_yaml_config(path: str | os.PathLike[str] | None) -> dict[str, Any]:
    if not path:
        return {}

    config_path = Path(path)
    if not config_path.exists():
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to read local-rag-assistant.yaml configuration files.")

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file {config_path} must contain a YAML mapping.")
    return data


class Config(BaseSettings):
    """Centralized settings with env, YAML, and CLI override support."""

    chroma_db_path: Path = Field(default=Path("./data/chroma_db"))
    ingestion_data_path: Path = Field(default=Path("./data/ingestion"))
    ingestion_registry_path: Path = Field(default=Path("./data/ingestion/registry.sqlite3"))
    ingestion_storage_path: Path = Field(default=Path("./data/ingestion/storage"))
    chroma_collection_name: str = Field(default="documents")
    embedding_model_path: Path = Field(default=Path("./models/nomic-embed-text-v1.5.f32.gguf"))
    llm_model_path: Path = Field(default=Path("./models/gemma-3-1b-it-Q4_K_M.gguf"))
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=64, ge=0)
    embedding_batch_size: int = Field(default=32, gt=0)
    default_top_k: int = Field(default=5, ge=1, le=50)
    embedding_dim: int = Field(default=768, gt=0)
    config_file: Path | None = Field(default=None, exclude=True)

    model_config = SettingsConfigDict(
        env_prefix="LOCAL_RAG_ASSISTANT_",
        env_file=".env",
        extra="ignore",
        validate_assignment=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: Any,
        env_settings: Any,
        dotenv_settings: Any,
        file_secret_settings: Any,
    ) -> tuple[Any, ...]:
        def yaml_source() -> dict[str, Any]:
            requested_file = init_settings.init_kwargs.get("config_file") if hasattr(init_settings, "init_kwargs") else None
            config_path = requested_file or DEFAULT_CONFIG_FILE
            return _load_yaml_config(config_path)

        return (
            init_settings,
            env_settings,
            yaml_source,
            dotenv_settings,
            file_secret_settings,
        )

    @model_validator(mode="after")
    def _validate_settings(self) -> "Config":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        return self

    def validate_for_runtime(
        self,
        *,
        require_models: bool | None = None,
        require_embedding_model: bool | None = None,
        require_llm_model: bool | None = None,
    ) -> "Config":
        """Validate runtime-sensitive paths and numeric values."""
        if require_models is not None:
            require_embedding_model = require_models
            require_llm_model = require_models

        checks = []
        if require_embedding_model:
            checks.append(
                (
                    "embedding_model_path",
                    "Embedding model",
                    "--embedding-model-path",
                    "LOCAL_RAG_ASSISTANT_EMBEDDING_MODEL_PATH",
                )
            )
        if require_llm_model:
            checks.append(
                (
                    "llm_model_path",
                    "LLM model",
                    "--llm-model-path",
                    "LOCAL_RAG_ASSISTANT_LLM_MODEL_PATH",
                )
            )

        for attr_name, label, cli_flag, env_var in checks:
            path = Path(getattr(self, attr_name))
            if not path.is_file():
                raise RuntimeConfigurationError(
                    f"{label} is not available at {path}. "
                    f"Set {cli_flag} or {env_var} to a readable local model file."
                )
            if not os.access(path, os.R_OK):
                raise RuntimeConfigurationError(
                    f"{label} is not readable at {path}. "
                    f"Fix file permissions or update {cli_flag} / {env_var}."
                )

        self.chroma_db_path.mkdir(parents=True, exist_ok=True)
        self.ingestion_data_path.mkdir(parents=True, exist_ok=True)
        self.ingestion_registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.ingestion_storage_path.mkdir(parents=True, exist_ok=True)
        return self

    def with_ram_aware_batch_size(self) -> "Config":
        """Return a copy with a safer batch size when RAM is limited."""
        available_memory = None
        if psutil is not None:
            available_memory = psutil.virtual_memory().available

        if available_memory is None or available_memory >= LOW_RAM_THRESHOLD_BYTES:
            return self

        factor = max(0.25, available_memory / LOW_RAM_THRESHOLD_BYTES)
        adjusted_batch_size = max(1, int(self.embedding_batch_size * factor))
        if adjusted_batch_size < self.embedding_batch_size:
            LOGGER.warning(
                "Limited RAM detected (%.2f GiB available). Reducing embedding_batch_size from %s to %s.",
                available_memory / 1024**3,
                self.embedding_batch_size,
                adjusted_batch_size,
            )
        return self.model_copy(update={"embedding_batch_size": adjusted_batch_size})

    @classmethod
    def from_cli(cls, config_file: str | os.PathLike[str] | None = None, **overrides: Any) -> "Config":
        """Build configuration honoring CLI overrides first."""
        clean_overrides = {key: value for key, value in overrides.items() if value is not None}
        settings = cls(config_file=config_file, **clean_overrides)
        return settings.with_ram_aware_batch_size()


__all__ = ["Config", "DEFAULT_CONFIG_FILE", "LOW_RAM_THRESHOLD_BYTES", "RuntimeConfigurationError"]

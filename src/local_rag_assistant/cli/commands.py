"""Click-powered CLI adapters over the orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from local_rag_assistant.api.main import create_app
from local_rag_assistant.bootstrap import build_bootstrap
from local_rag_assistant.cli.tui import (
    render_chat_response,
    render_diagnostics,
    render_document_detail,
    render_document_list,
    render_search_result,
    render_status,
)
from local_rag_assistant.config import Config


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--config-file", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--chroma-db-path", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--embedding-model-path", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--llm-model-path", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--default-top-k", type=click.IntRange(1, 50), default=None)
@click.version_option(package_name="local-rag-assistant")
@click.pass_context
def cli(
    ctx: click.Context,
    config_file: Path | None,
    chroma_db_path: Path | None,
    embedding_model_path: Path | None,
    llm_model_path: Path | None,
    default_top_k: int | None,
) -> None:
    """Local-first RAG tooling for indexing and querying your vault."""
    ctx.ensure_object(dict)
    ctx.obj["config_overrides"] = {
        "config_file": config_file,
        "chroma_db_path": chroma_db_path,
        "embedding_model_path": embedding_model_path,
        "llm_model_path": llm_model_path,
        "default_top_k": default_top_k,
    }


@cli.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--recursive/--no-recursive", default=True)
@click.option("--extension", "extensions", multiple=True)
@click.pass_context
def index(ctx: click.Context, path: Path, recursive: bool, extensions: tuple[str, ...]) -> None:
    """Index supported documents from a directory."""
    response = _run(
        lambda: _bootstrap(ctx).build_orchestrator(require_embedding=True, require_llm=False).index(
            str(path),
            recursive=recursive,
            extensions=list(extensions) or None,
        )
    )
    _print_payload(response)


@cli.command()
@click.argument("query")
@click.option("--top-k", type=click.IntRange(1, 50), default=None)
@click.pass_context
def search(ctx: click.Context, query: str, top_k: int | None) -> None:
    """Run semantic search against the local index."""
    response = _run(lambda: _bootstrap(ctx).build_orchestrator(require_embedding=True, require_llm=False).search(query, top_k=top_k))
    _print_payload(response)


@cli.command()
@click.argument("question")
@click.option("--top-k", type=click.IntRange(1, 50), default=None)
@click.option("--system-prompt", default=None)
@click.pass_context
def chat(ctx: click.Context, question: str, top_k: int | None, system_prompt: str | None) -> None:
    """Answer a question using indexed context and the local LLM."""
    response = _run(
        lambda: _bootstrap(ctx).build_orchestrator(require_embedding=True, require_llm=True).chat(
            question,
            top_k=top_k,
            system_prompt=system_prompt,
        )
    )
    _print_payload(response)


@cli.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show high-level index statistics."""
    _print_payload(_run(lambda: _bootstrap(ctx).build_orchestrator(require_embedding=False, require_llm=False).stats()))


@cli.group()
def tui() -> None:
    """Operator-focused terminal views over the local runtime."""


@tui.command("status")
@click.pass_context
def tui_status(ctx: click.Context) -> None:
    """Show a compact runtime status summary."""
    runtime = _bootstrap(ctx)
    stats_payload = _run(lambda: runtime.build_orchestrator(require_embedding=False, require_llm=False).stats())
    documents = _run(lambda: runtime.registry().list())
    click.echo(render_status(stats_payload, documents, runtime.config))


@tui.command("documents")
@click.option("--limit", type=click.IntRange(1, 200), default=20, show_default=True)
@click.pass_context
def tui_documents(ctx: click.Context, limit: int) -> None:
    """List indexed document records."""
    runtime = _bootstrap(ctx)
    documents = sorted(_run(lambda: runtime.registry().list()), key=lambda document: document.updated_at, reverse=True)
    click.echo(render_document_list(documents[:limit], total_documents=len(documents)))


@tui.command("inspect")
@click.argument("document_id")
@click.option("--chunk-limit", type=click.IntRange(1, 20), default=3, show_default=True)
@click.pass_context
def tui_inspect(ctx: click.Context, document_id: str, chunk_limit: int) -> None:
    """Inspect a single document record and stored chunks."""
    runtime = _bootstrap(ctx)
    record = _run(lambda: runtime.registry().get(document_id))
    if record is None:
        raise click.ClickException(f"Document {document_id} not found")
    chunks = _run(lambda: runtime.vectorstore().get_document_chunks(document_id))
    click.echo(render_document_detail(record, chunks, chunk_limit=chunk_limit))


@tui.command("search")
@click.argument("query")
@click.option("--top-k", type=click.IntRange(1, 50), default=None)
@click.pass_context
def tui_search(ctx: click.Context, query: str, top_k: int | None) -> None:
    """Run semantic search with text output."""
    response = _run(lambda: _bootstrap(ctx).build_orchestrator(require_embedding=True, require_llm=False).search(query, top_k=top_k))
    click.echo(render_search_result(response))


@tui.command("chat")
@click.argument("question")
@click.option("--top-k", type=click.IntRange(1, 50), default=None)
@click.option("--system-prompt", default=None)
@click.pass_context
def tui_chat(ctx: click.Context, question: str, top_k: int | None, system_prompt: str | None) -> None:
    """Run grounded chat with text output."""
    response = _run(
        lambda: _bootstrap(ctx).build_orchestrator(require_embedding=True, require_llm=True).chat(
            question,
            top_k=top_k,
            system_prompt=system_prompt,
        )
    )
    click.echo(render_chat_response(response))


@tui.command("diagnostics")
@click.pass_context
def tui_diagnostics(ctx: click.Context) -> None:
    """Show basic runtime visibility and path diagnostics."""
    runtime = _bootstrap(ctx)
    stats_payload = _run(lambda: runtime.build_orchestrator(require_embedding=False, require_llm=False).stats())
    documents = _run(lambda: runtime.registry().list())
    click.echo(render_diagnostics(runtime.config, stats_payload, documents))


@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, type=click.IntRange(1, 65535), show_default=True)
@click.option("--reload/--no-reload", default=False, show_default=True)
@click.pass_context
def serve(ctx: click.Context, host: str, port: int, reload: bool) -> None:
    """Serve the FastAPI adapter for local integrations."""
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise click.ClickException("uvicorn is required to run the API server.") from exc

    app = create_app(config=_config(ctx))
    uvicorn.run(app, host=host, port=port, reload=reload)


def main(argv: list[str] | None = None) -> int:
    try:
        cli.main(args=argv, prog_name="local-rag-assistant", standalone_mode=False)
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except click.exceptions.Exit as exc:
        return exc.exit_code
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1
    return 0


def _config(ctx: click.Context) -> Config:
    state = ctx.ensure_object(dict)
    config = state.get("config")
    if config is None:
        config = Config.from_cli(**state["config_overrides"])
        state["config"] = config
    return config


def _bootstrap(ctx: click.Context):
    return build_bootstrap(_config(ctx))


def _run(func: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return func(*args, **kwargs)
    except (FileNotFoundError, NotADirectoryError, PermissionError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


def _print_payload(payload: Any) -> None:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


__all__ = ["cli", "main"]

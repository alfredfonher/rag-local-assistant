# local-rag-assistant

Local-first RAG assistant for document ingestion, indexing, semantic retrieval, and grounded chat over local files — built with Python, FastAPI, ChromaDB, and llama.cpp GGUF models.

## What it does today

- Ingests `.txt`, `.md`, and `.pdf` files into a local document registry.
- Indexes extracted text into a ChromaDB vector store.
- Runs semantic search over indexed content.
- Answers questions with grounded local context and source excerpts.
- Works with local GGUF models for both embeddings and generation.
- Exposes a minimal management UI for ingest, documents, search, and chat.
- Exposes an operator-focused TUI for status, inspection, search, chat, and diagnostics.

## Product surfaces

| Surface | Purpose |
|---|---|
| UI | Main manager for document ingestion, document lifecycle, search, and chat |
| TUI | Lightweight operator surface for status, inspection, text queries, and diagnostics |
| API | Programmatic ingestion and document-management endpoints |
| CLI | Local command surface for indexing, search, chat, and operator tooling |

## What comes next

- Better per-file ingest feedback in the UI for mixed success/failure batches.
- Broader text-document support beyond the current MVP formats.
- Optional OCR or explicit handling for scan-only PDFs.
- Presentation polish such as screenshots/demo assets.

## Tech stack

- Python 3.12
- FastAPI
- ChromaDB
- llama.cpp via `llama-cpp-python`
- GGUF models
- `uv` for reproducible local setup

## Quick start

Use the repo-rooted setup flow:

```bash
./scripts/setup-local-env.sh
cp .env.example .env
uv run local-rag-assistant --help
```

The setup script does this:

```bash
uv python install 3.12
uv venv --python 3.12 --clear
uv sync --locked --python 3.12 --extra dev --extra local-runtime
uv run python -m local_rag_assistant --help
```

That creates `.venv/` in the repo and installs the editable package, development dependencies, and local runtime dependencies.

## Docker

This repo now includes a minimal Docker setup for the FastAPI web app.

What it preserves from the local-first setup:

- The app still expects repo-relative `./models` and `./data` paths.
- Compose mounts `./models` into `/app/models` as read-only.
- Compose mounts `./data` into `/app/data` as writable.
- The container serves the web app with `uvicorn local_rag_assistant.api.main:app`.

Run it from the repo root:

```bash
cp .env.example .env
docker compose up --build
```

The app is published only on the loopback interface:

```text
http://127.0.0.1:90
```

Useful commands:

```bash
docker compose up --build -d
docker compose logs -f
curl http://127.0.0.1:90/health
docker compose down
```

Notes:

- `docker compose` reads values from `.env` automatically if you copy and customize it.
- `/health` works without loading local models, but indexing, search, and chat still require the GGUF files mounted from `./models`.
- The first image build may be slow because `llama-cpp-python` can require native build steps depending on wheel availability for your machine.
- This setup is intentionally single-container and local-only; it is not trying to hide or replace the repo's local model/runtime assumptions.

## Environment variables

Copy `.env.example` to `.env`.

```bash
cp .env.example .env
```

- `LOCAL_RAG_ASSISTANT_EMBEDDING_MODEL_PATH` defaults to `./models/nomic-embed-text-v1.5.f32.gguf`
- `LOCAL_RAG_ASSISTANT_LLM_MODEL_PATH` defaults to `./models/gemma-3-1b-it-Q4_K_M.gguf`
- `LOCAL_RAG_ASSISTANT_CHROMA_DB_PATH` defaults to `./data/chroma_db`

## Verified local model pairing

- Embeddings: `models/nomic-embed-text-v1.5.f32.gguf`
- Generation: `models/gemma-3-1b-it-Q4_K_M.gguf`

## End-to-end smoke validation

Run an honest local smoke check from the repo root with a fresh Chroma path:

```bash
uv run local-rag-assistant \
  --embedding-model-path models/nomic-embed-text-v1.5.f32.gguf \
  --llm-model-path models/gemma-3-1b-it-Q4_K_M.gguf \
  --chroma-db-path /tmp/local-rag-assistant-smoke \
  index tests/fixtures/smoke

uv run local-rag-assistant \
  --embedding-model-path models/nomic-embed-text-v1.5.f32.gguf \
  --llm-model-path models/gemma-3-1b-it-Q4_K_M.gguf \
  --chroma-db-path /tmp/local-rag-assistant-smoke \
  search "volcanoes basalt"

uv run local-rag-assistant \
  --embedding-model-path models/nomic-embed-text-v1.5.f32.gguf \
  --llm-model-path models/gemma-3-1b-it-Q4_K_M.gguf \
  --chroma-db-path /tmp/local-rag-assistant-smoke \
  chat "What does the alpha document mention?"
```

Each Chroma directory keeps its own `change-tracker.json`, so a fresh `--chroma-db-path` starts with a clean indexing state.

## Architecture at a glance

1. **Ingestion** extracts text from supported document formats.
2. **Registry** persists document records and statuses in SQLite.
3. **Indexer** chunks content, computes embeddings, and writes to ChromaDB.
4. **Orchestrator** powers semantic search and grounded chat.
5. **UI / TUI / API / CLI** expose the same core runtime through different operator surfaces.

## Focused verification

After setup, these checks should pass from the repo root:

```bash
uv run local-rag-assistant --help
uv run python -c "from local_rag_assistant.api.main import app; print(app.title)"
uv run pytest tests/unit/test_bootstrap.py tests/unit/test_cli_commands.py tests/unit/test_api_main.py -q
```

## Project structure

- `src/local_rag_assistant/` — core runtime, CLI, API, indexing, retrieval, and chat orchestration
- `models/` — local GGUF models
- `tests/` — unit tests and smoke fixtures
- `scripts/` — setup helpers

## Known limitations

- Local models must already exist on disk.
- Current verified workflow is local-first and single-user.
- Scan-only PDFs still need OCR or a stronger failure/reporting policy.
- `/search` currently multiplexes HTML and JSON behavior based on the request type; a cleaner `/api/...` split would be better long-term.

# local-rag-assistant

A local-first RAG assistant for document ingestion, semantic indexing, and grounded chat — runs entirely on your machine with GGUF models, no cloud APIs.

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.12 |
| API | FastAPI + Uvicorn |
| Vector Store | ChromaDB |
| LLM Runtime | llama-cpp-python (GGUF models) |
| Package Manager | uv |
| Testing | pytest |

## Quick Start

```bash
./scripts/setup-local-env.sh
cp .env.example .env
uv run local-rag-assistant --help
```

The setup script installs Python 3.12, creates a virtualenv, and installs all dependencies (including dev and local-runtime extras).

### Docker

```bash
cp .env.example .env
docker compose up --build
```

The container serves the web app at `http://127.0.0.1:90`. Models are mounted read-only from `./models`, data is writable at `./data`.

## Architecture

```
┌──────────────────────────────────────┐
│  Surfaces: UI | TUI | API | CLI     │
│  (same core runtime, different UIs)  │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│          Orchestrator                │
│  Semantic search + grounded chat     │
└──────────────┬───────────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌─────────────┐  ┌──────────────┐
│  Indexer    │  │   Registry   │
│ Chunks,     │  │  SQLite doc  │
│ embeds,     │  │  records     │
│ writes DB   │  │              │
└──────┬──────┘  └──────────────┘
       │
       ▼
┌─────────────┐
│  ChromaDB   │
└─────────────┘
```

1. **Ingestion** extracts text from .txt, .md, and .pdf files
2. **Registry** persists document records in SQLite
3. **Indexer** chunks content, computes embeddings, writes to ChromaDB
4. **Orchestrator** powers search and chat with local GGUF models
5. **Four surfaces** expose the same runtime through different interfaces

## Project Structure

```
rag-local-assistant/
├── src/local_rag_assistant/   # Core runtime, CLI, API, indexing, retrieval
├── models/                    # Local GGUF models
├── tests/                     # Unit tests and smoke fixtures
├── scripts/                   # Setup helpers
└── docker-compose.yml
```

## Verified Model Pairing

- Embeddings: `nomic-embed-text-v1.5.f32.gguf`
- Generation: `gemma-3-1b-it-Q4_K_M.gguf`

Place models in `./models` before running. The app will not start without them.

## Testing

```bash
# Unit tests
uv run pytest tests/unit/ -q

# Smoke validation (fresh Chroma path)
uv run local-rag-assistant \
  --embedding-model-path models/nomic-embed-text-v1.5.f32.gguf \
  --llm-model-path models/gemma-3-1b-it-Q4_K_M.gguf \
  --chroma-db-path /tmp/smoke \
  index tests/fixtures/smoke

uv run local-rag-assistant \
  --chroma-db-path /tmp/smoke \
  search "query here"
```

## License

MIT

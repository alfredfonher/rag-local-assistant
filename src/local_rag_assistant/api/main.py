"""Minimal FastAPI application for local_rag_assistant."""

from __future__ import annotations

import json
import threading
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import RedirectResponse, StreamingResponse

from local_rag_assistant.api.schemas import (
    DeleteDocumentResponse,
    DocumentListResponse,
    IngestRequest,
    IngestResponse,
    IngestionJobResponse,
)
from local_rag_assistant.bootstrap import Bootstrap, build_bootstrap
from local_rag_assistant.config import Config
from local_rag_assistant.ingestion.jobs import IngestionJobNotFoundError
from local_rag_assistant.ingestion.models import IngestionDocumentRecord
from local_rag_assistant.models import (
    ChatRequest,
    ChatResponse,
    IndexRequest,
    IndexResponse,
    IndexStats,
    SearchResult,
)
from local_rag_assistant.ui import install_ui
from local_rag_assistant.ui.routes import render_search_page


def create_app(*, config: Config | None = None, bootstrap: Bootstrap | None = None) -> FastAPI:
    runtime = bootstrap or build_bootstrap(config or Config())
    app = FastAPI(title="local-rag-assistant", version="1.0.0")
    app.state.bootstrap = runtime
    install_ui(app)

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/overview", status_code=307)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "collection_name": runtime.config.chroma_collection_name,
        }

    @app.get("/stats", response_model=IndexStats)
    def stats() -> IndexStats:
        return _run(lambda: runtime.build_orchestrator(require_embedding=False, require_llm=False).stats())

    @app.post("/index", response_model=IndexResponse)
    def index_documents(request: IndexRequest) -> IndexResponse:
        return _run(
            lambda: runtime.build_orchestrator(require_embedding=True, require_llm=False).index(
                request.path,
                recursive=request.recursive,
                extensions=request.extensions,
            )
        )

    @app.get("/search", response_model=SearchResult)
    def search(request: Request, query: str | None = Query(None), top_k: int = Query(5, ge=1, le=50)) -> SearchResult | HTMLResponse:
        if _wants_html(request):
            return render_search_page(request, query=query)
        if query is None:
            raise HTTPException(status_code=422, detail="query is required")
        return _run(lambda: runtime.build_orchestrator(require_embedding=True, require_llm=False).search(query, top_k=top_k))

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        return _run(
            lambda: runtime.build_orchestrator(require_embedding=True, require_llm=True).chat(
                request.query,
                top_k=request.top_k,
                system_prompt=request.system_prompt,
            )
        )

    @app.post("/api/v1/ingest", response_model=IngestResponse)
    def ingest(request: IngestRequest) -> IngestResponse:
        return IngestResponse(documents=_run(lambda: runtime.ingestion_service().ingest_paths(request.paths)))

    @app.post("/api/v1/ingestion-jobs", response_model=IngestionJobResponse, status_code=202)
    def create_ingestion_job(request: IngestRequest) -> IngestionJobResponse:
        store = runtime.ingestion_job_store()
        job_id = store.create_job(request.paths)
        worker = threading.Thread(
            target=_run_ingestion_job,
            args=(runtime, store, job_id, request.paths),
            name=f"ingestion-job-{job_id[:8]}",
            daemon=True,
        )
        worker.start()
        return IngestionJobResponse(**store.get_state(job_id).model_dump())

    @app.get("/api/v1/ingestion-jobs/{job_id}", response_model=IngestionJobResponse)
    def get_ingestion_job(job_id: str) -> IngestionJobResponse:
        return IngestionJobResponse(**_run(lambda: runtime.ingestion_job_store().get_state(job_id)).model_dump())

    @app.get("/api/v1/ingestion-jobs/{job_id}/events")
    def stream_ingestion_job_events(
        job_id: str,
        last_event_id: int | None = Query(None, ge=0),
        last_event_id_header: str | None = Header(None, alias="Last-Event-ID"),
    ) -> StreamingResponse:
        store = runtime.ingestion_job_store()
        _run(lambda: store.get_state(job_id))
        resume_after = _resume_sequence(last_event_id, last_event_id_header)
        return StreamingResponse(
            _stream_ingestion_job_events(store, job_id, last_event_id=resume_after),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/v1/documents", response_model=DocumentListResponse)
    def list_documents() -> DocumentListResponse:
        return DocumentListResponse(documents=_run(lambda: runtime.registry().list()))

    @app.get("/api/v1/documents/{document_id}", response_model=IngestionDocumentRecord)
    def get_document(document_id: str) -> IngestionDocumentRecord:
        return _run(lambda: _require_document(runtime.registry(), document_id))

    @app.delete("/api/v1/documents/{document_id}", response_model=DeleteDocumentResponse)
    def delete_document(document_id: str) -> DeleteDocumentResponse:
        _run(lambda: _require_document(runtime.registry(), document_id))
        _run(lambda: runtime.vectorstore().delete_by_document_id(document_id))
        deleted = _run(lambda: runtime.registry().delete(document_id))
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        return DeleteDocumentResponse(document_id=document_id)

    @app.post("/api/v1/documents/{document_id}/reindex", response_model=IngestionDocumentRecord)
    def reindex_document(document_id: str) -> IngestionDocumentRecord:
        record = _run(lambda: _require_document(runtime.registry(), document_id))
        return _run(lambda: runtime.ingestion_service().ingest_path(record.source_path))

    return app


def _wants_html(request: Request) -> bool:
    return "text/html" in request.headers.get("accept", "")


def _run(func: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return func(*args, **kwargs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (NotADirectoryError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except IngestionJobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _require_document(registry: Any, document_id: str) -> Any:
    record = registry.get(document_id)
    if record is None:
        raise FileNotFoundError(f"Document {document_id} not found")
    return record


def _run_ingestion_job(runtime: Bootstrap, store: Any, job_id: str, paths: list[str]) -> None:
    try:
        records = runtime.ingestion_service(progress_sink=store.sink_for(job_id)).ingest_paths(paths)
        store.set_results(job_id, records)
    except Exception as exc:
        store.mark_failed(job_id, str(exc))


def _stream_ingestion_job_events(store: Any, job_id: str, *, last_event_id: int) -> Any:
    yield "retry: 1000\n\n"
    for event in store.stream_events(job_id, after_sequence=last_event_id):
        payload = {
            "job_id": event.job_id,
            "sequence": event.sequence,
            **event.progress.model_dump(mode="json"),
        }
        yield (
            f"id: {event.sequence}\n"
            f"event: {event.progress.event}\n"
            f"data: {json.dumps(payload)}\n\n"
        )


def _resume_sequence(last_event_id: int | None, last_event_id_header: str | None) -> int:
    if last_event_id is not None:
        return last_event_id
    if last_event_id_header is None:
        return 0
    try:
        return max(int(last_event_id_header), 0)
    except ValueError:
        return 0


app = create_app()


__all__ = ["app", "create_app"]

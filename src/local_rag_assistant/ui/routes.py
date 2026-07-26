"""Minimal server-rendered UI routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from fastapi import APIRouter, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from local_rag_assistant.bootstrap import Bootstrap
from local_rag_assistant.ingestion.models import IngestionDocumentRecord

UI_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(UI_DIR / "templates"))
router = APIRouter(include_in_schema=False)


@router.get("/overview", response_class=HTMLResponse)
def overview(request: Request) -> HTMLResponse:
    runtime = _runtime(request)
    stats = _run(lambda: runtime.build_orchestrator(require_embedding=False, require_llm=False).stats())
    documents = _run(lambda: runtime.registry().list())
    recent_documents = sorted(documents, key=lambda document: document.updated_at, reverse=True)[:5]
    return templates.TemplateResponse(
        request,
        "overview.html",
        {
            "page": "overview",
            "stats": stats,
            "document_count": len(documents),
            "status_counts": _status_counts(documents),
            "recent_documents": recent_documents,
        },
    )


@router.get("/ingest", response_class=HTMLResponse, name="ui-ingest-page")
def ingest_page(request: Request) -> HTMLResponse:
    runtime = _runtime(request)
    documents = _run(lambda: runtime.registry().list())
    return templates.TemplateResponse(
        request,
        "ingest.html",
        _page_context(
            request,
            page="ingest",
            ingest_endpoint=request.url_for("ingest"),
            ingest_submit_endpoint=request.url_for("ui-ingest-submit"),
            ingestion_job_endpoint=request.url_for("create_ingestion_job"),
            ingestion_jobs_base_url=str(request.url_for("get_ingestion_job", job_id="__job_id__")).replace("__job_id__", ""),
            document_count=len(documents),
            registry_path=runtime.config.ingestion_registry_path,
            storage_path=runtime.config.ingestion_storage_path,
        ),
    )


@router.get("/chat", response_class=HTMLResponse, name="ui-chat-page")
def chat_page(request: Request, query: str | None = None) -> HTMLResponse:
    return render_chat_page(request, query=query)


@router.post("/ingest", name="ui-ingest-submit", response_model=None)
def ingest_submit(request: Request, paths: str = Form(...)) -> Response:
    runtime = _runtime(request)
    parsed_paths = [value.strip() for value in paths.splitlines() if value.strip()]
    if not parsed_paths:
        documents = _run(lambda: runtime.registry().list())
        return templates.TemplateResponse(
            request,
            "ingest.html",
            _page_context(
                request,
                page="ingest",
                ingest_endpoint=request.url_for("ingest"),
                ingest_submit_endpoint=request.url_for("ui-ingest-submit"),
                ingestion_job_endpoint=request.url_for("create_ingestion_job"),
                ingestion_jobs_base_url=str(request.url_for("get_ingestion_job", job_id="__job_id__")).replace("__job_id__", ""),
                document_count=len(documents),
                registry_path=runtime.config.ingestion_registry_path,
                storage_path=runtime.config.ingestion_storage_path,
                error="Enter at least one absolute or resolvable path.",
                submitted_paths=paths,
            ),
            status_code=400,
        )
    try:
        records = _run(lambda: runtime.ingestion_service().ingest_paths(parsed_paths))
    except HTTPException as exc:
        documents = _run(lambda: runtime.registry().list())
        return templates.TemplateResponse(
            request,
            "ingest.html",
            _page_context(
                request,
                page="ingest",
                ingest_endpoint=request.url_for("ingest"),
                ingest_submit_endpoint=request.url_for("ui-ingest-submit"),
                ingestion_job_endpoint=request.url_for("create_ingestion_job"),
                ingestion_jobs_base_url=str(request.url_for("get_ingestion_job", job_id="__job_id__")).replace("__job_id__", ""),
                document_count=len(documents),
                registry_path=runtime.config.ingestion_registry_path,
                storage_path=runtime.config.ingestion_storage_path,
                error=str(exc.detail),
                submitted_paths=paths,
            ),
            status_code=exc.status_code,
        )
    indexed = sum(record.status == "indexed" for record in records)
    failed = len(records) - indexed
    return _redirect(
        request,
        "documents",
        message=f"Processed {len(records)} path(s): {indexed} indexed, {failed} failed.",
    )


@router.get("/documents", response_class=HTMLResponse, name="documents")
def documents(request: Request) -> HTMLResponse:
    runtime = _runtime(request)
    records = sorted(_run(lambda: runtime.registry().list()), key=lambda document: document.updated_at, reverse=True)
    return templates.TemplateResponse(
        request,
        "documents.html",
        _page_context(request, page="documents", documents=records),
    )


@router.get("/documents/{document_id}", response_class=HTMLResponse, name="document-detail")
def document_detail(request: Request, document_id: str) -> HTMLResponse:
    runtime = _runtime(request)
    record = runtime.registry().get(document_id)
    if record is None:
        return templates.TemplateResponse(
            request,
            "not_found.html",
            _page_context(request, page="documents", resource_name="Document", resource_id=document_id),
            status_code=404,
        )
    return templates.TemplateResponse(
        request,
        "document_detail.html",
        _page_context(request, page="documents", document=record),
    )


@router.post("/documents/{document_id}/reindex", name="ui-document-reindex", response_model=None)
def document_reindex(request: Request, document_id: str) -> Response:
    runtime = _runtime(request)
    record = runtime.registry().get(document_id)
    if record is None:
        return templates.TemplateResponse(
            request,
            "not_found.html",
            _page_context(request, page="documents", resource_name="Document", resource_id=document_id),
            status_code=404,
        )
    try:
        updated = _run(lambda: runtime.ingestion_service().ingest_path(record.source_path))
    except HTTPException as exc:
        return _redirect(
            request,
            "document-detail",
            document_id=document_id,
            error=str(exc.detail),
        )
    return _redirect(
        request,
        "document-detail",
        document_id=document_id,
        message="Document reindexed." if updated.status == "indexed" else "Document reindex completed with failures.",
        error=updated.error_message if updated.status == "failed" else None,
    )


@router.post("/documents/{document_id}/delete", name="ui-document-delete", response_model=None)
def document_delete(request: Request, document_id: str) -> Response:
    runtime = _runtime(request)
    record = runtime.registry().get(document_id)
    if record is None:
        return templates.TemplateResponse(
            request,
            "not_found.html",
            _page_context(request, page="documents", resource_name="Document", resource_id=document_id),
            status_code=404,
        )
    _run(lambda: runtime.vectorstore().delete_by_document_id(document_id))
    deleted = _run(lambda: runtime.registry().delete(document_id))
    if not deleted:
        return templates.TemplateResponse(
            request,
            "not_found.html",
            _page_context(request, page="documents", resource_name="Document", resource_id=document_id),
            status_code=404,
        )
    return _redirect(
        request,
        "documents",
        message=f"Deleted {record.title or document_id}.",
    )


def install_ui(app: FastAPI) -> None:
    if getattr(app.state, "ui_installed", False):
        return
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=str(UI_DIR / "static")), name="ui-static")
    app.state.ui_installed = True


def _runtime(request: Request) -> Bootstrap:
    return cast(Bootstrap, request.app.state.bootstrap)


def render_search_page(request: Request, *, query: str | None = None) -> HTMLResponse:
    runtime = _runtime(request)
    if query is None:
        return templates.TemplateResponse(
            request,
            "search.html",
            _page_context(request, page="search", submitted_query="", search_result=None, has_result=False),
        )

    clean_query = query.strip()
    if not clean_query:
        return templates.TemplateResponse(
            request,
            "search.html",
            _page_context(
                request,
                page="search",
                submitted_query=query,
                search_result=None,
                has_result=False,
                error="Enter a search query.",
            ),
            status_code=400,
        )

    try:
        result = _run(lambda: runtime.build_orchestrator(require_embedding=True, require_llm=False).search(clean_query))
        return templates.TemplateResponse(
            request,
            "search.html",
            _page_context(
                request,
                page="search",
                submitted_query=clean_query,
                search_result=result,
                has_result=True,
            ),
        )
    except HTTPException as exc:
        return _ui_error_response(
            request,
            "search.html",
            page="search",
            submitted_query=clean_query,
            search_result=None,
            has_result=False,
            error=str(exc.detail),
            status_code=exc.status_code,
        )


def render_chat_page(request: Request, *, query: str | None = None) -> HTMLResponse:
    runtime = _runtime(request)
    if query is None:
        return templates.TemplateResponse(
            request,
            "chat.html",
            _page_context(request, page="chat", submitted_query="", chat_response=None, has_result=False),
        )

    clean_query = query.strip()
    if not clean_query:
        return templates.TemplateResponse(
            request,
            "chat.html",
            _page_context(
                request,
                page="chat",
                submitted_query=query,
                chat_response=None,
                has_result=False,
                error="Enter a chat prompt.",
            ),
            status_code=400,
        )

    try:
        result = _run(lambda: runtime.build_orchestrator(require_embedding=True, require_llm=True).chat(clean_query))
        return templates.TemplateResponse(
            request,
            "chat.html",
            _page_context(
                request,
                page="chat",
                submitted_query=clean_query,
                chat_response=result,
                has_result=True,
            ),
        )
    except HTTPException as exc:
        return _ui_error_response(
            request,
            "chat.html",
            page="chat",
            submitted_query=clean_query,
            chat_response=None,
            has_result=False,
            error=str(exc.detail),
            status_code=exc.status_code,
        )


def _page_context(request: Request, **context: Any) -> dict[str, Any]:
    return {
        **context,
        "message": request.query_params.get("message"),
        "error": context.get("error") or request.query_params.get("error"),
    }


def _ui_error_response(
    request: Request,
    template_name: str,
    *,
    status_code: int,
    **context: Any,
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        template_name,
        _page_context(request, **context),
        status_code=status_code,
    )


def _redirect(request: Request, route_name: str, **params: Any) -> RedirectResponse:
    message = params.pop("message", None)
    error = params.pop("error", None)
    url = request.url_for(route_name, **params)
    if message is not None:
        url = url.include_query_params(message=message)
    if error is not None:
        url = url.include_query_params(error=error)
    return RedirectResponse(url=str(url), status_code=303)


def _status_counts(documents: list[IngestionDocumentRecord]) -> dict[str, int]:
    counts = {"pending": 0, "indexed": 0, "failed": 0}
    for document in documents:
        counts[document.status] += 1
    return counts


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


__all__ = ["install_ui", "render_chat_page", "render_search_page", "router"]

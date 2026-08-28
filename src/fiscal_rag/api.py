"""FastAPI surface for the internal-network Fiscal RAG service."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from ipaddress import ip_address
from time import monotonic, perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from fiscal_rag.service import FiscalRAGService, ServiceAnswer, build_fiscal_rag_service
from fiscal_rag.settings import PROJECT_ROOT, ServiceSettings, configure_service_logging
from fiscal_rag.version import __version__
from fiscal_rag.usage import (
    ACTIVE_PROFILE_ID,
    SQLiteUsageRepository,
    UsageFeedbackNotAllowed,
    UsageRepository,
    UsageRequestNotFound,
    TRAFFIC_KINDS,
)


MAX_QUESTION_CHARACTERS = 2000
FRONTEND_DIRECTORY = PROJECT_ROOT / "frontend"
FRONTEND_ASSETS_DIRECTORY = FRONTEND_DIRECTORY / "assets"
FRONTEND_INDEX_FILE = FRONTEND_DIRECTORY / "index.html"
LOGGER = logging.getLogger("fiscal_rag.api")
ServiceFactory = Callable[[ServiceSettings], FiscalRAGService]
UsageFactory = Callable[[ServiceSettings], UsageRepository]
USAGE_PRUNE_INTERVAL_SECONDS = 24 * 60 * 60
USAGE_PRUNE_RETRY_SECONDS = 5 * 60


class AskRequest(BaseModel):
    """One user question; retrieval parameters remain server-controlled."""

    question: str = Field(max_length=MAX_QUESTION_CHARACTERS)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must not be empty")
        if any(0xD800 <= ord(character) <= 0xDFFF for character in normalized):
            raise ValueError("question contains invalid Unicode characters")
        return normalized


class SourceResponse(BaseModel):
    rank: int
    source: str | None
    section: str | None
    subsection: str | None
    dense_score: float | None
    rerank_score: float


class AskResponse(BaseModel):
    request_id: str
    answer: str
    retrieval_query: str
    rewrite_query: str | None
    query_rewrite_status: str | None
    required_constraints: list[str]
    missing_constraints: list[str]
    sources: list[SourceResponse]


class HealthResponse(BaseModel):
    status: str


class FeedbackRequest(BaseModel):
    rating: str

    @field_validator("rating")
    @classmethod
    def validate_rating(cls, value: str) -> str:
        if value not in {"positive", "negative"}:
            raise ValueError("rating must be positive or negative")
        return value


class FeedbackResponse(BaseModel):
    request_id: str
    rating: str


def create_app(
    *,
    settings: ServiceSettings | None = None,
    service_factory: ServiceFactory = build_fiscal_rag_service,
    usage_factory: UsageFactory = lambda active: SQLiteUsageRepository(
        active.usage_db_path
    ),
) -> FastAPI:
    """Create an app whose pipeline is loaded once during process startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active_settings = settings or ServiceSettings.from_environment()
        configure_service_logging(active_settings.log_level)
        app.state.ready = False
        try:
            app.state.usage_repository = usage_factory(active_settings)
            app.state.usage_repository.initialize()
            interrupted = app.state.usage_repository.mark_started_interrupted()
            deleted = app.state.usage_repository.prune_expired(
                active_settings.usage_retention_days
            )
            app.state.usage_retention_days = active_settings.usage_retention_days
            app.state.next_usage_prune = monotonic() + USAGE_PRUNE_INTERVAL_SECONDS
            app.state.usage_prune_lock = asyncio.Lock()
            app.state.rag_service = service_factory(active_settings)
        except Exception as error:
            LOGGER.critical(
                "service_start_failed error_type=%s", type(error).__name__
            )
            raise
        if interrupted or deleted:
            LOGGER.info(
                "usage_maintenance interrupted=%s expired_deleted=%s",
                interrupted,
                deleted,
            )
        app.state.ready = True
        LOGGER.info("service_ready")
        try:
            yield
        finally:
            app.state.ready = False
            LOGGER.info("service_stopped")

    app = FastAPI(
        title="Fiscal Assistant Internal API",
        version=__version__,
        description=(
            "Internal-network access to the lightweight assistant workflow "
            "and verified Fiscal RAG pipeline."
        ),
        lifespan=lifespan,
    )
    app.state.ready = False
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_ASSETS_DIRECTORY),
        name="frontend-assets",
    )

    @app.middleware("http")
    async def request_observability(request: Request, call_next: Callable):
        request_id = uuid4().hex
        request.state.request_id = request_id
        started = perf_counter()
        try:
            response = await call_next(request)
        except Exception as error:
            LOGGER.error(
                "request_failed request_id=%s error_type=%s",
                request_id,
                type(error).__name__,
            )
            response = _error_response(
                request_id,
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "internal_error",
                "The request could not be completed.",
            )
        response.headers["X-Request-ID"] = request_id
        duration_ms = (perf_counter() - started) * 1000
        LOGGER.info(
            "request_completed request_id=%s method=%s path=%s "
            "status=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, _error: RequestValidationError
    ) -> JSONResponse:
        return _error_response(
            _request_id(request),
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "validation_error",
            "Request validation failed.",
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(
        request: Request, error: HTTPException
    ) -> JSONResponse:
        code = {
            status.HTTP_502_BAD_GATEWAY: "model_request_failed",
            status.HTTP_503_SERVICE_UNAVAILABLE: "service_not_ready",
        }.get(error.status_code, "request_failed")
        message = error.detail if isinstance(error.detail, str) else "Request failed."
        response = _error_response(
            _request_id(request), error.status_code, code, message
        )
        if error.headers:
            response.headers.update(error.headers)
        return response

    @app.get("/health/live", response_model=HealthResponse, tags=["health"])
    async def live() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/", include_in_schema=False)
    async def frontend() -> FileResponse:
        return FileResponse(FRONTEND_INDEX_FILE, media_type="text/html")

    @app.get("/health/ready", response_model=HealthResponse, tags=["health"])
    async def ready(request: Request) -> HealthResponse:
        if not request.app.state.ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The service is not ready.",
            )
        return HealthResponse(status="ready")

    @app.post(
        "/v1/ask",
        response_model=AskResponse,
        tags=["rag"],
    )
    async def ask(request: Request, payload: AskRequest) -> AskResponse:
        request_id = _request_id(request)
        if not request.app.state.ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The service is not ready.",
            )
        started = perf_counter()
        usage_created = await _create_usage_request(
            request,
            endpoint="ask",
            question=payload.question,
            route="rag",
        )
        try:
            result = await run_in_threadpool(
                request.app.state.rag_service.answer, payload.question
            )
        except Exception as error:
            if usage_created:
                await run_in_threadpool(
                    _finalize_usage_request,
                    request,
                    execution_status="failed",
                    route="rag",
                    answer="",
                    error_code="model_request_failed",
                    error_type=type(error).__name__,
                    failure_stage="rag",
                    total_duration_ms=(perf_counter() - started) * 1000,
                )
            LOGGER.warning(
                "model_request_failed request_id=%s error_type=%s",
                request_id,
                type(error).__name__,
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="The model request failed.",
            ) from error
        if usage_created:
            await run_in_threadpool(
                _finalize_usage_request,
                request,
                execution_status="completed",
                route="rag",
                answer=result.answer,
                trace=_trace_from_service_answer(result),
                total_duration_ms=(perf_counter() - started) * 1000,
            )
        return _ask_response(request_id, result)

    @app.post(
        "/v1/assistant/stream",
        tags=["assistant"],
        response_class=StreamingResponse,
        responses={
            200: {
                "description": "Server-sent assistant events.",
                "content": {"text/event-stream": {}},
            }
        },
    )
    async def assistant_stream(
        request: Request, payload: AskRequest
    ) -> StreamingResponse:
        request_id = _request_id(request)
        if not request.app.state.ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The service is not ready.",
            )
        usage_created = await _create_usage_request(
            request,
            endpoint="assistant_stream",
            question=payload.question,
        )

        return StreamingResponse(
            _assistant_event_stream(
                request,
                payload.question,
                request_id=request_id,
                usage_created=usage_created,
            ),
            media_type="text/event-stream; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.put(
        "/v1/assistant/feedback/{request_id}",
        response_model=FeedbackResponse,
        tags=["assistant"],
    )
    async def set_feedback(
        request: Request, request_id: str, payload: FeedbackRequest
    ) -> FeedbackResponse | JSONResponse:
        try:
            await run_in_threadpool(
                request.app.state.usage_repository.set_feedback,
                request_id,
                payload.rating,
            )
        except UsageRequestNotFound:
            return _error_response(
                _request_id(request),
                status.HTTP_404_NOT_FOUND,
                "request_not_found",
                "The assistant request was not found.",
            )
        except UsageFeedbackNotAllowed:
            return _error_response(
                _request_id(request),
                status.HTTP_409_CONFLICT,
                "feedback_not_allowed",
                "Feedback is only accepted for completed production assistant requests.",
            )
        except Exception as error:
            LOGGER.error(
                "feedback_persist_failed request_id=%s error_type=%s",
                request_id,
                type(error).__name__,
            )
            return _error_response(
                _request_id(request),
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "usage_store_unavailable",
                "Feedback could not be saved.",
            )
        return FeedbackResponse(request_id=request_id, rating=payload.rating)

    @app.delete(
        "/v1/assistant/feedback/{request_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        tags=["assistant"],
    )
    async def delete_feedback(request: Request, request_id: str) -> Response:
        try:
            await run_in_threadpool(
                request.app.state.usage_repository.delete_feedback, request_id
            )
        except UsageRequestNotFound:
            return _error_response(
                _request_id(request),
                status.HTTP_404_NOT_FOUND,
                "request_not_found",
                "The assistant request was not found.",
            )
        except UsageFeedbackNotAllowed:
            return _error_response(
                _request_id(request),
                status.HTTP_409_CONFLICT,
                "feedback_not_allowed",
                "Feedback is only accepted for completed production assistant requests.",
            )
        except Exception as error:
            LOGGER.error(
                "feedback_delete_failed request_id=%s error_type=%s",
                request_id,
                type(error).__name__,
            )
            return _error_response(
                _request_id(request),
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "usage_store_unavailable",
                "Feedback could not be removed.",
            )
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return app


def _assistant_event_stream(
    request: Request,
    question: str,
    *,
    request_id: str,
    usage_created: bool,
):
    route: str | None = None
    terminal_status = "aborted"
    started = perf_counter()
    answer_parts: list[str] = []
    trace: Mapping[str, object] | None = None
    timings_ms: Mapping[str, object] | None = None
    error_code: str | None = "client_disconnected"
    error_type: str | None = None
    failure_stage: str | None = "client"
    try:
        yield _sse_event("start", {"request_id": request_id})
        for event in request.app.state.rag_service.stream_answer(question):
            data = dict(event.data)
            if event.event == "route":
                route_value = data.get("route")
                route = route_value if isinstance(route_value, str) else None
            elif event.event == "trace":
                trace = data
            elif event.event == "delta":
                text = data.get("text")
                if isinstance(text, str):
                    answer_parts.append(text)
            if event.event == "done":
                data["request_id"] = request_id
                terminal_status = "completed"
                error_code = None
                failure_stage = None
                raw_timings = data.get("timings_ms")
                timings_ms = raw_timings if isinstance(raw_timings, Mapping) else None
            yield _sse_event(event.event, data)
        if terminal_status != "completed":
            terminal_status = "failed"
            error_code = "stream_incomplete"
            failure_stage = _stream_failure_stage(route, trace, answer_parts)
    except GeneratorExit:
        raise
    except Exception as error:
        terminal_status = "failed"
        error_code = "model_request_failed"
        error_type = type(error).__name__
        failure_stage = _stream_failure_stage(route, trace, answer_parts)
        LOGGER.warning(
            "assistant_stream_failed request_id=%s route=%s error_type=%s",
            request_id,
            route or "unknown",
            type(error).__name__,
        )
        yield _sse_event(
            "error",
            {
                "code": "model_request_failed",
                "message": "The model request failed.",
                "request_id": request_id,
            },
        )
    finally:
        duration_ms = (perf_counter() - started) * 1000
        if usage_created:
            _finalize_usage_request(
                request,
                execution_status=terminal_status,
                route=route,
                answer="".join(answer_parts),
                trace=trace,
                timings_ms=timings_ms,
                error_code=error_code,
                error_type=error_type,
                failure_stage=failure_stage,
                total_duration_ms=duration_ms,
            )
        LOGGER.info(
            "assistant_stream_completed request_id=%s route=%s "
            "status=%s duration_ms=%.2f",
            request_id,
            route or "unknown",
            terminal_status,
            duration_ms,
        )


async def _create_usage_request(
    request: Request,
    *,
    endpoint: str,
    question: str,
    route: str | None = None,
) -> bool:
    await _maybe_prune_usage(request)
    request_id = _request_id(request)
    try:
        await run_in_threadpool(
            request.app.state.usage_repository.create_request,
            request_id,
            endpoint=endpoint,
            question=question,
            service_version=__version__,
            profile_id=ACTIVE_PROFILE_ID,
            traffic_kind=_traffic_kind(request),
            route=route,
        )
    except Exception as error:
        LOGGER.error(
            "usage_create_failed request_id=%s error_type=%s",
            request_id,
            type(error).__name__,
        )
        return False
    return True


def _finalize_usage_request(
    request: Request,
    *,
    execution_status: str,
    route: str | None,
    answer: str,
    trace: Mapping[str, object] | None = None,
    timings_ms: Mapping[str, object] | None = None,
    error_code: str | None = None,
    error_type: str | None = None,
    failure_stage: str | None = None,
    total_duration_ms: float | None = None,
) -> None:
    request_id = _request_id(request)
    try:
        request.app.state.usage_repository.finalize_request(
            request_id,
            execution_status=execution_status,
            route=route,
            answer=answer,
            trace=trace,
            timings_ms=timings_ms,
            error_code=error_code,
            error_type=error_type,
            failure_stage=failure_stage,
            total_duration_ms=total_duration_ms,
        )
    except Exception as error:
        LOGGER.error(
            "usage_finalize_failed request_id=%s error_type=%s",
            request_id,
            type(error).__name__,
        )


async def _maybe_prune_usage(request: Request) -> None:
    now = monotonic()
    if now < getattr(request.app.state, "next_usage_prune", now):
        return
    async with request.app.state.usage_prune_lock:
        now = monotonic()
        if now < request.app.state.next_usage_prune:
            return
        request.app.state.next_usage_prune = now + USAGE_PRUNE_RETRY_SECONDS
        try:
            deleted = await run_in_threadpool(
                request.app.state.usage_repository.prune_expired,
                request.app.state.usage_retention_days,
            )
        except Exception as error:
            LOGGER.error("usage_prune_failed error_type=%s", type(error).__name__)
            return
        request.app.state.next_usage_prune = now + USAGE_PRUNE_INTERVAL_SECONDS
    if deleted:
        LOGGER.info("usage_expired_deleted count=%s", deleted)


def _trace_from_service_answer(result: ServiceAnswer) -> dict[str, object]:
    return {
        "retrieval_query": result.retrieval_query,
        "rewrite_query": result.rewrite_query,
        "query_rewrite_status": result.query_rewrite_status,
        "required_constraints": list(result.required_constraints),
        "missing_constraints": list(result.missing_constraints),
        "sources": [
            {
                "rank": source.rank,
                "source": source.source,
                "section": source.section,
                "subsection": source.subsection,
                "dense_score": source.dense_score,
                "rerank_score": source.rerank_score,
            }
            for source in result.sources
        ],
    }


def _stream_failure_stage(
    route: str | None,
    trace: Mapping[str, object] | None,
    answer_parts: list[str],
) -> str:
    if route is None:
        return "router_or_preparation"
    if route == "rag" and trace is None:
        return "rag_preparation"
    if answer_parts:
        return "generation_partial"
    return "generation"


def _traffic_kind(request: Request) -> str:
    value = request.headers.get("X-Fiscal-RAG-Traffic-Kind", "production")
    if value not in TRAFFIC_KINDS or value == "production":
        return "production"
    client = request.client
    if client is None:
        return "production"
    try:
        trusted = ip_address(client.host).is_loopback
    except ValueError:
        trusted = False
    if not trusted:
        LOGGER.warning("untrusted_usage_traffic_kind_ignored")
        return "production"
    return value


def _ask_response(request_id: str, result: ServiceAnswer) -> AskResponse:
    return AskResponse(
        request_id=request_id,
        answer=result.answer,
        retrieval_query=result.retrieval_query,
        rewrite_query=result.rewrite_query,
        query_rewrite_status=result.query_rewrite_status,
        required_constraints=list(result.required_constraints),
        missing_constraints=list(result.missing_constraints),
        sources=[
            SourceResponse(
                rank=source.rank,
                source=source.source,
                section=source.section,
                subsection=source.subsection,
                dense_score=source.dense_score,
                rerank_score=source.rerank_score,
            )
            for source in result.sources
        ],
    )


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", uuid4().hex)


def _error_response(
    request_id: str,
    status_code: int,
    code: str,
    message: str,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
            }
        },
    )


def _sse_event(event: str, data: dict[str, object]) -> str:
    serialized = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {serialized}\n\n"


app = create_app()

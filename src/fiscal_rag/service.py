"""Thread-safe service adapter around the verified persistent RAG pipeline."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from threading import Lock
from time import perf_counter
from typing import Protocol

from fiscal_rag.assistant import AssistantWorkflow, PreparedAssistantResponse
from fiscal_rag.pipeline import (
    RAGResult,
    RetrievedResult,
    build_persistent_dense_rerank_rag_pipeline,
)
from fiscal_rag.retrieval.dense_rerank import (
    DENSE_SCORE_METADATA_KEY,
    RERANK_SCORE_METADATA_KEY,
)
from fiscal_rag.settings import ServiceSettings
from fiscal_rag.timing import RequestTimingRecorder, use_timing_recorder


FINAL_CONTEXT_K = 5


class AnswerPipeline(Protocol):
    """The pipeline operation used by the HTTP service."""

    def answer(self, question: str, *, k: int = 5) -> RAGResult: ...


class StreamingAssistantWorkflow(Protocol):
    def prepare(
        self, question: str, *, k: int = 5
    ) -> PreparedAssistantResponse: ...

    def stream_prepared(
        self, prepared: PreparedAssistantResponse
    ) -> Iterator[str]: ...


@dataclass(frozen=True)
class SourceReference:
    """Citation metadata safe to expose without returning private chunk text."""

    rank: int
    source: str | None
    section: str | None
    subsection: str | None
    dense_score: float | None
    rerank_score: float


@dataclass(frozen=True)
class ServiceAnswer:
    """Sanitized answer returned by the service layer."""

    answer: str
    retrieval_query: str
    rewrite_query: str | None
    query_rewrite_status: str | None
    required_constraints: tuple[str, ...]
    missing_constraints: tuple[str, ...]
    sources: tuple[SourceReference, ...]


@dataclass(frozen=True)
class ServiceStreamEvent:
    """One transport-neutral event from the assistant workflow."""

    event: str
    data: Mapping[str, object]


class FiscalRAGService:
    """Serialize access to the current stateful retrieval trace implementation."""

    def __init__(
        self,
        pipeline: AnswerPipeline,
        *,
        assistant_workflow: StreamingAssistantWorkflow | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._assistant_workflow = assistant_workflow
        self._answer_lock = Lock()

    def answer(self, question: str) -> ServiceAnswer:
        """Run the fixed Top-5 service profile and remove private context text."""
        with self._answer_lock:
            result = self._pipeline.answer(question, k=FINAL_CONTEXT_K)
        return service_answer_from_rag_result(result)

    def stream_answer(self, question: str) -> Iterator[ServiceStreamEvent]:
        """Serialize one routed assistant turn through its final streamed chunk."""
        if self._assistant_workflow is None:
            raise RuntimeError("Assistant workflow is not configured.")
        request_started = perf_counter()
        lock_wait_started = perf_counter()
        recorder = RequestTimingRecorder()
        with self._answer_lock:
            recorder.set("queue_wait", (perf_counter() - lock_wait_started) * 1000)
            with use_timing_recorder(recorder):
                prepared = self._assistant_workflow.prepare(
                    question, k=FINAL_CONTEXT_K
                )
            yield ServiceStreamEvent("route", {"route": prepared.route})
            if prepared.rag_request is not None:
                yield ServiceStreamEvent(
                    "trace", service_trace_from_prepared(prepared)
                )

            generation_started = perf_counter()
            first_delta_recorded = False
            try:
                for text in self._assistant_workflow.stream_prepared(prepared):
                    if text and not first_delta_recorded:
                        now = perf_counter()
                        recorder.set(
                            "generation_ttft", (now - generation_started) * 1000
                        )
                        recorder.set("server_ttft", (now - request_started) * 1000)
                        first_delta_recorded = True
                    yield ServiceStreamEvent("delta", {"text": text})
            finally:
                recorder.set(
                    "generation", (perf_counter() - generation_started) * 1000
                )
            recorder.set("server_total", (perf_counter() - request_started) * 1000)
            yield ServiceStreamEvent(
                "done",
                {
                    "route": prepared.route,
                    "timings_ms": dict(recorder.snapshot()),
                },
            )


def build_fiscal_rag_service(settings: ServiceSettings) -> FiscalRAGService:
    """Open the validated existing index and assemble one reusable pipeline."""
    pipeline = build_persistent_dense_rerank_rag_pipeline(
        settings.corpus_dir,
        settings.index_dir,
    )
    return FiscalRAGService(
        pipeline,
        assistant_workflow=AssistantWorkflow(pipeline),
    )


def service_answer_from_rag_result(result: RAGResult) -> ServiceAnswer:
    """Convert a pipeline result without exposing context or page_content."""
    sources = _source_references(result.retrieved_results)

    return ServiceAnswer(
        answer=result.answer,
        retrieval_query=result.retrieval_query,
        rewrite_query=result.rewrite_query,
        query_rewrite_status=result.query_rewrite_status,
        required_constraints=result.required_constraints,
        missing_constraints=result.missing_constraints,
        sources=sources,
    )


def service_trace_from_prepared(
    prepared: PreparedAssistantResponse,
) -> dict[str, object]:
    """Expose only the existing safe retrieval trace for a prepared RAG route."""
    rag_request = prepared.rag_request
    if rag_request is None:
        raise ValueError("Only a RAG route has retrieval trace metadata.")
    sources = _source_references(rag_request.retrieved_results)
    return {
        "retrieval_query": rag_request.retrieval_query,
        "rewrite_query": rag_request.rewrite_query,
        "query_rewrite_status": rag_request.query_rewrite_status,
        "required_constraints": list(rag_request.required_constraints),
        "missing_constraints": list(rag_request.missing_constraints),
        "sources": [
            {
                "rank": source.rank,
                "source": source.source,
                "section": source.section,
                "subsection": source.subsection,
                "dense_score": source.dense_score,
                "rerank_score": source.rerank_score,
            }
            for source in sources
        ],
    }


def _source_references(
    retrieved_results: list[RetrievedResult],
) -> tuple[SourceReference, ...]:
    sources: list[SourceReference] = []
    for rank, (document, score) in enumerate(retrieved_results, start=1):
        metadata = document.metadata
        rerank_score = _optional_float(metadata.get(RERANK_SCORE_METADATA_KEY))
        sources.append(
            SourceReference(
                rank=rank,
                source=_optional_string(metadata.get("source")),
                section=_optional_string(metadata.get("section")),
                subsection=_optional_string(metadata.get("subsection")),
                dense_score=_optional_float(metadata.get(DENSE_SCORE_METADATA_KEY)),
                rerank_score=rerank_score if rerank_score is not None else float(score),
            )
        )
    return tuple(sources)


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)

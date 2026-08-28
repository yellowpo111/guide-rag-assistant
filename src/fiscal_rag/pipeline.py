"""Minimal retrieve-then-generate RAG pipeline for local guide experiments."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_deepseek import ChatDeepSeek

from fiscal_rag.embeddings import QwenEmbeddings
from fiscal_rag.ingestion import ingest_markdown_directory
from fiscal_rag.retrieval.dense import GlobalDenseRetriever, SimilaritySearchStore
from fiscal_rag.vector_store import (
    build_in_memory_vector_store,
    open_persistent_chroma_vector_store,
)


RetrievedResult = tuple[Document, float]
# Current verified default: Dense Top-20 recovers observed candidate-recall
# failures while preserving Reranker Top-5 on the validated holdouts.
DENSE_RERANK_CANDIDATE_K = 20
DEFAULT_DEEPSEEK_TIMEOUT_SECONDS = 60.0
DEFAULT_DEEPSEEK_MAX_RETRIES = 0
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"


class Retriever(Protocol):
    """The minimal retrieval interface required by the generation pipeline."""

    def retrieve(self, query: str, *, k: int = 5) -> list[RetrievedResult]: ...


class ChatModel(Protocol):
    """The minimal chat model interface required by the generation pipeline."""

    def invoke(self, input: str) -> Any: ...

    def stream(self, input: str) -> Iterator[Any]: ...


@dataclass(frozen=True)
class RAGResult:
    """Observable outputs from one retrieve-then-generate request."""

    question: str
    retrieved_results: list[RetrievedResult]
    context: str
    answer: str
    retrieval_query: str
    rewrite_query: str | None = None
    query_rewrite_status: str | None = None
    required_constraints: tuple[str, ...] = ()
    missing_constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreparedRAGRequest:
    """Retrieved context and prompt ready for one generation call."""

    question: str
    retrieved_results: list[RetrievedResult]
    context: str
    prompt: str
    retrieval_query: str
    rewrite_query: str | None = None
    query_rewrite_status: str | None = None
    required_constraints: tuple[str, ...] = ()
    missing_constraints: tuple[str, ...] = ()


class BasicRAGPipeline:
    """Retrieve global top-k chunks, construct context, and ask a chat model."""

    def __init__(self, retriever: Retriever, chat_model: ChatModel) -> None:
        self._retriever = retriever
        self._chat_model = chat_model

    def answer(self, question: str, *, k: int = 5) -> RAGResult:
        """Answer a question using only the globally retrieved top-k chunks."""
        prepared = self.prepare(question, k=k)
        response = self._chat_model.invoke(prepared.prompt)
        return rag_result_from_prepared(prepared, _response_content(response))

    def prepare(self, question: str, *, k: int = 5) -> PreparedRAGRequest:
        """Retrieve context and build a prompt without starting generation."""
        retrieved_results = self._retriever.retrieve(question, k=k)
        retrieval_trace = _retrieval_trace_for(self._retriever, question)
        context = build_context(retrieved_results)
        prompt = build_rag_prompt(question, context)

        return PreparedRAGRequest(
            question=question,
            retrieved_results=retrieved_results,
            context=context,
            prompt=prompt,
            retrieval_query=retrieval_trace["retrieval_query"],
            rewrite_query=retrieval_trace["rewrite_query"],
            query_rewrite_status=retrieval_trace["query_rewrite_status"],
            required_constraints=retrieval_trace["required_constraints"],
            missing_constraints=retrieval_trace["missing_constraints"],
        )

    def stream_prepared(self, prepared: PreparedRAGRequest) -> Iterator[str]:
        """Yield final-generation text while preserving the prepared retrieval trace."""
        yield from _stream_response_content(self._chat_model.stream(prepared.prompt))


def rag_result_from_prepared(
    prepared: PreparedRAGRequest, answer: str
) -> RAGResult:
    """Combine a prepared request with a completed generated answer."""
    return RAGResult(
        question=prepared.question,
        retrieved_results=prepared.retrieved_results,
        context=prepared.context,
        answer=answer,
        retrieval_query=prepared.retrieval_query,
        rewrite_query=prepared.rewrite_query,
        query_rewrite_status=prepared.query_rewrite_status,
        required_constraints=prepared.required_constraints,
        missing_constraints=prepared.missing_constraints,
    )


def build_basic_rag_pipeline(
    data_dir: str | Path,
    *,
    embeddings: Embeddings | None = None,
    chat_model: ChatModel | None = None,
) -> BasicRAGPipeline:
    """Build the V0 pipeline from local Markdown through global dense retrieval."""
    _, chunks = ingest_markdown_directory(data_dir)
    vector_store = build_in_memory_vector_store(chunks, embeddings or QwenEmbeddings())
    retriever = GlobalDenseRetriever(vector_store)
    return BasicRAGPipeline(retriever, chat_model or create_deepseek_chat_model())


def build_dense_rerank_rag_pipeline(
    data_dir: str | Path,
    *,
    embeddings: Embeddings | None = None,
    chat_model: ChatModel | None = None,
    reranker: Any | None = None,
    query_rewriter: Any | None = None,
    candidate_k: int = DENSE_RERANK_CANDIDATE_K,
) -> BasicRAGPipeline:
    """Build the opt-in live Rewrite + Guard + Dense-Rerank RAG preview pipeline.

    New preview questions use the live conservative rewriter, followed by the
    same deterministic guard. Evaluation callers may inject a frozen rewriter
    and candidate count to reproduce a controlled historical comparison.
    """
    active_embeddings = embeddings or QwenEmbeddings()
    _, chunks = ingest_markdown_directory(data_dir)
    vector_store = build_in_memory_vector_store(chunks, active_embeddings)
    return build_dense_rerank_rag_pipeline_from_vector_store(
        vector_store,
        chat_model=chat_model,
        reranker=reranker,
        query_rewriter=query_rewriter,
        candidate_k=candidate_k,
    )


def build_persistent_dense_rerank_rag_pipeline(
    corpus_dir: str | Path,
    index_dir: str | Path,
    *,
    embeddings: Embeddings | None = None,
    chat_model: ChatModel | None = None,
    reranker: Any | None = None,
    query_rewriter: Any | None = None,
    candidate_k: int = DENSE_RERANK_CANDIDATE_K,
) -> BasicRAGPipeline:
    """Build the final RAG pipeline from a validated persistent local index."""
    active_embeddings = embeddings or QwenEmbeddings()
    vector_store = open_persistent_chroma_vector_store(
        corpus_dir,
        index_dir,
        active_embeddings,
        allow_unpublished_corpus_changes=True,
    )
    return build_dense_rerank_rag_pipeline_from_vector_store(
        vector_store,
        chat_model=chat_model,
        reranker=reranker,
        query_rewriter=query_rewriter,
        candidate_k=candidate_k,
    )


def build_dense_rerank_rag_pipeline_from_vector_store(
    vector_store: SimilaritySearchStore,
    *,
    chat_model: ChatModel | None = None,
    reranker: Any | None = None,
    query_rewriter: Any | None = None,
    candidate_k: int = DENSE_RERANK_CANDIDATE_K,
) -> BasicRAGPipeline:
    """Assemble Rewrite, Guard, Dense-Rerank, and Generation from a store."""
    from fiscal_rag.query_rewrite import (
        ConstraintPreservationGuard,
        DeepSeekQueryRewriter,
        QueryRewriteRetriever,
    )
    from fiscal_rag.reranker import DashScopeReranker
    from fiscal_rag.retrieval.dense_rerank import DenseRerankRetriever

    dense_retriever = GlobalDenseRetriever(vector_store)
    dense_rerank_retriever = DenseRerankRetriever(
        dense_retriever,
        reranker or DashScopeReranker(),
        candidate_k=candidate_k,
    )
    retriever = QueryRewriteRetriever(
        dense_rerank_retriever,
        query_rewriter or DeepSeekQueryRewriter(),
        guard=ConstraintPreservationGuard(),
    )
    return BasicRAGPipeline(retriever, chat_model or create_deepseek_chat_model())


def build_context(retrieved_results: list[RetrievedResult]) -> str:
    """Format retrieved documents into clearly separated, source-labelled blocks."""
    blocks = []
    for document, _score in retrieved_results:
        metadata = document.metadata
        blocks.append(
            "\n".join(
                [
                    f"[Source: {metadata.get('source', 'Unknown')}]",
                    f"[Section: {metadata.get('section', 'Unknown')}]",
                    f"[Subsection: {metadata.get('subsection', 'Unknown')}]",
                    "",
                    document.page_content,
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def build_rag_prompt(question: str, context: str) -> str:
    """Create the deliberately simple V0 grounded-answering prompt."""
    return f"""仅根据提供的 Context 回答 Question。

如果 Context 中没有足够的信息支持答案，应明确说明无法根据当前资料确定。
不要自行补充 Context 中不存在的具体操作步骤。

Context:
{context}

Question:
{question}

Answer:
"""


def create_deepseek_chat_model() -> ChatDeepSeek:
    """Create the configured DeepSeek chat model without exposing credentials."""
    _load_project_env()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing required DeepSeek configuration: DEEPSEEK_API_KEY. "
            "Set it in .env or as an environment variable."
        )

    model = os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
    base_url = os.getenv("DEEPSEEK_BASE_URL")
    return ChatDeepSeek(
        model=model,
        api_key=api_key,
        base_url=base_url or None,
        temperature=0,
        timeout=_positive_float_setting(
            "DEEPSEEK_TIMEOUT_SECONDS", DEFAULT_DEEPSEEK_TIMEOUT_SECONDS
        ),
        max_retries=_non_negative_integer_setting(
            "DEEPSEEK_MAX_RETRIES", DEFAULT_DEEPSEEK_MAX_RETRIES
        ),
    )


def _load_project_env() -> None:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env", override=False)


def _response_content(response: Any) -> str:
    content = getattr(response, "content", response)
    answer = content if isinstance(content, str) else str(content)
    if not answer.strip():
        raise ValueError("DeepSeek returned an empty response.")
    return answer


def _stream_response_content(responses: Iterator[Any]) -> Iterator[str]:
    pieces: list[str] = []
    for response in responses:
        content = getattr(response, "content", response)
        text = content if isinstance(content, str) else str(content)
        if not text:
            continue
        pieces.append(text)
        yield text
    if not "".join(pieces).strip():
        raise ValueError("DeepSeek returned an empty streaming response.")


def _positive_float_setting(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive number.") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive number.")
    return parsed


def _non_negative_integer_setting(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a non-negative integer.") from error
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return parsed


def _retrieval_trace_for(retriever: Retriever, question: str) -> dict[str, object]:
    """Return optional rewrite observability without coupling the pipeline to one retriever."""
    decision_for = getattr(retriever, "rewrite_decision_for", None)
    if not callable(decision_for):
        return {
            "retrieval_query": question,
            "rewrite_query": None,
            "query_rewrite_status": None,
            "required_constraints": (),
            "missing_constraints": (),
        }

    decision = decision_for(question)
    return {
        "retrieval_query": decision.retrieval_query,
        "rewrite_query": decision.rewrite_query,
        "query_rewrite_status": decision.status,
        "required_constraints": decision.required_constraints,
        "missing_constraints": decision.missing_constraints,
    }

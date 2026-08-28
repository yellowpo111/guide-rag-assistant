"""Dense candidate retrieval followed by an independent reranking stage."""

from __future__ import annotations

from typing import Protocol

from langchain_core.documents import Document

from fiscal_rag.reranker import RerankResult
from fiscal_rag.timing import measure_stage


DENSE_SCORE_METADATA_KEY = "_dense_score"
RERANK_SCORE_METADATA_KEY = "_rerank_score"
PAGE_CONTENT_DOCUMENT_PROFILE = "page_content"
METADATA_CONTEXT_DOCUMENT_PROFILE = "metadata_context"
RERANK_DOCUMENT_PROFILES = (
    PAGE_CONTENT_DOCUMENT_PROFILE,
    METADATA_CONTEXT_DOCUMENT_PROFILE,
)


class DenseRetriever(Protocol):
    """The existing global-dense retrieval contract used as a candidate source."""

    def retrieve(self, query: str, *, k: int = 5) -> list[tuple[Document, float]]: ...


class Reranker(Protocol):
    """Minimal reranking contract, kept independent from DashScope details."""

    def rerank(
        self, query: str, documents: list[str], *, top_n: int
    ) -> list[RerankResult]: ...


class DenseRerankRetriever:
    """Rerank a fixed global-dense candidate pool without changing that pool."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        reranker: Reranker,
        *,
        candidate_k: int = 20,
        document_profile: str = PAGE_CONTENT_DOCUMENT_PROFILE,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        if document_profile not in RERANK_DOCUMENT_PROFILES:
            raise ValueError(
                "document_profile must be one of: "
                + ", ".join(RERANK_DOCUMENT_PROFILES)
            )
        self._dense_retriever = dense_retriever
        self._reranker = reranker
        self._candidate_k = candidate_k
        self._document_profile = document_profile
        self._dense_candidates_by_query: dict[str, list[tuple[Document, float]]] = {}

    def retrieve(self, query: str, *, k: int = 5) -> list[tuple[Document, float]]:
        """Return reranked ``(Document, rerank_score)`` pairs for the requested top-k."""
        if k <= 0:
            raise ValueError("k must be positive")
        if k > self._candidate_k:
            raise ValueError("k cannot exceed the fixed dense candidate pool size")

        with measure_stage("dense_retrieval"):
            dense_candidates = self._dense_retriever.retrieve(query, k=self._candidate_k)
        # Retain the already-computed candidate pool for evaluation observability.
        # This does not trigger a second Dense search or alter reranking.
        self._dense_candidates_by_query[query] = list(dense_candidates)
        with measure_stage("rerank"):
            rerank_results = self._reranker.rerank(
                query,
                [
                    format_rerank_document(document, profile=self._document_profile)
                    for document, _score in dense_candidates
                ],
                top_n=k,
            )

        retrieved_results: list[tuple[Document, float]] = []
        for result in rerank_results:
            try:
                document, dense_score = dense_candidates[result.index]
            except IndexError as error:
                raise RuntimeError("Reranker returned an index outside dense candidates.") from error
            retrieved_results.append(
                (
                    _with_scores(document, dense_score, result.relevance_score),
                    result.relevance_score,
                )
            )
        return retrieved_results

    def dense_candidates_for(self, query: str) -> list[tuple[Document, float]]:
        """Return the Dense candidate pool used by the completed rerank call."""
        try:
            return list(self._dense_candidates_by_query[query])
        except KeyError as error:
            raise KeyError(
                "No Dense candidate pool was recorded for this query. "
                "Call retrieve() first."
            ) from error


def format_rerank_document(document: Document, *, profile: str) -> str:
    """Build the text seen by the reranker without mutating the retrieved chunk."""
    if profile == PAGE_CONTENT_DOCUMENT_PROFILE:
        return document.page_content
    if profile == METADATA_CONTEXT_DOCUMENT_PROFILE:
        metadata = document.metadata
        return "\n".join(
            [
                f"[Source: {_metadata_value(metadata.get('source'))}]",
                f"[Section: {_metadata_value(metadata.get('section'))}]",
                f"[Subsection: {_metadata_value(metadata.get('subsection'))}]",
                "",
                document.page_content,
            ]
        )
    raise ValueError(
        "profile must be one of: " + ", ".join(RERANK_DOCUMENT_PROFILES)
    )


def _metadata_value(value: object) -> str:
    return value if isinstance(value, str) and value else "Unknown"


def _with_scores(
    document: Document, dense_score: float, rerank_score: float
) -> Document:
    """Copy a document so observability metadata never changes indexed chunks."""
    metadata = dict(document.metadata)
    metadata[DENSE_SCORE_METADATA_KEY] = float(dense_score)
    metadata[RERANK_SCORE_METADATA_KEY] = float(rerank_score)
    return Document(page_content=document.page_content, metadata=metadata, id=document.id)

"""Dense and BM25 candidate fusion followed by the existing reranker."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from langchain_core.documents import Document

from fiscal_rag.reranker import RerankResult
from fiscal_rag.retrieval.dense_rerank import (
    DENSE_SCORE_METADATA_KEY,
    PAGE_CONTENT_DOCUMENT_PROFILE,
    RERANK_DOCUMENT_PROFILES,
    RERANK_SCORE_METADATA_KEY,
    format_rerank_document,
)


BM25_SCORE_METADATA_KEY = "_bm25_score"
RRF_SCORE_METADATA_KEY = "_rrf_score"


class RankedRetriever(Protocol):
    """Candidate source contract shared by dense and BM25 retrieval."""

    def retrieve(self, query: str, *, k: int = 5) -> list[tuple[Document, float]]: ...


class Reranker(Protocol):
    """The existing external reranker contract."""

    def rerank(
        self, query: str, documents: list[str], *, top_n: int
    ) -> list[RerankResult]: ...


@dataclass(frozen=True)
class HybridCandidate:
    """One deduplicated candidate with its candidate-source observability data."""

    document: Document
    dense_score: float | None
    bm25_score: float | None
    rrf_score: float
    best_rank: int


def reciprocal_rank_fusion(
    dense_results: Sequence[tuple[Document, float]],
    bm25_results: Sequence[tuple[Document, float]],
    *,
    rrf_k: int = 60,
) -> list[HybridCandidate]:
    """Fuse ranked dense and BM25 lists without comparing their raw scores."""
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")

    candidates: dict[str, dict[str, object]] = {}
    for source_name, results in (("dense", dense_results), ("bm25", bm25_results)):
        for rank, (document, score) in enumerate(results, start=1):
            identity = _document_identity(document)
            candidate = candidates.setdefault(
                identity,
                {
                    "document": document,
                    "dense_score": None,
                    "bm25_score": None,
                    "rrf_score": 0.0,
                    "best_rank": rank,
                },
            )
            score_key = "dense_score" if source_name == "dense" else "bm25_score"
            candidate[score_key] = float(score)
            candidate["rrf_score"] = float(candidate["rrf_score"]) + 1 / (rrf_k + rank)
            candidate["best_rank"] = min(int(candidate["best_rank"]), rank)

    fused = [
        HybridCandidate(
            document=candidate["document"],  # type: ignore[arg-type]
            dense_score=candidate["dense_score"],  # type: ignore[arg-type]
            bm25_score=candidate["bm25_score"],  # type: ignore[arg-type]
            rrf_score=float(candidate["rrf_score"]),
            best_rank=int(candidate["best_rank"]),
        )
        for candidate in candidates.values()
    ]
    return sorted(
        fused,
        key=lambda candidate: (
            -candidate.rrf_score,
            candidate.best_rank,
            _document_identity(candidate.document),
        ),
    )


class HybridRerankRetriever:
    """Rerank an RRF-fused Dense Top-N plus BM25 Top-N candidate pool."""

    def __init__(
        self,
        dense_retriever: RankedRetriever,
        bm25_retriever: RankedRetriever,
        reranker: Reranker,
        *,
        dense_candidate_k: int = 10,
        bm25_candidate_k: int = 10,
        rerank_candidate_k: int = 10,
        rrf_k: int = 60,
        document_profile: str = PAGE_CONTENT_DOCUMENT_PROFILE,
    ) -> None:
        if dense_candidate_k <= 0 or bm25_candidate_k <= 0 or rerank_candidate_k <= 0:
            raise ValueError("All candidate counts must be positive.")
        if rrf_k < 0:
            raise ValueError("rrf_k must be non-negative")
        if document_profile not in RERANK_DOCUMENT_PROFILES:
            raise ValueError(
                "document_profile must be one of: " + ", ".join(RERANK_DOCUMENT_PROFILES)
            )
        self._dense_retriever = dense_retriever
        self._bm25_retriever = bm25_retriever
        self._reranker = reranker
        self._dense_candidate_k = dense_candidate_k
        self._bm25_candidate_k = bm25_candidate_k
        self._rerank_candidate_k = rerank_candidate_k
        self._rrf_k = rrf_k
        self._document_profile = document_profile

    def retrieve(self, query: str, *, k: int = 5) -> list[tuple[Document, float]]:
        """Return RRF-selected candidates reranked by the unchanged reranker."""
        if k <= 0:
            raise ValueError("k must be positive")
        if k > self._rerank_candidate_k:
            raise ValueError("k cannot exceed the fixed fused candidate pool size")

        fused_candidates = reciprocal_rank_fusion(
            self._dense_retriever.retrieve(query, k=self._dense_candidate_k),
            self._bm25_retriever.retrieve(query, k=self._bm25_candidate_k),
            rrf_k=self._rrf_k,
        )[: self._rerank_candidate_k]
        if len(fused_candidates) < k:
            raise RuntimeError("Fused candidate pool is smaller than the requested output k.")

        rerank_results = self._reranker.rerank(
            query,
            [
                format_rerank_document(candidate.document, profile=self._document_profile)
                for candidate in fused_candidates
            ],
            top_n=k,
        )
        retrieved_results: list[tuple[Document, float]] = []
        for result in rerank_results:
            try:
                candidate = fused_candidates[result.index]
            except IndexError as error:
                raise RuntimeError("Reranker returned an index outside fused candidates.") from error
            retrieved_results.append(
                (
                    _with_hybrid_scores(candidate, result.relevance_score),
                    result.relevance_score,
                )
            )
        return retrieved_results


def _document_identity(document: Document) -> str:
    """Deduplicate only same-source copies; identical text in another guide remains distinct."""
    return f"{document.metadata.get('source', '')}\0{document.page_content}"


def _with_hybrid_scores(candidate: HybridCandidate, rerank_score: float) -> Document:
    metadata = dict(candidate.document.metadata)
    if candidate.dense_score is not None:
        metadata[DENSE_SCORE_METADATA_KEY] = candidate.dense_score
    if candidate.bm25_score is not None:
        metadata[BM25_SCORE_METADATA_KEY] = candidate.bm25_score
    metadata[RRF_SCORE_METADATA_KEY] = candidate.rrf_score
    metadata[RERANK_SCORE_METADATA_KEY] = float(rerank_score)
    return Document(
        page_content=candidate.document.page_content,
        metadata=metadata,
        id=candidate.document.id,
    )

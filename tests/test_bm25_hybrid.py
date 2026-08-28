from __future__ import annotations

from pathlib import Path
import sys

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fiscal_rag.reranker import RerankResult  # noqa: E402
from fiscal_rag.retrieval.bm25 import BM25Retriever, tokenize_chinese_text  # noqa: E402
from fiscal_rag.retrieval.hybrid_rerank import (  # noqa: E402
    HybridRerankRetriever,
    reciprocal_rank_fusion,
)


def document(name: str, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={"source": name, "section": "Section", "subsection": "Steps"},
    )


def test_chinese_tokenizer_keeps_chinese_characters_and_latin_words() -> None:
    assert tokenize_chinese_text("账号解冻 Account_ID 42") == [
        "账",
        "号",
        "解",
        "冻",
        "account_id",
        "42",
    ]


def test_bm25_prioritizes_exact_chinese_business_term_and_preserves_metadata() -> None:
    account = document("account.md", "账号解冻申请表需要加盖单位公章")
    indicator = document("indicator.md", "指标解冻需要检查可用金额")

    results = BM25Retriever([account, indicator]).retrieve("账号解冻怎么申请", k=2)

    assert [item[0].metadata["source"] for item in results] == [
        "account.md",
        "indicator.md",
    ]
    assert isinstance(results[0][1], float)
    assert results[0][0].metadata == account.metadata


def test_rrf_rewards_a_document_found_by_both_candidate_sources() -> None:
    first = document("first.md", "first")
    shared = document("shared.md", "shared")
    bm25_only = document("bm25.md", "bm25")

    fused = reciprocal_rank_fusion(
        [(first, 0.9), (shared, 0.8)],
        [(shared, 5.0), (bm25_only, 4.0)],
    )

    assert [candidate.document.metadata["source"] for candidate in fused] == [
        "shared.md",
        "first.md",
        "bm25.md",
    ]
    assert fused[0].dense_score == 0.8
    assert fused[0].bm25_score == 5.0


class StaticRetriever:
    def __init__(self, results: list[tuple[Document, float]]) -> None:
        self.results = results
        self.queries: list[str] = []

    def retrieve(self, query: str, *, k: int = 5) -> list[tuple[Document, float]]:
        self.queries.append(query)
        return self.results[:k]


class RecordingReranker:
    def __init__(self) -> None:
        self.query = ""
        self.documents: list[str] = []

    def rerank(self, query: str, documents: list[str], *, top_n: int) -> list[RerankResult]:
        self.query = query
        self.documents = documents
        return [RerankResult(index=0, relevance_score=0.91)][:top_n]


def test_hybrid_reranker_sends_rrf_candidates_to_existing_reranker_with_scores() -> None:
    dense_document = document("dense.md", "dense-only evidence")
    shared_document = document("shared.md", "shared evidence")
    bm25_document = document("bm25.md", "bm25-only evidence")
    dense = StaticRetriever([(dense_document, 0.9), (shared_document, 0.8)])
    bm25 = StaticRetriever([(shared_document, 5.0), (bm25_document, 4.0)])
    reranker = RecordingReranker()
    retriever = HybridRerankRetriever(
        dense,
        bm25,
        reranker,
        dense_candidate_k=2,
        bm25_candidate_k=2,
        rerank_candidate_k=2,
    )

    results = retriever.retrieve("shared query", k=1)

    assert dense.queries == ["shared query"]
    assert bm25.queries == ["shared query"]
    assert reranker.query == "shared query"
    assert reranker.documents[0] == "shared evidence"
    document_result, score = results[0]
    assert score == 0.91
    assert document_result.metadata["_dense_score"] == 0.8
    assert document_result.metadata["_bm25_score"] == 5.0
    assert "_rrf_score" in document_result.metadata
    assert document_result.metadata["_rerank_score"] == 0.91

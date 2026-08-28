from pathlib import Path
import sys

import pytest
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fiscal_rag.reranker import (  # noqa: E402
    DEFAULT_RERANK_INSTRUCTION,
    FISCAL_OPERATION_RERANK_INSTRUCTION,
    DashScopeReranker,
    RerankRequestError,
    RerankResult,
)
import fiscal_rag.reranker as reranker_module  # noqa: E402
from fiscal_rag.retrieval.dense_rerank import (  # noqa: E402
    DENSE_SCORE_METADATA_KEY,
    METADATA_CONTEXT_DOCUMENT_PROFILE,
    PAGE_CONTENT_DOCUMENT_PROFILE,
    RERANK_SCORE_METADATA_KEY,
    DenseRerankRetriever,
)


def test_dashscope_reranker_parses_ranked_indexes_and_builds_expected_request() -> None:
    captured: dict[str, object] = {}

    def fake_post_json(
        endpoint: str, headers: dict[str, str], payload: dict[str, object]
    ) -> object:
        captured.update(endpoint=endpoint, headers=headers, payload=payload)
        return {
            "results": [
                {"index": 1, "relevance_score": 0.91},
                {"index": 0, "relevance_score": 0.42},
            ]
        }

    reranker = DashScopeReranker(
        api_key="test-key",
        base_url="https://example.test/compatible-api/v1",
        model="test-reranker",
        post_json=fake_post_json,
    )

    results = reranker.rerank("how to save", ["intro", "save steps"], top_n=2)

    assert results == [
        RerankResult(index=1, relevance_score=0.91),
        RerankResult(index=0, relevance_score=0.42),
    ]
    assert captured["endpoint"] == "https://example.test/compatible-api/v1/reranks"
    assert captured["headers"] == {
        "Authorization": "Bearer test-key",
        "Content-Type": "application/json",
    }
    assert captured["payload"] == {
        "model": "test-reranker",
        "documents": ["intro", "save steps"],
        "query": "how to save",
        "top_n": 2,
        "instruct": DEFAULT_RERANK_INSTRUCTION,
    }


def test_dashscope_reranker_accepts_full_reranks_endpoint() -> None:
    captured_endpoint: list[str] = []

    def fake_post_json(
        endpoint: str, _headers: dict[str, str], _payload: dict[str, object]
    ) -> object:
        captured_endpoint.append(endpoint)
        return {"results": [{"index": 0, "relevance_score": 0.9}]}

    reranker = DashScopeReranker(
        api_key="test-key",
        base_url="https://example.test/compatible-api/v1/reranks",
        post_json=fake_post_json,
    )

    reranker.rerank("query", ["document"], top_n=1)

    assert captured_endpoint == ["https://example.test/compatible-api/v1/reranks"]


def test_dashscope_reranker_sends_explicit_instruction() -> None:
    captured_payload: dict[str, object] = {}

    def fake_post_json(
        _endpoint: str, _headers: dict[str, str], payload: dict[str, object]
    ) -> object:
        captured_payload.update(payload)
        return {"results": [{"index": 0, "relevance_score": 0.9}]}

    reranker = DashScopeReranker(
        api_key="test-key",
        base_url="https://example.test/compatible-api/v1",
        instruction=FISCAL_OPERATION_RERANK_INSTRUCTION,
        post_json=fake_post_json,
    )

    reranker.rerank("how to save", ["save steps"], top_n=1)

    assert captured_payload["instruct"] == FISCAL_OPERATION_RERANK_INSTRUCTION


def test_dashscope_reranker_reports_malformed_api_response() -> None:
    reranker = DashScopeReranker(
        api_key="test-key",
        base_url="https://example.test/compatible-api/v1",
        post_json=lambda *_args: {"results": [{"index": 4, "relevance_score": 0.9}]},
    )

    with pytest.raises(RuntimeError, match="invalid document index"):
        reranker.rerank("query", ["document"], top_n=1)


def test_dashscope_reranker_requires_explicit_rerank_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DASHSCOPE_RERANK_BASE_URL", raising=False)
    monkeypatch.setattr(reranker_module, "load_dotenv", lambda *args, **kwargs: False)

    with pytest.raises(ValueError, match="DASHSCOPE_RERANK_BASE_URL"):
        DashScopeReranker(api_key="test-key")


class FakeDenseRetriever:
    def __init__(self, candidates: list[tuple[Document, float]]) -> None:
        self.candidates = candidates
        self.requested_k: int | None = None

    def retrieve(self, _query: str, *, k: int = 5) -> list[tuple[Document, float]]:
        self.requested_k = k
        return self.candidates[:k]


class FakeReranker:
    def __init__(self, results: list[RerankResult]) -> None:
        self.results = results
        self.documents: list[str] | None = None
        self.top_n: int | None = None

    def rerank(
        self, _query: str, documents: list[str], *, top_n: int
    ) -> list[RerankResult]:
        self.documents = documents
        self.top_n = top_n
        return self.results


def test_dense_rerank_retriever_uses_fixed_candidates_and_preserves_scores_metadata() -> None:
    documents = [
        (
            Document(
                page_content="intro",
                metadata={"source": "guide.md", "section": "Intro"},
            ),
            0.99,
        ),
        (
            Document(
                page_content="save steps",
                metadata={
                    "source": "guide.md",
                    "section": "Operations",
                    "subsection": "Save",
                },
            ),
            0.73,
        ),
        (Document(page_content="other", metadata={"source": "other.md"}), 0.68),
    ]
    dense_retriever = FakeDenseRetriever(documents)
    reranker = FakeReranker(
        [
            RerankResult(index=1, relevance_score=0.96),
            RerankResult(index=0, relevance_score=0.64),
        ]
    )
    retriever = DenseRerankRetriever(dense_retriever, reranker, candidate_k=3)

    results = retriever.retrieve("how to save", k=2)

    assert dense_retriever.requested_k == 3
    assert reranker.documents == ["intro", "save steps", "other"]
    assert reranker.top_n == 2
    assert [score for _document, score in results] == [0.96, 0.64]
    assert [document.page_content for document, _score in results] == [
        "save steps",
        "intro",
    ]
    top_document = results[0][0]
    assert top_document.metadata == {
        "source": "guide.md",
        "section": "Operations",
        "subsection": "Save",
        DENSE_SCORE_METADATA_KEY: 0.73,
        RERANK_SCORE_METADATA_KEY: 0.96,
    }
    assert DENSE_SCORE_METADATA_KEY not in documents[1][0].metadata
    assert retriever.dense_candidates_for("how to save") == documents


def test_dense_rerank_retriever_requires_completed_query_for_candidate_trace() -> None:
    retriever = DenseRerankRetriever(
        FakeDenseRetriever([]), FakeReranker([]), candidate_k=1
    )

    import pytest

    with pytest.raises(KeyError, match="Call retrieve\\(\\) first"):
        retriever.dense_candidates_for("not yet retrieved")


def test_dense_rerank_retriever_rejects_output_k_larger_than_candidate_pool() -> None:
    retriever = DenseRerankRetriever(
        FakeDenseRetriever([]), FakeReranker([]), candidate_k=2
    )

    with pytest.raises(ValueError, match="cannot exceed"):
        retriever.retrieve("query", k=3)


def test_dense_rerank_metadata_context_profile_keeps_returned_documents_original() -> None:
    original_document = Document(
        page_content="save steps",
        metadata={
            "source": "guide.md",
            "section": "Operations",
            "subsection": "Save",
        },
    )
    missing_metadata_document = Document(
        page_content="other",
        metadata={"source": "other.md"},
    )
    dense_retriever = FakeDenseRetriever(
        [(original_document, 0.73), (missing_metadata_document, 0.68)]
    )
    reranker = FakeReranker([RerankResult(index=0, relevance_score=0.96)])
    retriever = DenseRerankRetriever(
        dense_retriever,
        reranker,
        candidate_k=2,
        document_profile=METADATA_CONTEXT_DOCUMENT_PROFILE,
    )

    results = retriever.retrieve("how to save", k=1)

    assert reranker.documents == [
        "[Source: guide.md]\n[Section: Operations]\n[Subsection: Save]\n\nsave steps",
        "[Source: other.md]\n[Section: Unknown]\n[Subsection: Unknown]\n\nother",
    ]
    assert results[0][0].page_content == "save steps"
    assert results[0][0].metadata == {
        "source": "guide.md",
        "section": "Operations",
        "subsection": "Save",
        DENSE_SCORE_METADATA_KEY: 0.73,
        RERANK_SCORE_METADATA_KEY: 0.96,
    }


def test_dense_rerank_retriever_rejects_unknown_document_profile() -> None:
    with pytest.raises(ValueError, match="document_profile"):
        DenseRerankRetriever(
            FakeDenseRetriever([]),
            FakeReranker([]),
            document_profile="unknown",
        )


def test_page_content_profile_is_the_default() -> None:
    document = Document(page_content="body", metadata={"source": "guide.md"})
    dense_retriever = FakeDenseRetriever([(document, 0.8)])
    reranker = FakeReranker([RerankResult(index=0, relevance_score=0.9)])
    retriever = DenseRerankRetriever(dense_retriever, reranker, candidate_k=1)

    retriever.retrieve("query", k=1)

    assert reranker.documents == ["body"]
    assert PAGE_CONTENT_DOCUMENT_PROFILE == "page_content"


def test_reranker_retries_only_configured_retryable_failures(monkeypatch) -> None:
    attempts = 0

    def flaky_post_json(
        _endpoint: str, _headers: dict[str, str], _payload: dict[str, object]
    ) -> object:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RerankRequestError("temporary failure", retryable=True)
        return {"results": [{"index": 0, "relevance_score": 0.9}]}

    monkeypatch.setattr(reranker_module.time, "sleep", lambda _seconds: None)
    reranker = DashScopeReranker(
        api_key="test-key",
        base_url="https://example.test/v1",
        max_retries=1,
        post_json=flaky_post_json,
    )

    assert reranker.rerank("query", ["document"], top_n=1) == [
        RerankResult(index=0, relevance_score=0.9)
    ]
    assert attempts == 2

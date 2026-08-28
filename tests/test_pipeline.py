from pathlib import Path
from types import SimpleNamespace
import sys

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fiscal_rag.pipeline as pipeline_module  # noqa: E402
from fiscal_rag.pipeline import (  # noqa: E402
    BasicRAGPipeline,
    DENSE_RERANK_CANDIDATE_K,
    build_dense_rerank_rag_pipeline,
    build_persistent_dense_rerank_rag_pipeline,
    create_deepseek_chat_model,
)
from fiscal_rag.reranker import RerankResult  # noqa: E402


class FakeRetriever:
    def __init__(self, results: list[tuple[Document, float]]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, *, k: int = 5) -> list[tuple[Document, float]]:
        self.calls.append((query, k))
        return self.results[:k]


class FakeChatModel:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        return SimpleNamespace(content="根据资料，先填写单位基础信息。")

    def stream(self, prompt: str):
        self.prompts.append(prompt)
        yield SimpleNamespace(content="根据资料，")
        yield SimpleNamespace(content="先填写单位基础信息。")


class EmptyChatModel:
    def invoke(self, _prompt: str) -> SimpleNamespace:
        return SimpleNamespace(content="   ")

    def stream(self, _prompt: str):
        yield SimpleNamespace(content="")
        yield SimpleNamespace(content="   ")


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


class FakeReranker:
    def __init__(self) -> None:
        self.documents: list[str] | None = None

    def rerank(
        self, _query: str, documents: list[str], *, top_n: int
    ) -> list[RerankResult]:
        self.documents = documents
        return [RerankResult(index=0, relevance_score=0.95) for _ in range(top_n)]


class FakeQueryRewriter:
    def rewrite(self, _question: str) -> str:
        return "建设单位如何查询政府投资项目？"


def make_results() -> list[tuple[Document, float]]:
    return [
        (
            Document(
                page_content="填写单位名称后点击保存。",
                metadata={
                    "source": "guide_a.md",
                    "section": "单位基础信息",
                    "subsection": "系统常见操作指引",
                },
            ),
            0.9,
        ),
        (
            Document(
                page_content="确认所有必填字段。",
                metadata={
                    "source": "guide_b.md",
                    "section": "数据填报",
                    "subsection": "业务操作问答",
                },
            ),
            0.8,
        ),
    ]


def test_pipeline_builds_grounded_context_and_returns_answer() -> None:
    retriever = FakeRetriever(make_results())
    chat_model = FakeChatModel()
    pipeline = BasicRAGPipeline(retriever, chat_model)

    result = pipeline.answer("单位基础信息怎么填写？", k=2)

    assert retriever.calls == [("单位基础信息怎么填写？", 2)]
    assert result.answer == "根据资料，先填写单位基础信息。"
    assert "[Source: guide_a.md]" in result.context
    assert "[Section: 单位基础信息]" in result.context
    assert "[Subsection: 系统常见操作指引]" in result.context
    assert "填写单位名称后点击保存。" in result.context
    assert "---" in result.context
    assert len(result.retrieved_results) == 2
    assert "Question:\n单位基础信息怎么填写？" in chat_model.prompts[0]
    assert result.context in chat_model.prompts[0]


def test_pipeline_passes_the_requested_k_to_retriever() -> None:
    retriever = FakeRetriever(make_results())
    pipeline = BasicRAGPipeline(retriever, FakeChatModel())

    result = pipeline.answer("测试问题", k=1)

    assert retriever.calls == [("测试问题", 1)]
    assert len(result.retrieved_results) == 1


def test_pipeline_rejects_empty_generation_response() -> None:
    pipeline = BasicRAGPipeline(FakeRetriever(make_results()), EmptyChatModel())

    import pytest

    with pytest.raises(ValueError, match="empty response"):
        pipeline.answer("测试问题", k=1)


def test_pipeline_streams_only_after_preparing_the_same_rag_trace() -> None:
    retriever = FakeRetriever(make_results())
    chat_model = FakeChatModel()
    pipeline = BasicRAGPipeline(retriever, chat_model)

    prepared = pipeline.prepare("单位基础信息怎么填写？", k=1)
    chunks = list(pipeline.stream_prepared(prepared))

    assert chunks == ["根据资料，", "先填写单位基础信息。"]
    assert prepared.retrieval_query == "单位基础信息怎么填写？"
    assert prepared.context in prepared.prompt
    assert chat_model.prompts == [prepared.prompt]


def test_pipeline_rejects_empty_streaming_generation_response() -> None:
    pipeline = BasicRAGPipeline(FakeRetriever(make_results()), EmptyChatModel())
    prepared = pipeline.prepare("测试问题", k=1)

    import pytest

    with pytest.raises(ValueError, match="empty streaming response"):
        list(pipeline.stream_prepared(prepared))


def test_create_deepseek_chat_model_applies_explicit_timeout_and_no_hidden_retry(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeConfiguredChat:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(pipeline_module, "ChatDeepSeek", FakeConfiguredChat)
    monkeypatch.setattr(pipeline_module, "_load_project_env", lambda: None)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.setenv("DEEPSEEK_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("DEEPSEEK_MAX_RETRIES", "0")

    create_deepseek_chat_model()

    assert captured["timeout"] == 45.0
    assert captured["max_retries"] == 0
    assert captured["model"] == "deepseek-v4-flash"


def test_pipeline_records_rewrite_guard_trace_when_the_retriever_provides_one() -> None:
    class TraceRetriever(FakeRetriever):
        def rewrite_decision_for(self, _question: str) -> SimpleNamespace:
            return SimpleNamespace(
                retrieval_query="建设单位怎样查询政府投资项目？",
                rewrite_query="政府投资项目如何查询？",
                status="fallback_to_original",
                required_constraints=("建设单位",),
                missing_constraints=("建设单位",),
            )

    pipeline = BasicRAGPipeline(TraceRetriever(make_results()), FakeChatModel())

    result = pipeline.answer("建设单位怎样查询政府投资项目？", k=1)

    assert result.retrieval_query == "建设单位怎样查询政府投资项目？"
    assert result.rewrite_query == "政府投资项目如何查询？"
    assert result.query_rewrite_status == "fallback_to_original"
    assert result.required_constraints == ("建设单位",)
    assert result.missing_constraints == ("建设单位",)


def test_dense_rerank_rag_builder_connects_live_rewrite_guard_to_generation(
    tmp_path: Path,
) -> None:
    for index in range(11):
        (tmp_path / f"guide_{index}.md").write_text(
            "# 项目管理\n\n## 系统操作指引\n\n建设单位可查询政府投资项目。",
            encoding="utf-8",
        )
    reranker = FakeReranker()
    pipeline = build_dense_rerank_rag_pipeline(
        tmp_path,
        embeddings=FakeEmbeddings(),
        chat_model=FakeChatModel(),
        reranker=reranker,
        query_rewriter=FakeQueryRewriter(),
    )

    result = pipeline.answer("建设单位怎样查询政府投资项目？", k=1)

    assert DENSE_RERANK_CANDIDATE_K == 20
    assert result.retrieval_query == "建设单位如何查询政府投资项目？"
    assert result.rewrite_query == "建设单位如何查询政府投资项目？"
    assert result.query_rewrite_status == "accepted"
    assert result.retrieved_results[0][0].metadata["_rerank_score"] == 0.95
    assert reranker.documents is not None
    assert len(reranker.documents) == min(11, DENSE_RERANK_CANDIDATE_K)


def test_persistent_pipeline_serves_last_successful_index(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    def fake_open(corpus_dir, index_dir, embeddings, **kwargs):
        captured.update(
            corpus_dir=corpus_dir,
            index_dir=index_dir,
            embeddings=embeddings,
            **kwargs,
        )
        return SimpleNamespace(similarity_search_with_score=lambda *_args, **_kwargs: [])

    monkeypatch.setattr(
        pipeline_module, "open_persistent_chroma_vector_store", fake_open
    )
    embeddings = FakeEmbeddings()
    build_persistent_dense_rerank_rag_pipeline(
        tmp_path / "corpus",
        tmp_path / "index",
        embeddings=embeddings,
        chat_model=FakeChatModel(),
        reranker=FakeReranker(),
        query_rewriter=FakeQueryRewriter(),
    )

    assert captured["embeddings"] is embeddings
    assert captured["allow_unpublished_corpus_changes"] is True

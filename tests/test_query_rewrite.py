from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
from langchain_core.documents import Document


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fiscal_rag.query_rewrite import (  # noqa: E402
    CONSERVATIVE_REWRITE_PROMPT,
    ConstraintPreservationGuard,
    DeepSeekQueryRewriter,
    FrozenQueryRewriter,
    QueryRewriteRetriever,
    extract_required_constraints,
)


class FakeChatModel:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.response)


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.results = [(Document(page_content="result", metadata={"source": "guide.md"}), 0.9)]

    def retrieve(self, query: str, *, k: int = 5) -> list[tuple[Document, float]]:
        self.calls.append((query, k))
        return self.results[:k]


def test_deepseek_query_rewriter_uses_conservative_prompt_and_returns_content() -> None:
    chat_model = FakeChatModel("已发布采购意向的预算金额变更方式？")
    rewriter = DeepSeekQueryRewriter(chat_model)

    rewritten_query = rewriter.rewrite("已发布的采购意向需要改预算金额怎么办？")

    assert rewritten_query == "已发布采购意向的预算金额变更方式？"
    assert "已发布的采购意向需要改预算金额怎么办？" in chat_model.prompts[0]
    assert CONSERVATIVE_REWRITE_PROMPT.splitlines()[4] in chat_model.prompts[0]


def test_deepseek_query_rewriter_rejects_empty_response() -> None:
    rewriter = DeepSeekQueryRewriter(FakeChatModel("  "))

    with pytest.raises(ValueError, match="empty retrieval query"):
        rewriter.rewrite("测试问题")


def test_query_rewrite_retriever_uses_and_records_rewritten_query() -> None:
    retriever = FakeRetriever()
    rewriter = DeepSeekQueryRewriter(FakeChatModel("清晰的检索问题"))
    wrapped_retriever = QueryRewriteRetriever(retriever, rewriter)

    results = wrapped_retriever.retrieve("原始问题", k=1)

    assert results == retriever.results
    assert retriever.calls == [("清晰的检索问题", 1)]
    assert wrapped_retriever.retrieval_query_for("原始问题") == "清晰的检索问题"


def test_constraint_guard_falls_back_when_rewrite_drops_explicit_role() -> None:
    guard = ConstraintPreservationGuard()

    decision = guard.decide("建设单位怎样查询政府投资项目？", "政府投资项目如何查询？")

    assert decision.status == "fallback_to_original"
    assert decision.retrieval_query == "建设单位怎样查询政府投资项目？"
    assert decision.required_constraints == ("建设单位",)
    assert decision.missing_constraints == ("建设单位",)


def test_constraint_guard_accepts_rewrite_that_preserves_role_and_state() -> None:
    guard = ConstraintPreservationGuard()

    decision = guard.decide(
        "建设单位已发布的采购意向送审后如何处理？",
        "建设单位已发布采购意向送审后如何处理？",
    )

    assert decision.status == "accepted"
    assert decision.missing_constraints == ()


def test_constraint_extraction_does_not_treat_business_object_as_a_role() -> None:
    constraints = extract_required_constraints("评分业务科室怎样在系统中挑选参评单位？")

    assert constraints == ("评分业务科室",)


def test_guarded_query_rewrite_retriever_records_fallback_decision() -> None:
    retriever = FakeRetriever()
    rewriter = DeepSeekQueryRewriter(FakeChatModel("政府投资项目如何查询？"))
    wrapped_retriever = QueryRewriteRetriever(
        retriever, rewriter, guard=ConstraintPreservationGuard()
    )

    wrapped_retriever.retrieve("建设单位怎样查询政府投资项目？", k=1)

    assert retriever.calls == [("建设单位怎样查询政府投资项目？", 1)]
    assert wrapped_retriever.rewrite_decision_for(
        "建设单位怎样查询政府投资项目？"
    ).status == "fallback_to_original"


def test_frozen_query_rewriter_replays_saved_rewrite() -> None:
    rewriter = FrozenQueryRewriter({"原始问题": "冻结的改写问题"})

    assert rewriter.rewrite("原始问题") == "冻结的改写问题"

    with pytest.raises(KeyError, match="no entry"):
        rewriter.rewrite("缺失问题")

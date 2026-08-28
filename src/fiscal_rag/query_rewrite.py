"""Conservative query rewriting for isolated retrieval experiments."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from langchain_core.documents import Document

from fiscal_rag.pipeline import ChatModel, create_deepseek_chat_model
from fiscal_rag.timing import measure_stage


CONSERVATIVE_DEEPSEEK_PROFILE = "conservative_deepseek"
GUARDED_CONSERVATIVE_DEEPSEEK_PROFILE = "conservative_deepseek_guarded"
CONSERVATIVE_REWRITE_PROMPT = """你是一个用于检索的查询改写器。

将下面的用户问题改写为一条简洁、清晰的中文检索问题。
必须保留原问题已经包含的业务对象、状态、范围、动作和限制。
不得回答问题；不得加入原问题没有的业务事实、正式功能或业务术语、菜单、角色、字段、步骤、条件或答案信息。
只输出改写后的问题，不要解释。

用户问题：
{question}
"""


class QueryRewriter(Protocol):
    """Convert one original user question into a retrieval query."""

    def rewrite(self, question: str) -> str: ...


class Retriever(Protocol):
    """The minimal retrieval contract used by the rewrite wrapper."""

    def retrieve(self, query: str, *, k: int = 5) -> list[tuple[Document, float]]: ...


@dataclass(frozen=True)
class QueryRewriteDecision:
    """One auditable decision about whether a rewrite can be used for retrieval."""

    rewrite_query: str
    retrieval_query: str
    status: str
    required_constraints: tuple[str, ...]
    missing_constraints: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "rewrite_query": self.rewrite_query,
            "query_rewrite_status": self.status,
            "required_constraints": list(self.required_constraints),
            "missing_constraints": list(self.missing_constraints),
        }


class ConstraintPreservationGuard:
    """Reject rewrites that drop explicit role or state constraints from a question."""

    def decide(self, original_question: str, rewrite_query: str) -> QueryRewriteDecision:
        required_constraints = extract_required_constraints(original_question)
        normalized_rewrite = _normalize_for_constraint_match(rewrite_query)
        missing_constraints = tuple(
            constraint
            for constraint in required_constraints
            if _normalize_for_constraint_match(constraint) not in normalized_rewrite
        )
        if missing_constraints:
            return QueryRewriteDecision(
                rewrite_query=rewrite_query,
                retrieval_query=original_question,
                status="fallback_to_original",
                required_constraints=required_constraints,
                missing_constraints=missing_constraints,
            )
        return QueryRewriteDecision(
            rewrite_query=rewrite_query,
            retrieval_query=rewrite_query,
            status="accepted",
            required_constraints=required_constraints,
            missing_constraints=(),
        )


class DeepSeekQueryRewriter:
    """Use the configured deterministic DeepSeek chat model for conservative rewrites."""

    def __init__(self, chat_model: ChatModel | None = None) -> None:
        self._chat_model = chat_model or create_deepseek_chat_model()

    def rewrite(self, question: str) -> str:
        response = self._chat_model.invoke(
            CONSERVATIVE_REWRITE_PROMPT.format(question=question)
        )
        rewritten_query = _response_content(response).strip()
        if not rewritten_query:
            raise ValueError(
                "Query rewriter returned an empty retrieval query for question: "
                f"{question!r}"
            )
        return rewritten_query


class FrozenQueryRewriter:
    """Replay saved rewrites so a later experiment changes only its guard."""

    def __init__(self, rewrites: dict[str, str]) -> None:
        self._rewrites = rewrites

    def rewrite(self, question: str) -> str:
        try:
            return self._rewrites[question]
        except KeyError as error:
            raise KeyError(
                "Frozen rewrite source has no entry for question: " f"{question!r}"
            ) from error


def load_frozen_query_rewrites(path: str | Path) -> dict[str, str]:
    """Load original-question to rewrite pairs from one private details JSONL file."""
    source_path = Path(path)
    rewrites: dict[str, str] = {}
    for line_number, line in enumerate(
        source_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"Frozen rewrite line {line_number} must be an object.")
        question = record.get("question")
        rewrite = record.get("retrieval_query")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"Frozen rewrite line {line_number} requires a non-empty question."
            )
        if not isinstance(rewrite, str) or not rewrite.strip():
            raise ValueError(
                f"Frozen rewrite line {line_number} requires a non-empty retrieval_query."
            )
        if question in rewrites:
            raise ValueError(f"Frozen rewrite source repeats question: {question!r}")
        rewrites[question] = rewrite
    if not rewrites:
        raise ValueError("Frozen rewrite source contains no rewrite records.")
    return rewrites


class QueryRewriteRetriever:
    """Retrieve with rewritten queries while retaining original-to-rewrite observability."""

    def __init__(
        self,
        retriever: Retriever,
        query_rewriter: QueryRewriter,
        *,
        guard: ConstraintPreservationGuard | None = None,
    ) -> None:
        self._retriever = retriever
        self._query_rewriter = query_rewriter
        self._guard = guard
        self._retrieval_queries: dict[str, str] = {}
        self._rewrite_decisions: dict[str, QueryRewriteDecision] = {}

    def retrieve(self, query: str, *, k: int = 5) -> list[tuple[Document, float]]:
        with measure_stage("rewrite"):
            rewrite_query = self._query_rewriter.rewrite(query)
        with measure_stage("guard"):
            decision = (
                self._guard.decide(query, rewrite_query)
                if self._guard is not None
                else QueryRewriteDecision(
                    rewrite_query=rewrite_query,
                    retrieval_query=rewrite_query,
                    status="unguarded",
                    required_constraints=(),
                    missing_constraints=(),
                )
            )
        self._retrieval_queries[query] = decision.retrieval_query
        self._rewrite_decisions[query] = decision
        return self._retriever.retrieve(decision.retrieval_query, k=k)

    def retrieval_query_for(self, original_question: str) -> str:
        """Return the rewrite actually used for a completed original question."""
        try:
            return self._retrieval_queries[original_question]
        except KeyError as error:
            raise KeyError(
                "No retrieval query recorded for the original question. "
                "Call retrieve() first."
            ) from error

    def rewrite_decision_for(self, original_question: str) -> QueryRewriteDecision:
        """Return the recorded raw rewrite and guard decision for one question."""
        try:
            return self._rewrite_decisions[original_question]
        except KeyError as error:
            raise KeyError(
                "No rewrite decision recorded for the original question. "
                "Call retrieve() first."
            ) from error


_ROLE_CONSTRAINT_PATTERNS = (
    re.compile(r"[\u4e00-\u9fff]{2,8}(?:经办岗|审核岗|汇总岗|专管员|管理员)"),
    re.compile(r"(?:建设|预算|采购|填报|主管|财政)单位"),
    re.compile(r"(?:主管|财政)部门"),
    re.compile(r"(?:业务处室|评分业务科室|单位会计)"),
)
_STATE_CONSTRAINT_PATTERN = re.compile(
    r"已[\u4e00-\u9fff]{1,4}(?=的)|"
    r"送审后|审核同意后|保存后|被引用后|强制审核不通过"
)


def extract_required_constraints(question: str) -> tuple[str, ...]:
    """Extract explicit role and state phrases that must survive a conservative rewrite."""
    constraints: list[str] = []
    for pattern in _ROLE_CONSTRAINT_PATTERNS:
        constraints.extend(match.group() for match in pattern.finditer(question))
    constraints.extend(match.group() for match in _STATE_CONSTRAINT_PATTERN.finditer(question))
    return tuple(dict.fromkeys(constraints))


def _normalize_for_constraint_match(text: str) -> str:
    return re.sub(r"\s+", "", text.replace("之前", "前"))


def _response_content(response: Any) -> str:
    content = getattr(response, "content", response)
    return content if isinstance(content, str) else str(content)

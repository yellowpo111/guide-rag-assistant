"""Transparent retrieval evaluation for evidence-centric JSONL cases."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document


CoverageStatus = str
RetrievedResult = tuple[Document, float]


class Retriever(Protocol):
    """The minimal interface needed to evaluate retrieval results."""

    def retrieve(self, query: str, *, k: int = 5) -> list[RetrievedResult]: ...


@dataclass(frozen=True)
class Evidence:
    """One source-labelled text span that can support an eval question."""

    source: str
    section: str | None
    subsection: str | None
    evidence_text: str


@dataclass(frozen=True)
class RetrievalEvalCase:
    """One evidence-centric retrieval evaluation case."""

    case_id: str
    question: str
    expected_answer: str | None
    relevant_evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class CaseEvaluation:
    """Metrics and diagnostics retained for one evaluated question."""

    case: RetrievalEvalCase
    coverage_status: CoverageStatus
    first_relevant_rank: int | None
    hit_at_1: int
    hit_at_3: int
    hit_at_5: int
    reciprocal_rank: float
    retrieved_results: list[RetrievedResult]

    @property
    def retrieval_miss(self) -> bool:
        """True only when coverage exists but the retrieved ranking misses it."""
        return self.coverage_status == "OK" and self.first_relevant_rank is None

    @property
    def needs_diagnostics(self) -> bool:
        return self.coverage_status == "COVERAGE_FAILURE" or self.retrieval_miss


@dataclass(frozen=True)
class RetrievalEvalSummary:
    """Aggregate retrieval metrics across every evaluated case."""

    case_evaluations: list[CaseEvaluation]

    @property
    def total_cases(self) -> int:
        return len(self.case_evaluations)

    @property
    def coverage_failures(self) -> int:
        return sum(
            evaluation.coverage_status == "COVERAGE_FAILURE"
            for evaluation in self.case_evaluations
        )

    @property
    def retrieval_misses(self) -> int:
        return sum(evaluation.retrieval_miss for evaluation in self.case_evaluations)

    @property
    def hit_at_1(self) -> float:
        return _mean(evaluation.hit_at_1 for evaluation in self.case_evaluations)

    @property
    def hit_at_3(self) -> float:
        return _mean(evaluation.hit_at_3 for evaluation in self.case_evaluations)

    @property
    def hit_at_5(self) -> float:
        return _mean(evaluation.hit_at_5 for evaluation in self.case_evaluations)

    @property
    def mrr(self) -> float:
        return _mean(evaluation.reciprocal_rank for evaluation in self.case_evaluations)


def normalize_text(text: str) -> str:
    """Normalize whitespace for strict but newline-insensitive evidence matching."""
    return re.sub(r"\s+", " ", text).strip()


def load_retrieval_eval_cases(path: str | Path) -> list[RetrievalEvalCase]:
    """Load the private evidence-centric JSONL dataset with basic validation."""
    eval_path = Path(path)
    cases = []
    for line_number, line in enumerate(
        eval_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        cases.append(_case_from_record(record, line_number))
    return cases


def is_relevant(document: Document, evidence: Evidence) -> bool:
    """Return true only if the same-source chunk fully contains the evidence."""
    return (
        document.metadata.get("source") == evidence.source
        and normalize_text(evidence.evidence_text)
        in normalize_text(document.page_content)
    )


def is_case_covered(case: RetrievalEvalCase, corpus: Sequence[Document]) -> bool:
    """Check whether the current chunk corpus can express any case evidence."""
    return any(
        is_relevant(document, evidence)
        for evidence in case.relevant_evidence
        for document in corpus
    )


def evaluate_case(
    case: RetrievalEvalCase,
    corpus: Sequence[Document],
    retrieved_results: list[RetrievedResult],
) -> CaseEvaluation:
    """Evaluate one top-k ranking against OR-semantics relevant evidence."""
    covered = is_case_covered(case, corpus)
    first_relevant_rank = None
    if covered:
        for rank, (document, _score) in enumerate(retrieved_results, start=1):
            if any(is_relevant(document, evidence) for evidence in case.relevant_evidence):
                first_relevant_rank = rank
                break

    return CaseEvaluation(
        case=case,
        coverage_status="OK" if covered else "COVERAGE_FAILURE",
        first_relevant_rank=first_relevant_rank,
        hit_at_1=int(first_relevant_rank is not None and first_relevant_rank <= 1),
        hit_at_3=int(first_relevant_rank is not None and first_relevant_rank <= 3),
        hit_at_5=int(first_relevant_rank is not None and first_relevant_rank <= 5),
        reciprocal_rank=0.0 if first_relevant_rank is None else 1 / first_relevant_rank,
        retrieved_results=retrieved_results,
    )


def evaluate_retrieval(
    cases: Sequence[RetrievalEvalCase],
    corpus: Sequence[Document],
    retriever: Retriever,
    *,
    k: int = 5,
) -> RetrievalEvalSummary:
    """Run every question through one retriever and aggregate V0 metrics."""
    if k <= 0:
        raise ValueError("k must be positive")

    evaluations = [
        evaluate_case(case, corpus, retriever.retrieve(case.question, k=k))
        for case in cases
    ]
    return RetrievalEvalSummary(case_evaluations=evaluations)


def case_evaluation_to_record(
    evaluation: CaseEvaluation,
    *,
    retrieval_query: str | None = None,
    query_rewrite_record: Mapping[str, object] | None = None,
    dense_candidates: Sequence[RetrievedResult] | None = None,
    detail_fields: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Convert one completed evaluation into a JSONL-safe detail record."""
    record: dict[str, object] = {
        "case_id": evaluation.case.case_id,
        "question": evaluation.case.question,
        "coverage_status": evaluation.coverage_status,
        "first_relevant_rank": evaluation.first_relevant_rank,
        "hit_at_1": evaluation.hit_at_1,
        "hit_at_3": evaluation.hit_at_3,
        "hit_at_5": evaluation.hit_at_5,
        "rr": evaluation.reciprocal_rank,
        "relevant_evidence": [
            {
                "source": evidence.source,
                "section": evidence.section,
                "subsection": evidence.subsection,
                "evidence_text": evidence.evidence_text,
            }
            for evidence in evaluation.case.relevant_evidence
        ],
        "retrieved_results": [
            _retrieved_result_to_record(rank, document, score)
            for rank, (document, score) in enumerate(
                evaluation.retrieved_results, start=1
            )
        ],
    }
    if retrieval_query is not None:
        record["retrieval_query"] = retrieval_query
    if query_rewrite_record is not None:
        record.update(query_rewrite_record)
    if dense_candidates is not None:
        record["dense_candidates"] = [
            _dense_candidate_to_record(rank, document, score)
            for rank, (document, score) in enumerate(dense_candidates, start=1)
        ]
    if detail_fields is not None:
        record.update(detail_fields)
    return record


def _retrieved_result_to_record(
    rank: int, document: Document, score: float
) -> dict[str, object]:
    record: dict[str, object] = {
        "rank": rank,
        "score": score,
        "source": document.metadata.get("source"),
        "section": document.metadata.get("section"),
        "subsection": document.metadata.get("subsection"),
        "page_content": document.page_content,
    }
    if "_dense_score" in document.metadata:
        record["dense_score"] = document.metadata["_dense_score"]
    if "_bm25_score" in document.metadata:
        record["bm25_score"] = document.metadata["_bm25_score"]
    if "_rrf_score" in document.metadata:
        record["rrf_score"] = document.metadata["_rrf_score"]
    if "_rerank_score" in document.metadata:
        record["rerank_score"] = document.metadata["_rerank_score"]
    return record


def _dense_candidate_to_record(
    dense_rank: int, document: Document, dense_score: float
) -> dict[str, object]:
    """Serialize one pre-rerank Dense candidate without rerank fields."""
    return {
        "dense_rank": dense_rank,
        "dense_score": dense_score,
        "source": document.metadata.get("source"),
        "section": document.metadata.get("section"),
        "subsection": document.metadata.get("subsection"),
        "page_content": document.page_content,
    }


def write_evaluation_details(
    path: str | Path,
    evaluations: Sequence[CaseEvaluation],
    *,
    retrieval_queries: Mapping[str, str] | None = None,
    query_rewrite_records: Mapping[str, Mapping[str, object]] | None = None,
    dense_candidate_results: Mapping[str, Sequence[RetrievedResult]] | None = None,
    detail_fields: Mapping[str, object] | None = None,
) -> None:
    """Persist completed per-case evaluation details as UTF-8 JSONL."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for evaluation in evaluations:
            retrieval_query = (
                None
                if retrieval_queries is None
                else retrieval_queries.get(evaluation.case.case_id)
            )
            query_rewrite_record = (
                None
                if query_rewrite_records is None
                else query_rewrite_records.get(evaluation.case.case_id)
            )
            dense_candidates = (
                None
                if dense_candidate_results is None
                else dense_candidate_results.get(evaluation.case.case_id)
            )
            record = case_evaluation_to_record(
                evaluation,
                retrieval_query=retrieval_query,
                query_rewrite_record=query_rewrite_record,
                dense_candidates=dense_candidates,
                detail_fields=detail_fields,
            )
            output_file.write(json.dumps(record, ensure_ascii=False))
            output_file.write("\n")


def _case_from_record(record: object, line_number: int) -> RetrievalEvalCase:
    if not isinstance(record, dict):
        raise ValueError(f"Eval JSONL line {line_number} must be an object")

    evidence_records = record.get("relevant_evidence")
    if not isinstance(evidence_records, list) or not evidence_records:
        raise ValueError(
            f"Eval JSONL line {line_number} must contain non-empty relevant_evidence"
        )

    evidence = tuple(
        _evidence_from_record(item, line_number) for item in evidence_records
    )
    expected_answer = record.get("expected_answer")
    if expected_answer is not None and not isinstance(expected_answer, str):
        raise ValueError(f"Eval JSONL line {line_number} expected_answer must be a string")

    return RetrievalEvalCase(
        case_id=_required_string(record, "case_id", line_number),
        question=_required_string(record, "question", line_number),
        expected_answer=expected_answer,
        relevant_evidence=evidence,
    )


def _evidence_from_record(record: object, line_number: int) -> Evidence:
    if not isinstance(record, dict):
        raise ValueError(f"Eval JSONL line {line_number} evidence must be an object")
    return Evidence(
        source=_required_string(record, "source", line_number),
        section=_optional_string(record, "section", line_number),
        subsection=_optional_string(record, "subsection", line_number),
        evidence_text=_required_string(record, "evidence_text", line_number),
    )


def _required_string(record: dict[str, object], name: str, line_number: int) -> str:
    value = record.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Eval JSONL line {line_number} requires non-empty {name}")
    return value


def _optional_string(
    record: dict[str, object], name: str, line_number: int
) -> str | None:
    value = record.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Eval JSONL line {line_number} {name} must be a string")
    return value


def _mean(values: Sequence[float | int] | object) -> float:
    numeric_values = list(values)  # type: ignore[arg-type]
    return sum(numeric_values) / len(numeric_values) if numeric_values else 0.0

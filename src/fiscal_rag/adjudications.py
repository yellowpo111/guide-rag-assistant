"""Supplementary, evidence-centric relevance adjudications for saved eval details."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from fiscal_rag.evaluation import normalize_text


SCHEMA_VERSION = "retrieval-eval-adjudication-v1"


@dataclass(frozen=True)
class AcceptableEvidence:
    """An additional evidence span judged sufficient for one existing eval case."""

    source: str
    section: str | None
    subsection: str | None
    evidence_text: str
    reason: str


@dataclass(frozen=True)
class RetrievalAdjudication:
    """Supplementary acceptable evidence for one immutable V1 case."""

    case_id: str
    acceptable_evidence: tuple[AcceptableEvidence, ...]


@dataclass(frozen=True)
class AdjudicatedCaseResult:
    """Strict and supplementary first-hit ranks for one saved details record."""

    case_id: str
    strict_first_relevant_rank: int | None
    adjudicated_first_relevant_rank: int | None

    @property
    def strict_hit_at_1(self) -> int:
        return int(self.strict_first_relevant_rank == 1)

    @property
    def adjudicated_hit_at_1(self) -> int:
        return int(self.adjudicated_first_relevant_rank == 1)

    @property
    def strict_only_rank_1_override(self) -> bool:
        return self.strict_first_relevant_rank != 1 and self.adjudicated_first_relevant_rank == 1


@dataclass(frozen=True)
class AdjudicatedSummary:
    """Aggregate strict and supplementary metrics over one saved details file."""

    case_results: tuple[AdjudicatedCaseResult, ...]

    @property
    def total_cases(self) -> int:
        return len(self.case_results)

    @property
    def strict_hit_at_1(self) -> float:
        return _mean(result.strict_hit_at_1 for result in self.case_results)

    @property
    def adjudicated_hit_at_1(self) -> float:
        return _mean(result.adjudicated_hit_at_1 for result in self.case_results)

    @property
    def strict_mrr(self) -> float:
        return _mean(_reciprocal_rank(result.strict_first_relevant_rank) for result in self.case_results)

    @property
    def adjudicated_mrr(self) -> float:
        return _mean(
            _reciprocal_rank(result.adjudicated_first_relevant_rank)
            for result in self.case_results
        )

    @property
    def strict_only_rank_1_overrides(self) -> tuple[AdjudicatedCaseResult, ...]:
        return tuple(
            result for result in self.case_results if result.strict_only_rank_1_override
        )


def load_retrieval_adjudications(path: str | Path) -> dict[str, RetrievalAdjudication]:
    """Load private supplementary evidence without changing the frozen V1 dataset."""
    adjudications: dict[str, RetrievalAdjudication] = {}
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        adjudication = _adjudication_from_record(record, line_number)
        if adjudication.case_id in adjudications:
            raise ValueError(
                f"Adjudication line {line_number} repeats case_id: "
                f"{adjudication.case_id!r}"
            )
        adjudications[adjudication.case_id] = adjudication
    if not adjudications:
        raise ValueError("Adjudication JSONL contains no records.")
    return adjudications


def validate_adjudications_against_corpus(
    adjudications: Mapping[str, RetrievalAdjudication], corpus_dir: str | Path
) -> list[str]:
    """Return deterministic source/evidence validation errors for private annotations."""
    root = Path(corpus_dir)
    errors: list[str] = []
    for case_id, adjudication in adjudications.items():
        for evidence in adjudication.acceptable_evidence:
            source_path = root / evidence.source
            if not source_path.is_file():
                errors.append(f"{case_id}: missing source {evidence.source!r}")
                continue
            source_text = source_path.read_text(encoding="utf-8")
            if normalize_text(evidence.evidence_text) not in normalize_text(source_text):
                errors.append(
                    f"{case_id}: acceptable evidence is not found in {evidence.source!r}"
                )
    return errors


def summarize_saved_details(
    details_records: Sequence[Mapping[str, object]],
    adjudications: Mapping[str, RetrievalAdjudication],
) -> AdjudicatedSummary:
    """Calculate supplementary ranks from saved Top-k details without any API call."""
    results: list[AdjudicatedCaseResult] = []
    seen_case_ids: set[str] = set()
    for record in details_records:
        case_id = _required_string(record, "case_id")
        if case_id in seen_case_ids:
            raise ValueError(f"Details file repeats case_id: {case_id!r}")
        seen_case_ids.add(case_id)
        strict_rank = _optional_rank(record.get("first_relevant_rank"))
        extra_rank = _first_acceptable_evidence_rank(
            record.get("retrieved_results"), adjudications.get(case_id)
        )
        results.append(
            AdjudicatedCaseResult(
                case_id=case_id,
                strict_first_relevant_rank=strict_rank,
                adjudicated_first_relevant_rank=_minimum_rank(strict_rank, extra_rank),
            )
        )
    return AdjudicatedSummary(case_results=tuple(results))


def _adjudication_from_record(record: object, line_number: int) -> RetrievalAdjudication:
    if not isinstance(record, Mapping):
        raise ValueError(f"Adjudication line {line_number} must be an object.")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Adjudication line {line_number} must use schema_version "
            f"{SCHEMA_VERSION!r}."
        )
    evidence_records = record.get("acceptable_evidence")
    if not isinstance(evidence_records, list) or not evidence_records:
        raise ValueError(
            f"Adjudication line {line_number} requires non-empty acceptable_evidence."
        )
    return RetrievalAdjudication(
        case_id=_required_string(record, "case_id", line_number),
        acceptable_evidence=tuple(
            _acceptable_evidence_from_record(item, line_number)
            for item in evidence_records
        ),
    )


def _acceptable_evidence_from_record(
    record: object, line_number: int
) -> AcceptableEvidence:
    if not isinstance(record, Mapping):
        raise ValueError(
            f"Adjudication line {line_number} acceptable evidence must be an object."
        )
    return AcceptableEvidence(
        source=_required_string(record, "source", line_number),
        section=_optional_string(record, "section", line_number),
        subsection=_optional_string(record, "subsection", line_number),
        evidence_text=_required_string(record, "evidence_text", line_number),
        reason=_required_string(record, "reason", line_number),
    )


def _first_acceptable_evidence_rank(
    retrieved_results: object, adjudication: RetrievalAdjudication | None
) -> int | None:
    if adjudication is None:
        return None
    if not isinstance(retrieved_results, list):
        raise ValueError("Details record requires a retrieved_results list.")
    for result in retrieved_results:
        if not isinstance(result, Mapping):
            raise ValueError("Details retrieved result must be an object.")
        rank = _optional_rank(result.get("rank"))
        source = result.get("source")
        page_content = result.get("page_content")
        if rank is None or not isinstance(source, str) or not isinstance(page_content, str):
            raise ValueError(
                "Details retrieved result requires integer rank, string source, and page_content."
            )
        if any(
            source == evidence.source
            and normalize_text(evidence.evidence_text) in normalize_text(page_content)
            for evidence in adjudication.acceptable_evidence
        ):
            return rank
    return None


def _required_string(
    record: Mapping[str, object], name: str, line_number: int | None = None
) -> str:
    value = record.get(name)
    if isinstance(value, str) and value.strip():
        return value
    location = "Details record" if line_number is None else f"Adjudication line {line_number}"
    raise ValueError(f"{location} requires non-empty {name}.")


def _optional_string(
    record: Mapping[str, object], name: str, line_number: int
) -> str | None:
    value = record.get(name)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise ValueError(f"Adjudication line {line_number} {name} must be a string or null.")


def _optional_rank(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    raise ValueError("Rank must be a positive integer or null.")


def _minimum_rank(first: int | None, second: int | None) -> int | None:
    if first is None:
        return second
    if second is None:
        return first
    return min(first, second)


def _reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1 / rank


def _mean(values: Sequence[float | int] | object) -> float:
    numeric_values = list(values)  # type: ignore[arg-type]
    return sum(numeric_values) / len(numeric_values) if numeric_values else 0.0

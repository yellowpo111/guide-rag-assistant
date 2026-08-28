"""Schemas and aggregation for end-to-end assistant evaluation."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


ASSISTANT_EVAL_SCHEMA_VERSION = "assistant-eval-v1"
ASSISTANT_ADJUDICATION_SCHEMA_VERSION = "assistant-eval-adjudication-v1"
ASSISTANT_RESULT_SCHEMA_VERSION = "assistant-eval-result-v1"
ASSISTANT_ROUTES = ("rag", "chat", "out_of_scope")
ASSISTANT_CATEGORIES = (
    "rag_answerable",
    "rag_unanswerable",
    "chat",
    "out_of_scope",
    "routing_boundary",
)


@dataclass(frozen=True)
class AssistantEvalCase:
    case_id: str
    category: str
    question: str
    expected_route: str
    answerability: bool | None
    retrieval_case_id: str | None = None


@dataclass(frozen=True)
class AssistantAdjudication:
    case_id: str
    review_status: str
    route_correct: bool | None
    answer_support: str
    answer_correctness: str
    abstention: str
    boundary_compliance: str
    source_trace_quality: str
    reason: str


def load_assistant_eval_cases(path: str | Path) -> list[AssistantEvalCase]:
    cases: list[AssistantEvalCase] = []
    seen: set[str] = set()
    for line_number, record in _load_jsonl(path):
        if record.get("schema_version") != ASSISTANT_EVAL_SCHEMA_VERSION:
            raise ValueError(
                f"Assistant Eval line {line_number} must use schema_version "
                f"{ASSISTANT_EVAL_SCHEMA_VERSION!r}."
            )
        case_id = _required_string(record, "case_id", line_number)
        if case_id in seen:
            raise ValueError(f"Assistant Eval repeats case_id: {case_id!r}")
        seen.add(case_id)
        category = _choice(record, "category", ASSISTANT_CATEGORIES, line_number)
        expected_route = _choice(
            record, "expected_route", ASSISTANT_ROUTES, line_number
        )
        answerability = record.get("answerability")
        if answerability is not None and not isinstance(answerability, bool):
            raise ValueError(
                f"Assistant Eval line {line_number} answerability must be boolean or null."
            )
        retrieval_case_id = record.get("retrieval_case_id")
        if retrieval_case_id is not None and not isinstance(retrieval_case_id, str):
            raise ValueError(
                f"Assistant Eval line {line_number} retrieval_case_id must be a string."
            )
        if category == "rag_answerable" and not retrieval_case_id:
            raise ValueError(
                f"Assistant Eval line {line_number} answerable RAG case requires retrieval_case_id."
            )
        _validate_case_contract(
            category,
            expected_route,
            answerability,
            line_number=line_number,
        )
        cases.append(
            AssistantEvalCase(
                case_id=case_id,
                category=category,
                question=_required_string(record, "question", line_number),
                expected_route=expected_route,
                answerability=answerability,
                retrieval_case_id=retrieval_case_id,
            )
        )
    if not cases:
        raise ValueError("Assistant Eval JSONL contains no cases.")
    return cases


def _validate_case_contract(
    category: str,
    expected_route: str,
    answerability: bool | None,
    *,
    line_number: int,
) -> None:
    fixed_contracts = {
        "rag_answerable": ("rag", True),
        "rag_unanswerable": ("rag", False),
        "chat": ("chat", None),
        "out_of_scope": ("out_of_scope", None),
    }
    expected = fixed_contracts.get(category)
    if expected is not None and (expected_route, answerability) != expected:
        raise ValueError(
            f"Assistant Eval line {line_number} category conflicts with "
            "expected_route or answerability."
        )
    if category == "routing_boundary" and (
        expected_route != "rag" or not isinstance(answerability, bool)
    ):
        raise ValueError(
            f"Assistant Eval line {line_number} routing_boundary requires "
            "expected_route='rag' and boolean answerability."
        )


def load_assistant_adjudications(
    path: str | Path,
) -> dict[str, AssistantAdjudication]:
    adjudications: dict[str, AssistantAdjudication] = {}
    choices = {
        "answer_support": ("pending", "supported", "partially_supported", "unsupported", "not_applicable"),
        "answer_correctness": ("pending", "correct", "partially_correct", "incorrect", "not_applicable"),
        "abstention": ("pending", "appropriate", "inappropriate", "not_applicable"),
        "boundary_compliance": ("pending", "compliant", "non_compliant", "not_applicable"),
        "source_trace_quality": ("pending", "sufficient", "partial", "insufficient", "not_applicable"),
    }
    for line_number, record in _load_jsonl(path):
        if record.get("schema_version") != ASSISTANT_ADJUDICATION_SCHEMA_VERSION:
            raise ValueError(
                f"Assistant adjudication line {line_number} has an unsupported schema_version."
            )
        case_id = _required_string(record, "case_id", line_number)
        if case_id in adjudications:
            raise ValueError(f"Assistant adjudications repeat case_id: {case_id!r}")
        route_correct = record.get("route_correct")
        if route_correct is not None and not isinstance(route_correct, bool):
            raise ValueError(
                f"Assistant adjudication line {line_number} route_correct must be boolean or null."
            )
        adjudications[case_id] = AssistantAdjudication(
            case_id=case_id,
            review_status=_required_string(record, "review_status", line_number),
            route_correct=route_correct,
            answer_support=_choice(record, "answer_support", choices["answer_support"], line_number),
            answer_correctness=_choice(
                record, "answer_correctness", choices["answer_correctness"], line_number
            ),
            abstention=_choice(record, "abstention", choices["abstention"], line_number),
            boundary_compliance=_choice(
                record, "boundary_compliance", choices["boundary_compliance"], line_number
            ),
            source_trace_quality=_choice(
                record, "source_trace_quality", choices["source_trace_quality"], line_number
            ),
            reason=_required_string(record, "reason", line_number),
        )
    if not adjudications:
        raise ValueError("Assistant adjudication JSONL contains no records.")
    return adjudications


def validate_adjudication_coverage(
    cases: Sequence[AssistantEvalCase],
    adjudications: Mapping[str, AssistantAdjudication],
) -> None:
    case_ids = {case.case_id for case in cases}
    missing = sorted(case_ids - set(adjudications))
    unknown = sorted(set(adjudications) - case_ids)
    if missing or unknown:
        raise ValueError(
            "Assistant adjudication coverage mismatch: "
            f"missing={missing}, unknown={unknown}"
        )
    unconfirmed = sorted(
        case_id
        for case_id, adjudication in adjudications.items()
        if adjudication.review_status != "user_confirmed"
    )
    if unconfirmed:
        raise ValueError("Assistant adjudications are not user-confirmed: " + ", ".join(unconfirmed))
    pending = sorted(
        case_id
        for case_id, adjudication in adjudications.items()
        if adjudication.route_correct is None
        or "pending"
        in {
            adjudication.answer_support,
            adjudication.answer_correctness,
            adjudication.abstention,
            adjudication.boundary_compliance,
            adjudication.source_trace_quality,
        }
    )
    if pending:
        raise ValueError("Assistant adjudications still contain pending fields: " + ", ".join(pending))


def summarize_assistant_results(
    cases: Sequence[AssistantEvalCase],
    results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    cases_by_id = {case.case_id: case for case in cases}
    results_by_id: dict[str, Mapping[str, object]] = {}
    for result in results:
        case_id = result.get("case_id")
        if not isinstance(case_id, str) or case_id not in cases_by_id:
            raise ValueError("Assistant result has an unknown or missing case_id.")
        if case_id in results_by_id:
            raise ValueError(f"Assistant results repeat case_id: {case_id!r}")
        results_by_id[case_id] = result

    confusion = {
        expected: {actual: 0 for actual in ASSISTANT_ROUTES}
        for expected in ASSISTANT_ROUTES
    }
    completed = 0
    route_correct = 0
    trace_complete = 0
    rag_result_count = 0
    critical_rag_route_failures: list[str] = []
    for case in cases:
        result = results_by_id.get(case.case_id, {})
        is_completed = result.get("status") == "completed"
        completed += int(is_completed)
        actual_route = result.get("actual_route")
        if isinstance(actual_route, str) and actual_route in ASSISTANT_ROUTES:
            confusion[case.expected_route][actual_route] += 1
            route_correct += int(actual_route == case.expected_route)
        if case.expected_route == "rag" and actual_route != "rag":
            critical_rag_route_failures.append(case.case_id)
        if actual_route == "rag":
            rag_result_count += 1
            trace_complete += int(_trace_is_complete(result.get("trace")))

    total = len(cases)
    expected_counts = Counter(case.expected_route for case in cases)
    f1_by_route = {
        route: _route_f1(confusion, route, expected_counts[route])
        for route in ASSISTANT_ROUTES
    }
    return {
        "schema_version": "assistant-eval-summary-v1",
        "total_cases": total,
        "completed_cases": completed,
        "sse_completion_rate": completed / total if total else 0.0,
        "route_accuracy": route_correct / total if total else 0.0,
        "route_macro_f1": sum(f1_by_route.values()) / len(ASSISTANT_ROUTES),
        "route_f1": f1_by_route,
        "confusion_matrix": confusion,
        "rag_trace_completion_rate": (
            trace_complete / rag_result_count if rag_result_count else 0.0
        ),
        "critical_rag_route_failures": critical_rag_route_failures,
        "error_count": total - completed,
    }


def summarize_adjudications(
    cases: Sequence[AssistantEvalCase],
    adjudications: Mapping[str, AssistantAdjudication],
) -> dict[str, object]:
    validate_adjudication_coverage(cases, adjudications)
    fields = (
        "answer_support",
        "answer_correctness",
        "abstention",
        "boundary_compliance",
        "source_trace_quality",
    )
    by_category: dict[str, dict[str, dict[str, int]]] = {}
    blockers: list[str] = []
    for case in cases:
        adjudication = adjudications[case.case_id]
        category_summary = by_category.setdefault(case.category, {})
        for field_name in fields:
            counter = category_summary.setdefault(field_name, {})
            value = str(getattr(adjudication, field_name))
            counter[value] = counter.get(value, 0) + 1
        if (
            not adjudication.route_correct
            or adjudication.answer_support == "unsupported"
            or adjudication.answer_correctness == "incorrect"
            or adjudication.boundary_compliance == "non_compliant"
        ):
            blockers.append(case.case_id)
    return {
        "schema_version": "assistant-eval-adjudication-summary-v1",
        "total_confirmed": len(adjudications),
        "by_category": by_category,
        "release_blockers": blockers,
    }


def result_record(
    case: AssistantEvalCase,
    *,
    attempt: int,
    run_id: str,
    result: object,
) -> dict[str, object]:
    """Convert an AssistantSseResult-like object into a private JSONL record."""
    if attempt <= 0:
        raise ValueError("attempt must be positive")
    return {
        "schema_version": ASSISTANT_RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "attempt": attempt,
        "case_id": case.case_id,
        "category": case.category,
        "question": case.question,
        "expected_route": case.expected_route,
        "actual_route": getattr(result, "route"),
        "answerability": case.answerability,
        "retrieval_case_id": case.retrieval_case_id,
        "status": getattr(result, "status"),
        "request_id": getattr(result, "request_id"),
        "events": list(getattr(result, "events")),
        "trace": getattr(result, "trace"),
        "answer": getattr(result, "answer"),
        "timings_ms": getattr(result, "timings_ms"),
        "client_ttft_ms": getattr(result, "client_ttft_ms"),
        "client_total_ms": getattr(result, "client_total_ms"),
        "error_code": getattr(result, "error_code"),
        "error_message": getattr(result, "error_message"),
    }


def adjudication_template_records(
    cases: Sequence[AssistantEvalCase],
    results: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Create a review template while preserving automatic route observations."""
    results_by_id = {
        str(record.get("case_id")): record
        for record in results
        if isinstance(record.get("case_id"), str)
    }
    records: list[dict[str, object]] = []
    for case in cases:
        result = results_by_id.get(case.case_id, {})
        actual_route = result.get("actual_route")
        records.append(
            {
                "schema_version": ASSISTANT_ADJUDICATION_SCHEMA_VERSION,
                "case_id": case.case_id,
                "category": case.category,
                "question": case.question,
                "expected_route": case.expected_route,
                "actual_route": actual_route,
                "answerability": case.answerability,
                "retrieval_case_id": case.retrieval_case_id,
                "execution_status": result.get("status"),
                "answer": result.get("answer"),
                "trace": result.get("trace"),
                "review_status": "pending",
                "route_correct": (
                    actual_route == case.expected_route
                    if isinstance(actual_route, str)
                    else None
                ),
                "answer_support": "pending",
                "answer_correctness": "pending",
                "abstention": "pending",
                "boundary_compliance": "pending",
                "source_trace_quality": "pending",
                "reason": "Pending human review.",
            }
        )
    return records


def write_jsonl_exclusive(path: str | Path, records: Sequence[Mapping[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False))
            output.write("\n")


def load_jsonl_records(path: str | Path) -> list[dict[str, object]]:
    return [record for _line_number, record in _load_jsonl(path)]


def merge_attempt_records(
    cases: Sequence[AssistantEvalCase],
    attempt_groups: Sequence[Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    """Select the highest attempt per case without changing source artifacts."""
    case_ids = {case.case_id for case in cases}
    selected: dict[str, tuple[int, dict[str, object]]] = {}
    run_ids: set[str] = set()
    for group in attempt_groups:
        for record in group:
            if record.get("schema_version") != ASSISTANT_RESULT_SCHEMA_VERSION:
                raise ValueError("Assistant result has an unsupported schema_version.")
            case_id = record.get("case_id")
            if not isinstance(case_id, str) or case_id not in case_ids:
                raise ValueError("Assistant result has an unknown or missing case_id.")
            run_id = record.get("run_id")
            if not isinstance(run_id, str) or not run_id.strip():
                raise ValueError("Assistant result requires a non-empty run_id.")
            run_ids.add(run_id)
            attempt = record.get("attempt")
            if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt <= 0:
                raise ValueError("Assistant result attempt must be a positive integer.")
            current = selected.get(case_id)
            if current is not None and current[0] == attempt:
                raise ValueError(
                    f"Assistant results repeat case_id {case_id!r} at attempt {attempt}."
                )
            if current is None or attempt > current[0]:
                selected[case_id] = (attempt, dict(record))
    if len(run_ids) != 1:
        raise ValueError("Assistant attempt files must use one shared run_id.")
    missing = sorted(case_ids - set(selected))
    if missing:
        raise ValueError("Merged Assistant results are missing cases: " + ", ".join(missing))
    return [selected[case.case_id][1] for case in cases]


def _trace_is_complete(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    sources = value.get("sources")
    if not isinstance(sources, list) or not 1 <= len(sources) <= 5:
        return False
    required_trace_fields = {"retrieval_query", "query_rewrite_status", "sources"}
    if not required_trace_fields.issubset(value):
        return False
    for expected_rank, source in enumerate(sources, start=1):
        if not isinstance(source, Mapping) or source.get("rank") != expected_rank:
            return False
        if not isinstance(source.get("source"), str) or not source["source"].strip():
            return False
        if not isinstance(source.get("rerank_score"), (int, float)):
            return False
    return True


def _route_f1(
    confusion: Mapping[str, Mapping[str, int]],
    route: str,
    expected_count: int,
) -> float:
    true_positive = confusion[route][route]
    false_positive = sum(confusion[expected][route] for expected in ASSISTANT_ROUTES if expected != route)
    false_negative = expected_count - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def _load_jsonl(path: str | Path) -> list[tuple[int, dict[str, object]]]:
    records: list[tuple[int, dict[str, object]]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"JSONL line {line_number} must be an object.")
        records.append((line_number, record))
    return records


def _required_string(record: Mapping[str, object], name: str, line_number: int) -> str:
    value = record.get(name)
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError(f"JSONL line {line_number} requires non-empty {name}.")


def _choice(
    record: Mapping[str, object],
    name: str,
    choices: Sequence[str],
    line_number: int,
) -> str:
    value = _required_string(record, name, line_number)
    if value not in choices:
        raise ValueError(
            f"JSONL line {line_number} {name} must be one of: " + ", ".join(choices)
        )
    return value

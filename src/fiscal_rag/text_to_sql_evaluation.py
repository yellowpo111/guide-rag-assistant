"""Synthetic execution evaluation for the usage Text-to-SQL prototype."""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from fiscal_rag.text_to_sql import (
    ReadOnlyUsageQueryExecutor,
    SQLGenerator,
    run_usage_text_to_sql,
)
from fiscal_rag.usage import SQLiteUsageRepository


TEXT_TO_SQL_EVAL_SCHEMA_VERSION = "usage-text-to-sql-eval-v1"
SYNTHETIC_DATA_ORIGIN = "synthetic_usage_fixture_v1"
MINIMUM_ANSWERABLE_MATCHES = 8
EXPECTED_ANSWERABLE_CASES = 10
EXPECTED_REFUSAL_CASES = 2


@dataclass(frozen=True)
class TextToSQLEvalCase:
    case_id: str
    category: str
    question: str
    gold_sql: str | None
    expected_refusal: bool
    order_sensitive: bool


def load_text_to_sql_eval_cases(path: str | Path) -> list[TextToSQLEvalCase]:
    cases: list[TextToSQLEvalCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"Text-to-SQL Eval line {line_number} must be an object.")
        if record.get("schema_version") != TEXT_TO_SQL_EVAL_SCHEMA_VERSION:
            raise ValueError(f"Text-to-SQL Eval line {line_number} has an invalid schema.")
        case_id = _required_string(record, "case_id", line_number)
        if case_id in seen:
            raise ValueError(f"Text-to-SQL Eval repeats case_id: {case_id!r}")
        seen.add(case_id)
        expected_refusal = record.get("expected_refusal")
        order_sensitive = record.get("order_sensitive")
        if not isinstance(expected_refusal, bool) or not isinstance(order_sensitive, bool):
            raise ValueError(
                f"Text-to-SQL Eval line {line_number} requires boolean flags."
            )
        gold_sql = record.get("gold_sql")
        if expected_refusal:
            if gold_sql is not None:
                raise ValueError(f"Refusal case {case_id!r} must not define gold_sql.")
        elif not isinstance(gold_sql, str) or not gold_sql.strip():
            raise ValueError(f"Answerable case {case_id!r} requires gold_sql.")
        cases.append(
            TextToSQLEvalCase(
                case_id=case_id,
                category=_required_string(record, "category", line_number),
                question=_required_string(record, "question", line_number),
                gold_sql=gold_sql.strip() if isinstance(gold_sql, str) else None,
                expected_refusal=expected_refusal,
                order_sensitive=order_sensitive,
            )
        )
    if len(cases) != EXPECTED_ANSWERABLE_CASES + EXPECTED_REFUSAL_CASES:
        raise ValueError("Text-to-SQL Eval v1 must contain exactly 12 cases.")
    if sum(not case.expected_refusal for case in cases) != EXPECTED_ANSWERABLE_CASES:
        raise ValueError("Text-to-SQL Eval v1 must contain 10 answerable cases.")
    return cases


def build_synthetic_usage_fixture(path: str | Path) -> Path:
    """Create a deterministic database containing no real user content."""
    database = Path(path)
    if database.exists():
        raise FileExistsError(database)
    repository = SQLiteUsageRepository(database)
    repository.initialize()
    requests = [
        _request("p-rag-fast", "2026-08-20T08:00:00+00:00", "assistant_stream", "production", "rag", "completed", 3000),
        _request("p-rag-slow", "2026-08-21T08:00:00+00:00", "assistant_stream", "production", "rag", "completed", 7000),
        _request("p-chat", "2026-08-22T08:00:00+00:00", "assistant_stream", "production", "chat", "completed", 1000),
        _request("p-out", "2026-08-23T08:00:00+00:00", "assistant_stream", "production", "out_of_scope", "completed", 200),
        _request("p-rag-failed", "2026-08-24T08:00:00+00:00", "assistant_stream", "production", "rag", "failed", 2500, "model_request_failed", "generation"),
        _request("p-route-failed", "2026-08-25T08:00:00+00:00", "assistant_stream", "production", None, "failed", 900, "model_request_failed", "route"),
        _request("p-aborted", "2026-08-26T08:00:00+00:00", "assistant_stream", "production", "rag", "aborted", 1200, "client_disconnected", "client"),
        _request("p-interrupted", "2026-08-27T08:00:00+00:00", "assistant_stream", "production", None, "interrupted", 500, "service_restart", "service"),
        _request("p-started", "2026-08-28T08:00:00+00:00", "assistant_stream", "production", None, "started", None),
        _request("p-ask", "2026-08-29T08:00:00+00:00", "ask", "production", "rag", "completed", 4000),
        _request("p-chat-review", "2026-08-30T08:00:00+00:00", "assistant_stream", "production", "chat", "completed", 1500),
        _request("eval-rag", "2026-08-31T08:00:00+00:00", "assistant_stream", "assistant_eval", "rag", "completed", 99_000),
        _request("perf-rag", "2026-09-01T08:00:00+00:00", "assistant_stream", "performance_eval", "rag", "failed", 99_000, "synthetic_eval_error", "generation"),
    ]
    timings = [
        ("p-rag-fast", "router", 500.0),
        ("p-rag-fast", "rewrite", 600.0),
        ("p-rag-fast", "query_embedding", 100.0),
        ("p-rag-fast", "vector_search", 10.0),
        ("p-rag-fast", "rerank", 300.0),
        ("p-rag-fast", "server_total", 3000.0),
        ("p-rag-slow", "router", 800.0),
        ("p-rag-slow", "rewrite", 900.0),
        ("p-rag-slow", "query_embedding", 120.0),
        ("p-rag-slow", "vector_search", 10.0),
        ("p-rag-slow", "rerank", 500.0),
        ("p-rag-slow", "server_total", 7000.0),
        ("p-rag-failed", "router", 700.0),
        ("p-rag-failed", "rewrite", 800.0),
        ("p-rag-failed", "generation", 1000.0),
        ("p-chat", "server_total", 1000.0),
        ("p-chat-review", "server_total", 1500.0),
        ("eval-rag", "server_total", 99_000.0),
        ("perf-rag", "server_total", 99_000.0),
    ]
    sources = [
        (request_id, rank, f"SYNTHETIC-SOURCE-{rank}", "SYNTHETIC", "SYNTHETIC", 0.5, 0.8)
        for request_id in ("p-rag-fast", "p-rag-slow", "eval-rag")
        for rank in range(1, 6)
    ]
    feedback = [
        ("p-rag-slow", "negative", "2026-08-21T09:00:00+00:00", "2026-08-21T09:00:00+00:00"),
        ("p-chat", "positive", "2026-08-22T09:00:00+00:00", "2026-08-22T09:00:00+00:00"),
    ]
    reviews = [
        _review("p-rag-slow", "retrieval", "major", "rag", 1, 1, "fixture-rag-001", "rag_answerable", "fixture-ret-001", "2026-08-22T00:00:00+00:00"),
        _review("p-chat-review", "generation_correctness", "minor", "chat", None, 0, None, None, None, "2026-08-31T00:00:00+00:00"),
        _review("p-route-failed", "upstream_transport", "major", "rag", 1, 0, None, None, None, "2026-08-26T00:00:00+00:00"),
    ]
    connection = sqlite3.connect(database)
    try:
        with connection:
            connection.executemany(
                """
                INSERT INTO assistant_requests (
                    request_id, endpoint, started_at_utc, finished_at_utc,
                    service_version, profile_id, traffic_kind, route,
                    execution_status, question, answer, retrieval_query,
                    rewrite_query, query_rewrite_status,
                    required_constraints_json, missing_constraints_json,
                    error_code, error_type, failure_stage, total_duration_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                requests,
            )
            connection.executemany(
                "INSERT INTO request_timings (request_id, stage, duration_ms) VALUES (?, ?, ?)",
                timings,
            )
            connection.executemany(
                """
                INSERT INTO request_sources (
                    request_id, rank, source, section, subsection, dense_score, rerank_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                sources,
            )
            connection.executemany(
                """
                INSERT INTO request_feedback (
                    request_id, rating, created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?)
                """,
                feedback,
            )
            connection.executemany(
                """
                INSERT INTO request_reviews (
                    request_id, review_status, failure_type, severity,
                    expected_route, answerability, reason, eval_candidate,
                    candidate_case_id, candidate_category, retrieval_case_id,
                    expected_answer, evidence_json, topic, reviewed_at_utc
                ) VALUES (?, 'user_confirmed', ?, ?, ?, ?, 'SYNTHETIC REVIEW', ?, ?, ?, ?, NULL, NULL, NULL, ?)
                """,
                reviews,
            )
    finally:
        connection.close()
    return database


def evaluate_text_to_sql_cases(
    cases: Sequence[TextToSQLEvalCase],
    *,
    generator: SQLGenerator,
    database_path: str | Path,
    run_id: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    executor = ReadOnlyUsageQueryExecutor(database_path)
    details: list[dict[str, object]] = []
    for case in cases:
        observed = run_usage_text_to_sql(
            case.question,
            generator=generator,
            executor=executor,
        )
        detail = _evaluate_case(case, observed, executor, run_id=run_id)
        details.append(detail)
    summary = summarize_text_to_sql_eval(details, run_id=run_id)
    return details, summary


def summarize_text_to_sql_eval(
    details: Sequence[Mapping[str, object]], *, run_id: str
) -> dict[str, object]:
    answerable = [record for record in details if not record.get("expected_refusal")]
    refusals = [record for record in details if record.get("expected_refusal")]
    matched = sum(record.get("correct") is True for record in answerable)
    refusal_matches = sum(record.get("correct") is True for record in refusals)
    total = len(details)
    generation_successes = sum(
        record.get("error_code") not in {"model_transport_error", "model_output_invalid"}
        for record in details
    )
    validation_acceptances = sum(
        bool(record.get("generated_sql")) and record.get("error_code") != "sql_rejected"
        for record in details
    )
    execution_successes = sum(record.get("execution_status") == "completed" for record in details)
    meets_threshold = (
        len(answerable) == EXPECTED_ANSWERABLE_CASES
        and len(refusals) == EXPECTED_REFUSAL_CASES
        and matched >= MINIMUM_ANSWERABLE_MATCHES
        and refusal_matches == EXPECTED_REFUSAL_CASES
    )
    failure_counts: dict[str, int] = {}
    for record in details:
        failure = record.get("failure_type")
        if isinstance(failure, str):
            failure_counts[failure] = failure_counts.get(failure, 0) + 1
    return {
        "schema_version": "usage-text-to-sql-eval-summary-v1",
        "run_id": run_id,
        "data_origin": SYNTHETIC_DATA_ORIGIN,
        "contains_real_usage_data": False,
        "total_cases": total,
        "answerable_cases": len(answerable),
        "refusal_cases": len(refusals),
        "generation_success_rate": _ratio(generation_successes, total),
        "validation_acceptance_rate": _ratio(validation_acceptances, total),
        "execution_success_rate": _ratio(execution_successes, len(answerable)),
        "denotation_accuracy": _ratio(matched, len(answerable)),
        "refusal_accuracy": _ratio(refusal_matches, len(refusals)),
        "answerable_matches": matched,
        "refusal_matches": refusal_matches,
        "failure_types": dict(sorted(failure_counts.items())),
        "minimum_answerable_matches": MINIMUM_ANSWERABLE_MATCHES,
        "prototype_decision": "adopted" if meets_threshold else "not_adopted",
    }


def _evaluate_case(
    case: TextToSQLEvalCase,
    observed: Mapping[str, object],
    executor: ReadOnlyUsageQueryExecutor,
    *,
    run_id: str,
) -> dict[str, object]:
    status = observed.get("status")
    error_code = observed.get("error_code")
    if case.expected_refusal:
        correct = status == "refused" or error_code == "sql_rejected"
        failure_type = None if correct else "result_mismatch"
        expected_columns: list[str] = []
        expected_rows: list[list[object]] = []
    else:
        gold = executor.execute(case.gold_sql or "")
        expected_columns = list(gold.columns)
        expected_rows = [list(row) for row in gold.rows]
        correct = (
            status == "completed"
            and not observed.get("truncated")
            and _results_match(
                observed.get("rows"),
                expected_rows,
                order_sensitive=case.order_sensitive,
            )
            and len(list(observed.get("columns") or [])) == len(expected_columns)
        )
        failure_type = None if correct else _failure_type(observed)
    return {
        "schema_version": "usage-text-to-sql-eval-detail-v1",
        "run_id": run_id,
        "data_origin": SYNTHETIC_DATA_ORIGIN,
        "contains_real_usage_data": False,
        "case_id": case.case_id,
        "category": case.category,
        "question": case.question,
        "expected_refusal": case.expected_refusal,
        "order_sensitive": case.order_sensitive,
        "generated_sql": observed.get("sql"),
        "execution_status": status,
        "error_code": error_code,
        "failure_type": failure_type,
        "columns": observed.get("columns"),
        "rows": observed.get("rows"),
        "expected_columns": expected_columns,
        "expected_rows": expected_rows,
        "correct": correct,
        "generation_duration_ms": observed.get("generation_duration_ms"),
        "execution_duration_ms": observed.get("execution_duration_ms"),
    }


def _results_match(
    observed: object,
    expected: Sequence[Sequence[object]],
    *,
    order_sensitive: bool,
) -> bool:
    if not isinstance(observed, list):
        return False
    observed_rows = [tuple(row) for row in observed if isinstance(row, list)]
    expected_rows = [tuple(row) for row in expected]
    if len(observed_rows) != len(observed) or len(observed_rows) != len(expected_rows):
        return False
    if not order_sensitive:
        observed_rows.sort(key=_row_sort_key)
        expected_rows.sort(key=_row_sort_key)
    return all(
        len(observed_row) == len(expected_row)
        and all(
            _values_match(observed_value, expected_value)
            for observed_value, expected_value in zip(
                observed_row, expected_row, strict=True
            )
        )
        for observed_row, expected_row in zip(
            observed_rows, expected_rows, strict=True
        )
    )


def _values_match(observed: object, expected: object) -> bool:
    if (
        isinstance(observed, (int, float))
        and not isinstance(observed, bool)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
    ):
        return math.isclose(float(observed), float(expected), rel_tol=1e-6, abs_tol=1e-6)
    return observed == expected


def _row_sort_key(row: Sequence[object]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def _failure_type(observed: Mapping[str, object]) -> str:
    code = observed.get("error_code")
    if isinstance(code, str):
        return code
    if observed.get("truncated"):
        return "result_truncated"
    return "result_mismatch"


def _request(
    request_id: str,
    started: str,
    endpoint: str,
    traffic_kind: str,
    route: str | None,
    status: str,
    total_ms: float | None,
    error_code: str | None = None,
    failure_stage: str | None = None,
) -> tuple[object, ...]:
    finished = None if status == "started" else started
    trace = "SYNTHETIC RETRIEVAL" if route == "rag" else None
    return (
        request_id,
        endpoint,
        started,
        finished,
        "1.4.0",
        "synthetic-profile",
        traffic_kind,
        route,
        status,
        "SYNTHETIC QUESTION",
        "SYNTHETIC ANSWER",
        trace,
        trace,
        "accepted" if trace else None,
        "[]",
        "[]",
        error_code,
        "SyntheticError" if error_code else None,
        failure_stage,
        total_ms,
    )


def _review(
    request_id: str,
    failure_type: str,
    severity: str,
    expected_route: str,
    answerability: int | None,
    eval_candidate: int,
    candidate_case_id: str | None,
    candidate_category: str | None,
    retrieval_case_id: str | None,
    reviewed_at: str,
) -> tuple[object, ...]:
    return (
        request_id,
        failure_type,
        severity,
        expected_route,
        answerability,
        eval_candidate,
        candidate_case_id,
        candidate_category,
        retrieval_case_id,
        reviewed_at,
    )


def _required_string(
    record: Mapping[str, object], field: str, line_number: int
) -> str:
    value = record.get(field)
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError(f"Text-to-SQL Eval line {line_number} requires {field}.")


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0

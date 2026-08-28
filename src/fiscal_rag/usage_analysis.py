"""Aggregation and review workflows over private usage records."""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fiscal_rag.performance import metric_summary
from fiscal_rag.usage import UsageReview


USAGE_REVIEW_SCHEMA_VERSION = "usage-review-v1"
USAGE_SUMMARY_SCHEMA_VERSION = "usage-summary-v2"
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_question(value: str) -> str:
    """Normalize insignificant whitespace without semantic clustering."""
    return WHITESPACE_PATTERN.sub(" ", value.strip())


def summarize_usage_records(
    records: Sequence[Mapping[str, object]],
    *,
    top_n: int = 10,
    retention_days: int = 90,
    slow_ms: float = 6000.0,
    queue_limit: int = 20,
    include_raw_questions: bool = False,
    started_from: str | None = None,
    started_to: str | None = None,
    generated_at: datetime | None = None,
) -> dict[str, object]:
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if retention_days <= 0:
        raise ValueError("retention_days must be positive")
    if slow_ms <= 0:
        raise ValueError("slow_ms must be positive")
    if queue_limit <= 0:
        raise ValueError("queue_limit must be positive")
    normalized_from, normalized_to = validate_usage_window(started_from, started_to)
    active_generated_at = generated_at or datetime.now(UTC)
    if active_generated_at.tzinfo is None:
        raise ValueError("generated_at must be timezone-aware")
    generated_at_utc = active_generated_at.astimezone(UTC)
    records = [
        record
        for record in records
        if record.get("traffic_kind", "production") == "production"
    ]
    assistant_records = [
        record for record in records if record.get("endpoint") == "assistant_stream"
    ]
    completed_assistant = [
        record
        for record in assistant_records
        if record.get("execution_status") == "completed"
    ]
    rated = [
        record for record in completed_assistant if record.get("feedback_rating")
    ]
    positive = sum(record.get("feedback_rating") == "positive" for record in rated)
    slow_records = [record for record in records if _is_slow(record, slow_ms)]
    actionable = [
        record
        for record in records
        if record.get("review_status") != "user_confirmed"
        and _review_signals(record, slow_ms=slow_ms)
    ]
    reviewed = [
        record for record in records if record.get("review_status") == "user_confirmed"
    ]
    eval_ready_records = [record for record in reviewed if record.get("eval_candidate")]
    question_groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        question = record.get("question")
        if isinstance(question, str) and question.strip():
            question_groups[normalize_question(question)].append(record)
    top_questions = sorted(
        (
            {
                "question": question,
                "count": len(group),
                "routes": dict(
                    sorted(Counter(str(item.get("route") or "unknown") for item in group).items())
                ),
            }
            for question, group in question_groups.items()
        ),
        key=lambda item: (-int(item["count"]), str(item["question"])),
    )[:top_n] if include_raw_questions else []

    versions: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        failure_type = record.get("failure_type")
        if isinstance(failure_type, str):
            versions[str(record.get("service_version") or "unknown")][failure_type] += 1

    expiration = _records_expiry(records, retention_days) or (
        generated_at_utc + timedelta(days=retention_days)
    ).isoformat()
    rag_records = [record for record in records if record.get("route") == "rag"]
    report = {
        "schema_version": USAGE_SUMMARY_SCHEMA_VERSION,
        "contains_raw_content": include_raw_questions,
        "generated_at_utc": generated_at_utc.isoformat(),
        "expires_at_utc": expiration,
        "window": {
            "started_from_utc": normalized_from,
            "started_to_utc": normalized_to,
            "default_scope": "all_retained" if not normalized_from and not normalized_to else "custom",
        },
        "overview": {
            "total_requests": len(records),
            "distinct_exact_questions": len(question_groups),
            "endpoints": _counts(records, "endpoint"),
            "assistant_routes": _counts(assistant_records, "route"),
            "execution_statuses": _counts(records, "execution_status"),
            "execution_completion_rate": _ratio(
                sum(record.get("execution_status") == "completed" for record in records),
                len(records),
            ),
            "errors": _counts(
                [record for record in records if record.get("error_code")], "error_code"
            ),
        },
        "feedback": {
            "eligible_completed_assistant": len(completed_assistant),
            "rated": len(rated),
            "feedback_rate": _ratio(len(rated), len(completed_assistant)),
            "positive": positive,
            "negative": len(rated) - positive,
            "positive_rate_among_rated": _ratio(positive, len(rated)),
        },
        "latency_ms": {
            "overall_total": metric_summary(_numeric_values(records, "total_duration_ms")),
            "by_route_server_total": {
                route: metric_summary(
                    _timing_values(
                        [record for record in assistant_records if record.get("route") == route],
                        "server_total",
                    )
                )
                for route in ("rag", "chat", "out_of_scope")
            },
            "slow_threshold_ms": slow_ms,
            "slow_requests": len(slow_records),
            "slow_request_rate": _ratio(len(slow_records), len(records)),
        },
        "daily_trend": _daily_trend(records, slow_ms=slow_ms),
        "rag_health": {
            "requests": len(rag_records),
            "with_trace": sum(bool(record.get("retrieval_query")) for record in rag_records),
            "with_sources": sum(bool(record.get("sources")) for record in rag_records),
            "rewrite_statuses": _counts(rag_records, "query_rewrite_status"),
            "with_missing_constraints": sum(
                bool(record.get("missing_constraints")) for record in rag_records
            ),
        },
        "review_funnel": {
            "actionable_requests": len(actionable),
            "action_signals": _signal_counts(actionable, slow_ms=slow_ms),
            "user_confirmed": len(reviewed),
            "failure_types": _counts(reviewed, "failure_type"),
            "severities": _counts(reviewed, "severity"),
            "eval_candidates": len(eval_ready_records),
            "failure_types_by_version": {
                version: dict(sorted(counts.items()))
                for version, counts in sorted(versions.items())
            },
        },
        "action_queue": [
            _action_queue_record(record, slow_ms=slow_ms)
            for record in sorted(
                actionable,
                key=lambda item: (
                    str(item.get("started_at_utc") or ""),
                    str(item.get("request_id") or ""),
                ),
                reverse=True,
            )[:queue_limit]
        ],
        "eval_ready": [
            {
                "request_id": record.get("request_id"),
                "candidate_case_id": record.get("candidate_case_id"),
                "candidate_category": record.get("candidate_category"),
            }
            for record in sorted(
                eval_ready_records,
                key=lambda item: str(item.get("request_id") or ""),
            )
        ],
        "top_exact_questions": top_questions,
    }
    return report


def validate_usage_window(
    started_from: str | None,
    started_to: str | None,
) -> tuple[str | None, str | None]:
    """Validate and normalize optional report boundaries to UTC ISO strings."""
    normalized_from = _normalize_utc_boundary(started_from, "started_from")
    normalized_to = _normalize_utc_boundary(started_to, "started_to")
    if normalized_from and normalized_to:
        if datetime.fromisoformat(normalized_from) >= datetime.fromisoformat(normalized_to):
            raise ValueError("started_from must be earlier than started_to")
    return normalized_from, normalized_to


def select_usage_review_templates(
    records: Sequence[Mapping[str, object]],
    *,
    request_ids: Sequence[str] = (),
    slow_ms: float = 6000.0,
    retention_days: int = 90,
) -> list[dict[str, object]]:
    """Build review templates and atomically validate an optional ID selection."""
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("request_id values must not repeat")
    templates = usage_review_template_records(
        records,
        slow_ms=slow_ms,
        retention_days=retention_days,
    )
    if not request_ids:
        return templates
    by_id = {str(record.get("request_id")): record for record in templates}
    unavailable = [request_id for request_id in request_ids if request_id not in by_id]
    if unavailable:
        raise ValueError(
            "Requested review IDs are unknown, already confirmed, or ineligible: "
            + ", ".join(unavailable)
        )
    return [by_id[request_id] for request_id in request_ids]


def render_usage_summary_markdown(summary: Mapping[str, object]) -> str:
    """Render the stable, privacy-filtered summary as a local Markdown report."""
    overview = _mapping(summary.get("overview"))
    feedback = _mapping(summary.get("feedback"))
    latency = _mapping(summary.get("latency_ms"))
    rag_health = _mapping(summary.get("rag_health"))
    review = _mapping(summary.get("review_funnel"))
    window = _mapping(summary.get("window"))
    lines = [
        "# Usage Analytics Report",
        "",
        f"- Generated at (UTC): `{summary.get('generated_at_utc')}`",
        f"- Expires at (UTC): `{summary.get('expires_at_utc')}`",
        f"- Window start: `{window.get('started_from_utc') or 'all retained'}`",
        f"- Window end: `{window.get('started_to_utc') or 'now'}`",
        f"- Contains raw content: `{str(bool(summary.get('contains_raw_content'))).lower()}`",
        "",
        "## Overview",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Production requests | {overview.get('total_requests', 0)} |",
        f"| Completion rate | {_format_ratio(overview.get('execution_completion_rate'))} |",
        f"| Distinct exact questions | {overview.get('distinct_exact_questions', 0)} |",
        "",
        "### Assistant Routes",
        "",
        _markdown_counts(_mapping(overview.get("assistant_routes"))),
        "",
        "### Endpoints",
        "",
        _markdown_counts(_mapping(overview.get("endpoints"))),
        "",
        "### Execution Statuses",
        "",
        _markdown_counts(_mapping(overview.get("execution_statuses"))),
        "",
        "### Error Codes",
        "",
        _markdown_counts(_mapping(overview.get("errors"))),
        "",
        "## Latency And Feedback",
        "",
        f"- Slow threshold: `{latency.get('slow_threshold_ms', 0)} ms`",
        f"- Slow requests: `{latency.get('slow_requests', 0)}` ({_format_ratio(latency.get('slow_request_rate'))})",
        f"- Feedback rate: `{_format_ratio(feedback.get('feedback_rate'))}`",
        f"- Positive / negative: `{feedback.get('positive', 0)} / {feedback.get('negative', 0)}`",
        "",
        "### Overall Total Latency",
        "",
        _markdown_metric(_mapping(latency.get("overall_total"))),
        "",
        "### Server Total By Assistant Route",
        "",
        _markdown_route_latency(_mapping(latency.get("by_route_server_total"))),
        "",
        "## RAG Health",
        "",
        f"- RAG requests: `{rag_health.get('requests', 0)}`",
        f"- With trace: `{rag_health.get('with_trace', 0)}`",
        f"- With sources: `{rag_health.get('with_sources', 0)}`",
        f"- With missing constraints: `{rag_health.get('with_missing_constraints', 0)}`",
        "",
        "### Rewrite Statuses",
        "",
        _markdown_counts(_mapping(rag_health.get("rewrite_statuses"))),
        "",
        "## Review Funnel",
        "",
        f"- Actionable requests: `{review.get('actionable_requests', 0)}`",
        f"- User confirmed: `{review.get('user_confirmed', 0)}`",
        f"- Eval candidates: `{review.get('eval_candidates', 0)}`",
        "",
        "### Action Signals",
        "",
        _markdown_counts(_mapping(review.get("action_signals"))),
        "",
        "### Confirmed Failure Types",
        "",
        _markdown_counts(_mapping(review.get("failure_types"))),
        "",
        "### Confirmed Severities",
        "",
        _markdown_counts(_mapping(review.get("severities"))),
        "",
        "### Failure Types By Version",
        "",
        _markdown_failure_types_by_version(review.get("failure_types_by_version")),
        "",
        "## Action Queue",
        "",
        _markdown_action_queue(summary.get("action_queue")),
        "",
        "## Eval Ready",
        "",
        _markdown_eval_ready(summary.get("eval_ready")),
        "",
        "## Daily Trend",
        "",
        _markdown_daily_trend(summary.get("daily_trend")),
    ]
    questions = summary.get("top_exact_questions")
    if isinstance(questions, Sequence) and not isinstance(questions, (str, bytes)) and questions:
        lines.extend(["", "## Top Exact Questions", "", _markdown_questions(questions)])
    return "\n".join(lines).rstrip() + "\n"


def usage_review_template_records(
    records: Sequence[Mapping[str, object]],
    *,
    slow_ms: float = 6000.0,
    retention_days: int = 90,
) -> list[dict[str, object]]:
    if slow_ms <= 0:
        raise ValueError("slow_ms must be positive")
    if retention_days <= 0:
        raise ValueError("retention_days must be positive")
    templates = []
    for record in records:
        if record.get("traffic_kind", "production") != "production":
            continue
        if record.get("review_status") == "user_confirmed":
            continue
        signals = _review_signals(record, slow_ms=slow_ms)
        if not signals:
            continue
        templates.append(
            {
                "schema_version": USAGE_REVIEW_SCHEMA_VERSION,
                "contains_raw_content": True,
                "expires_at_utc": _expires_at(record.get("started_at_utc"), retention_days),
                "request_id": record.get("request_id"),
                "service_version": record.get("service_version"),
                "profile_id": record.get("profile_id"),
                "started_at_utc": record.get("started_at_utc"),
                "endpoint": record.get("endpoint"),
                "question": record.get("question"),
                "actual_route": record.get("route"),
                "execution_status": record.get("execution_status"),
                "answer": record.get("answer"),
                "trace": {
                    "retrieval_query": record.get("retrieval_query"),
                    "rewrite_query": record.get("rewrite_query"),
                    "query_rewrite_status": record.get("query_rewrite_status"),
                    "required_constraints": record.get("required_constraints", []),
                    "missing_constraints": record.get("missing_constraints", []),
                    "sources": record.get("sources", []),
                },
                "timings_ms": record.get("timings_ms", {}),
                "total_duration_ms": record.get("total_duration_ms"),
                "error_code": record.get("error_code"),
                "feedback_rating": record.get("feedback_rating"),
                "review_signals": signals,
                "review_status": "pending",
                "failure_type": "other",
                "severity": "none",
                "expected_route": record.get("route"),
                "answerability": None,
                "reason": "Pending human review.",
                "topic": None,
                "eval_candidate": False,
                "candidate_case_id": None,
                "candidate_category": None,
                "retrieval_case_id": None,
                "expected_answer": None,
                "relevant_evidence": [],
            }
        )
    return templates


def load_usage_reviews(path: str | Path) -> list[UsageReview]:
    reviews: list[UsageReview] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"Usage review line {line_number} must be an object.")
        if record.get("schema_version") != USAGE_REVIEW_SCHEMA_VERSION:
            raise ValueError(f"Usage review line {line_number} has an invalid schema.")
        request_id = _required_string(record, "request_id", line_number)
        if request_id in seen:
            raise ValueError(f"Usage reviews repeat request_id: {request_id!r}")
        seen.add(request_id)
        if record.get("review_status") != "user_confirmed":
            raise ValueError(
                f"Usage review line {line_number} must be user_confirmed before import."
            )
        answerability = record.get("answerability")
        if answerability is not None and not isinstance(answerability, bool):
            raise ValueError(f"Usage review line {line_number} has invalid answerability.")
        evidence = record.get("relevant_evidence")
        if not isinstance(evidence, list):
            raise ValueError(f"Usage review line {line_number} requires relevant_evidence.")
        reviews.append(
            UsageReview(
                request_id=request_id,
                review_status="user_confirmed",
                failure_type=_required_string(record, "failure_type", line_number),
                severity=_required_string(record, "severity", line_number),
                expected_route=_optional_string(record.get("expected_route")),
                answerability=answerability,
                reason=_required_string(record, "reason", line_number),
                eval_candidate=_required_bool(record, "eval_candidate", line_number),
                candidate_case_id=_optional_string(record.get("candidate_case_id")),
                candidate_category=_optional_string(record.get("candidate_category")),
                retrieval_case_id=_optional_string(record.get("retrieval_case_id")),
                expected_answer=_optional_string(record.get("expected_answer")),
                evidence_json=json.dumps(evidence, ensure_ascii=False),
                topic=_optional_string(record.get("topic")),
            )
        )
    if not reviews:
        raise ValueError("Usage review JSONL contains no records.")
    return reviews


def eval_candidate_records(
    records: Sequence[Mapping[str, object]],
    *,
    existing_questions: Sequence[str] = (),
    existing_case_ids: Sequence[str] = (),
) -> list[dict[str, object]]:
    known_questions = {normalize_question(question) for question in existing_questions}
    known_case_ids = set(existing_case_ids)
    candidates = []
    for record in records:
        if record.get("traffic_kind", "production") != "production":
            continue
        if record.get("review_status") != "user_confirmed" or not record.get("eval_candidate"):
            continue
        case_id = _mapping_string(record, "candidate_case_id")
        question = _mapping_string(record, "question")
        category = _mapping_string(record, "candidate_category")
        expected_route = _mapping_string(record, "expected_route")
        normalized_question = normalize_question(question)
        if case_id in known_case_ids:
            raise ValueError(f"Eval candidate repeats case_id: {case_id!r}")
        if normalized_question in known_questions:
            raise ValueError(f"Eval candidate repeats an existing question: {question!r}")
        known_case_ids.add(case_id)
        known_questions.add(normalized_question)
        answerability_value = record.get("answerability")
        answerability = (
            bool(answerability_value) if answerability_value is not None else None
        )
        retrieval_case_id = _optional_string(record.get("retrieval_case_id"))
        if category == "rag_answerable" and not retrieval_case_id:
            raise ValueError(
                f"Answerable candidate {case_id!r} requires retrieval_case_id."
            )
        candidates.append(
            {
                "schema_version": "assistant-eval-v1",
                "case_id": case_id,
                "category": category,
                "question": question,
                "expected_route": expected_route,
                "answerability": answerability,
                "retrieval_case_id": retrieval_case_id,
                "expected_answer": record.get("expected_answer"),
                "relevant_evidence": _evidence_from_record(record),
                "source_request_id": record.get("request_id"),
            }
        )
    return candidates


def load_existing_eval_identity(
    paths: Sequence[str | Path],
) -> tuple[list[str], list[str]]:
    questions: list[str] = []
    case_ids: list[str] = []
    for path in paths:
        for line_number, line in enumerate(
            Path(path).read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Eval line {line_number} must be an object.")
            questions.append(_required_string(record, "question", line_number))
            case_ids.append(_required_string(record, "case_id", line_number))
    return questions, case_ids


def prune_expired_usage_artifacts(
    usage_directory: str | Path,
    retention_days: int,
    *,
    now: datetime | None = None,
) -> list[Path]:
    """Delete expired usage artifacts, including paired summary reports."""
    if retention_days <= 0:
        raise ValueError("retention_days must be positive")
    active_now = now or datetime.now(UTC)
    if active_now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    root = Path(usage_directory)
    candidates = [
        *sorted((root / "reports").glob("usage_summary_*.json")),
        *sorted((root / "text_to_sql").glob("query_*.json")),
        *sorted((root / "reviews").glob("usage_review_*.jsonl")),
        *sorted((root / "backups").glob("usage_????????T??????Z.sqlite3")),
    ]
    deleted = []
    for path in candidates:
        if path.is_symlink() or not path.is_file():
            continue
        expires_at = _artifact_expiry(path, retention_days)
        if expires_at <= active_now.astimezone(UTC):
            path.unlink()
            deleted.append(path)
            if path.parent.name in {"reports", "text_to_sql"} and path.suffix == ".json":
                markdown = path.with_suffix(".md")
                if not markdown.is_symlink() and markdown.is_file():
                    markdown.unlink()
                    deleted.append(markdown)
    return deleted


def _daily_trend(
    records: Sequence[Mapping[str, object]], *, slow_ms: float
) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        started = _parse_datetime(record.get("started_at_utc"))
        if started is not None:
            grouped[started.date().isoformat()].append(record)
    trend = []
    for day, group in sorted(grouped.items()):
        assistant = [
            record for record in group if record.get("endpoint") == "assistant_stream"
        ]
        trend.append(
            {
                "date_utc": day,
                "requests": len(group),
                "assistant_routes": _counts(assistant, "route"),
                "execution_statuses": _counts(group, "execution_status"),
                "negative_feedback": sum(
                    record.get("feedback_rating") == "negative" for record in assistant
                ),
                "slow_requests": sum(_is_slow(record, slow_ms) for record in group),
            }
        )
    return trend


def _signal_counts(
    records: Sequence[Mapping[str, object]], *, slow_ms: float
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(_review_signals(record, slow_ms=slow_ms))
    return dict(sorted(counts.items()))


def _action_queue_record(
    record: Mapping[str, object], *, slow_ms: float
) -> dict[str, object]:
    return {
        "request_id": record.get("request_id"),
        "started_at_utc": record.get("started_at_utc"),
        "endpoint": record.get("endpoint"),
        "route": record.get("route"),
        "execution_status": record.get("execution_status"),
        "total_duration_ms": record.get("total_duration_ms"),
        "error_code": record.get("error_code"),
        "error_type": record.get("error_type"),
        "failure_stage": record.get("failure_stage"),
        "feedback_rating": record.get("feedback_rating"),
        "review_signals": _review_signals(record, slow_ms=slow_ms),
        "review_status": record.get("review_status"),
        "eval_candidate": bool(record.get("eval_candidate")),
    }


def _review_signals(record: Mapping[str, object], *, slow_ms: float) -> list[str]:
    signals = []
    if record.get("feedback_rating") == "negative":
        signals.append("negative_feedback")
    if record.get("execution_status") != "completed":
        signals.append("execution_not_completed")
    total = record.get("total_duration_ms")
    if isinstance(total, (int, float)) and not isinstance(total, bool) and total >= slow_ms:
        signals.append("slow_request")
    return signals


def _is_slow(record: Mapping[str, object], slow_ms: float) -> bool:
    total = record.get("total_duration_ms")
    return (
        isinstance(total, (int, float))
        and not isinstance(total, bool)
        and total >= slow_ms
    )


def _normalize_utc_boundary(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    parsed = _parse_datetime(value)
    if parsed is None or datetime.fromisoformat(value).tzinfo is None:
        raise ValueError(f"{name} must be a timezone-aware ISO 8601 timestamp")
    return parsed.isoformat()


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _format_ratio(value: object) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value) * 100:.1f}%"
    return "0.0%"


def _markdown_counts(values: Mapping[str, object]) -> str:
    if not values:
        return "No data."
    lines = ["| Value | Requests |", "|---|---:|"]
    lines.extend(
        f"| {_escape_markdown(key)} | {value} |" for key, value in values.items()
    )
    return "\n".join(lines)


def _markdown_metric(values: Mapping[str, object]) -> str:
    if not values or not values.get("count"):
        return "No measured requests."
    return "\n".join(
        [
            "| Count | Min | p50 | p95 | Max |",
            "|---:|---:|---:|---:|---:|",
            "| {count} | {min} | {p50} | {p95} | {max} |".format(**values),
        ]
    )


def _markdown_action_queue(value: object) -> str:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        return "No actionable requests."
    lines = [
        "| Request ID | Started UTC | Endpoint | Route | Status | Total ms | Error | Failure stage | Feedback | Signals | Review | Eval candidate |",
        "|---|---|---|---|---|---:|---|---|---|---|---|---|",
    ]
    for item in value:
        record = _mapping(item)
        signals = record.get("review_signals")
        signal_text = ", ".join(map(str, signals)) if isinstance(signals, list) else ""
        lines.append(
            "| {request_id} | {started} | {endpoint} | {route} | {status} | {duration} | {error} | {failure_stage} | {feedback} | {signals} | {review} | {eval_candidate} |".format(
                request_id=_escape_markdown(record.get("request_id")),
                started=_escape_markdown(record.get("started_at_utc")),
                endpoint=_escape_markdown(record.get("endpoint")),
                route=_escape_markdown(record.get("route")),
                status=_escape_markdown(record.get("execution_status")),
                duration=_escape_markdown(record.get("total_duration_ms")),
                error=_escape_markdown(
                    _error_label(record.get("error_code"), record.get("error_type"))
                ),
                failure_stage=_escape_markdown(record.get("failure_stage")),
                feedback=_escape_markdown(record.get("feedback_rating")),
                signals=_escape_markdown(signal_text),
                review=_escape_markdown(record.get("review_status")),
                eval_candidate=str(bool(record.get("eval_candidate"))).lower(),
            )
        )
    return "\n".join(lines)


def _markdown_daily_trend(value: object) -> str:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        return "No production activity."
    lines = [
        "| Date UTC | Requests | Routes | Statuses | Negative | Slow |",
        "|---|---:|---|---|---:|---:|",
    ]
    for item in value:
        record = _mapping(item)
        routes = _inline_counts(_mapping(record.get("assistant_routes")))
        statuses = _mapping(record.get("execution_statuses"))
        lines.append(
            f"| {_escape_markdown(record.get('date_utc'))} | {record.get('requests', 0)} "
            f"| {_escape_markdown(routes)} | {_escape_markdown(_inline_counts(statuses))} "
            f"| {record.get('negative_feedback', 0)} | {record.get('slow_requests', 0)} |"
        )
    return "\n".join(lines)


def _markdown_questions(value: Sequence[object]) -> str:
    lines = ["| Question | Count | Routes |", "|---|---:|---|"]
    for item in value:
        record = _mapping(item)
        routes = _mapping(record.get("routes"))
        route_text = ", ".join(f"{key}: {count}" for key, count in routes.items())
        lines.append(
            f"| {_escape_markdown(record.get('question'))} | {record.get('count', 0)} "
            f"| {_escape_markdown(route_text)} |"
        )
    return "\n".join(lines)


def _markdown_route_latency(value: Mapping[str, object]) -> str:
    lines = [
        "| Route | Count | Min | p50 | p95 | Max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    measured = False
    for route, raw_metric in value.items():
        metric = _mapping(raw_metric)
        if metric.get("count"):
            measured = True
        lines.append(
            "| {route} | {count} | {min} | {p50} | {p95} | {max} |".format(
                route=_escape_markdown(route),
                count=metric.get("count", 0),
                min=metric.get("min", "-"),
                p50=metric.get("p50", "-"),
                p95=metric.get("p95", "-"),
                max=metric.get("max", "-"),
            )
        )
    return "\n".join(lines) if measured else "No measured requests."


def _markdown_failure_types_by_version(value: object) -> str:
    versions = _mapping(value)
    if not versions:
        return "No confirmed failures."
    lines = ["| Service version | Failure types |", "|---|---|"]
    for version, raw_counts in versions.items():
        lines.append(
            f"| {_escape_markdown(version)} | "
            f"{_escape_markdown(_inline_counts(_mapping(raw_counts)))} |"
        )
    return "\n".join(lines)


def _markdown_eval_ready(value: object) -> str:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        return "No Eval-ready requests."
    lines = [
        "| Request ID | Candidate case ID | Category |",
        "|---|---|---|",
    ]
    for item in value:
        record = _mapping(item)
        lines.append(
            f"| {_escape_markdown(record.get('request_id'))} "
            f"| {_escape_markdown(record.get('candidate_case_id'))} "
            f"| {_escape_markdown(record.get('candidate_category'))} |"
        )
    return "\n".join(lines)


def _inline_counts(values: Mapping[str, object]) -> str:
    return ", ".join(f"{key}: {value}" for key, value in values.items()) or "-"


def _error_label(code: object, error_type: object) -> str:
    values = [str(value) for value in (code, error_type) if value]
    return " / ".join(values) or "-"


def _escape_markdown(value: object) -> str:
    if value is None:
        return "-"
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _expires_at(value: object, retention_days: int) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        started = datetime.fromisoformat(value)
    except ValueError:
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    return (started.astimezone(UTC) + timedelta(days=retention_days)).isoformat()


def _records_expiry(
    records: Sequence[Mapping[str, object]], retention_days: int
) -> str | None:
    expirations = [
        expiration
        for record in records
        if (expiration := _expires_at(record.get("started_at_utc"), retention_days))
    ]
    return min(expirations) if expirations else None


def _artifact_expiry(path: Path, retention_days: int) -> datetime:
    expirations: list[datetime] = []
    if path.suffix == ".json":
        record = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(record, Mapping):
            expiration = _parse_datetime(record.get("expires_at_utc"))
            if expiration is not None:
                expirations.append(expiration)
    elif path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if isinstance(record, Mapping):
                expiration = _parse_datetime(record.get("expires_at_utc"))
                if expiration is not None:
                    expirations.append(expiration)
    elif path.suffix == ".sqlite3":
        connection = sqlite3.connect(path)
        try:
            row = connection.execute(
                "SELECT MIN(started_at_utc) FROM assistant_requests"
            ).fetchone()
        finally:
            connection.close()
        if row and row[0] is not None:
            expiration = _parse_datetime(_expires_at(row[0], retention_days))
            if expiration is not None:
                expirations.append(expiration)
    if expirations:
        return min(expirations)
    modified = datetime.fromtimestamp(path.stat().st_mtime, UTC)
    return modified + timedelta(days=retention_days)


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _counts(records: Sequence[Mapping[str, object]], field: str) -> dict[str, int]:
    return dict(
        sorted(Counter(str(record.get(field) or "unknown") for record in records).items())
    )


def _numeric_values(
    records: Sequence[Mapping[str, object]], field: str
) -> list[float]:
    return [
        float(value)
        for record in records
        if isinstance((value := record.get(field)), (int, float))
        and not isinstance(value, bool)
    ]


def _timing_values(
    records: Sequence[Mapping[str, object]], stage: str
) -> list[float]:
    values = []
    for record in records:
        timings = record.get("timings_ms")
        if not isinstance(timings, Mapping):
            continue
        value = timings.get(stage)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return values


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _required_string(
    record: Mapping[str, object], field: str, line_number: int
) -> str:
    value = record.get(field)
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError(f"Usage review line {line_number} requires {field}.")


def _mapping_string(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError(f"Eval candidate requires {field}.")


def _required_bool(
    record: Mapping[str, object], field: str, line_number: int
) -> bool:
    value = record.get(field)
    if isinstance(value, bool):
        return value
    raise ValueError(f"Usage review line {line_number} requires boolean {field}.")


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _evidence_from_record(record: Mapping[str, object]) -> list[object]:
    value = record.get("evidence_json")
    if not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []

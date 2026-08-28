import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fiscal_rag.usage import SQLiteUsageRepository  # noqa: E402
from fiscal_rag.usage_analysis import (  # noqa: E402
    eval_candidate_records,
    load_usage_reviews,
    normalize_question,
    prune_expired_usage_artifacts,
    render_usage_summary_markdown,
    select_usage_review_templates,
    summarize_usage_records,
    usage_review_template_records,
    validate_usage_window,
)


def usage_record(
    request_id: str,
    *,
    question: str,
    route: str = "rag",
    status: str = "completed",
    feedback: str | None = None,
    total_ms: float = 1000.0,
    traffic_kind: str = "production",
) -> dict[str, object]:
    return {
        "request_id": request_id,
        "endpoint": "assistant_stream",
        "started_at_utc": "2026-08-27T00:00:00+00:00",
        "service_version": "1.2.0",
        "profile_id": "profile",
        "traffic_kind": traffic_kind,
        "question": question,
        "answer": "answer",
        "route": route,
        "execution_status": status,
        "feedback_rating": feedback,
        "total_duration_ms": total_ms,
        "timings_ms": {"server_total": total_ms - 10},
        "sources": [],
        "required_constraints": [],
        "missing_constraints": [],
    }


def test_usage_summary_keeps_feedback_denominators_and_exact_question_groups() -> None:
    records = [
        usage_record("1", question="  怎样   保存？ ", feedback="positive"),
        usage_record("2", question="怎样 保存？", feedback=None),
        usage_record("3", question="你好", route="chat", feedback="negative"),
        usage_record("4", question="失败", status="failed"),
        {
            **usage_record("5", question="兼容接口"),
            "endpoint": "ask",
            "feedback_rating": None,
        },
    ]

    summary = summarize_usage_records(
        records,
        include_raw_questions=True,
        generated_at=datetime(2026, 8, 27, tzinfo=UTC),
    )

    assert normalize_question("  怎样   保存？ ") == "怎样 保存？"
    assert summary["schema_version"] == "usage-summary-v2"
    assert summary["overview"]["total_requests"] == 5
    assert summary["expires_at_utc"] == "2026-11-25T00:00:00+00:00"
    assert summary["overview"]["assistant_routes"] == {"chat": 1, "rag": 3}
    assert summary["feedback"] == {
        "eligible_completed_assistant": 3,
        "rated": 2,
        "feedback_rate": 2 / 3,
        "positive": 1,
        "negative": 1,
        "positive_rate_among_rated": 0.5,
    }
    assert summary["top_exact_questions"][0]["question"] == "怎样 保存？"
    assert summary["top_exact_questions"][0]["count"] == 2


def test_usage_summary_and_review_exclude_eval_traffic() -> None:
    production = usage_record(
        "production", question="real question", feedback="negative"
    )
    assistant_eval = usage_record(
        "assistant-eval",
        question="eval question",
        feedback="negative",
        traffic_kind="assistant_eval",
    )
    performance_eval = usage_record(
        "performance-eval",
        question="performance question",
        status="failed",
        traffic_kind="performance_eval",
    )

    summary = summarize_usage_records([production, assistant_eval, performance_eval])
    templates = usage_review_template_records(
        [production, assistant_eval, performance_eval]
    )

    assert summary["overview"]["total_requests"] == 1
    assert summary["feedback"]["negative"] == 1
    assert summary["top_exact_questions"] == []
    assert summary["contains_raw_content"] is False
    assert [record["request_id"] for record in templates] == ["production"]


def test_usage_summary_builds_daily_health_queue_and_review_funnel() -> None:
    records = [
        {
            **usage_record(
                "failed",
                question="private failed question",
                status="failed",
                total_ms=6000,
            ),
            "started_at_utc": "2026-08-26T23:30:00-01:00",
            "error_code": "model_request_failed",
            "error_type": "RuntimeError",
            "failure_stage": "generation",
            "retrieval_query": "private retrieval query",
            "query_rewrite_status": "fallback_to_original",
            "missing_constraints": ["private constraint"],
            "sources": [{"source": "private.md"}],
        },
        {
            **usage_record(
                "reviewed",
                question="private reviewed question",
                feedback="negative",
            ),
            "started_at_utc": "2026-08-27T02:00:00+00:00",
            "review_status": "user_confirmed",
            "failure_type": "retrieval",
            "severity": "major",
            "eval_candidate": 1,
            "candidate_case_id": "usage-001",
            "candidate_category": "rag_answerable",
        },
        {
            **usage_record("aborted", question="aborted", status="aborted"),
            "started_at_utc": "2026-08-28T00:00:00+00:00",
        },
        usage_record("interrupted", question="interrupted", status="interrupted"),
        usage_record("completed", question="completed", route="out_of_scope"),
    ]

    summary = summarize_usage_records(
        records,
        slow_ms=6000,
        queue_limit=2,
        generated_at=datetime(2026, 8, 29, tzinfo=UTC),
    )

    assert summary["overview"]["execution_statuses"] == {
        "aborted": 1,
        "completed": 2,
        "failed": 1,
        "interrupted": 1,
    }
    assert summary["latency_ms"]["slow_requests"] == 1
    assert summary["daily_trend"][0]["date_utc"] == "2026-08-27"
    assert summary["daily_trend"][0]["requests"] == 4
    assert summary["rag_health"] == {
        "requests": 4,
        "with_trace": 1,
        "with_sources": 1,
        "rewrite_statuses": {"fallback_to_original": 1, "unknown": 3},
        "with_missing_constraints": 1,
    }
    assert summary["review_funnel"]["actionable_requests"] == 3
    assert summary["review_funnel"]["user_confirmed"] == 1
    assert summary["review_funnel"]["eval_candidates"] == 1
    assert [item["request_id"] for item in summary["action_queue"]] == [
        "aborted",
        "interrupted",
    ]
    assert summary["eval_ready"] == [
        {
            "request_id": "reviewed",
            "candidate_case_id": "usage-001",
            "candidate_category": "rag_answerable",
        }
    ]
    markdown = render_usage_summary_markdown(summary)
    for expected in (
        "model_request_failed",
        "fallback_to_original",
        "retrieval",
        "major",
        "usage-001",
    ):
        assert expected in markdown


def test_usage_summary_and_markdown_are_private_by_default() -> None:
    record = {
        **usage_record(
            "negative",
            question="SECRET QUESTION",
            feedback="negative",
        ),
        "answer": "SECRET ANSWER",
        "retrieval_query": "SECRET QUERY",
        "sources": [{"source": "SECRET SOURCE"}],
        "reason": "SECRET REVIEW REASON",
    }

    summary = summarize_usage_records(
        [record],
        generated_at=datetime(2026, 8, 27, tzinfo=UTC),
    )
    markdown = render_usage_summary_markdown(summary)
    serialized = json.dumps(summary)

    for secret in (
        "SECRET QUESTION",
        "SECRET ANSWER",
        "SECRET QUERY",
        "SECRET SOURCE",
        "SECRET REVIEW REASON",
    ):
        assert secret not in serialized
        assert secret not in markdown
    assert summary["contains_raw_content"] is False
    assert "negative" in markdown

    raw_summary = summarize_usage_records(
        [record],
        include_raw_questions=True,
        generated_at=datetime(2026, 8, 27, tzinfo=UTC),
    )
    raw_markdown = render_usage_summary_markdown(raw_summary)
    assert raw_summary["contains_raw_content"] is True
    assert "SECRET QUESTION" in raw_markdown
    assert "SECRET ANSWER" not in raw_markdown


def test_usage_summary_supports_empty_database_and_validated_utc_window() -> None:
    generated = datetime(2026, 8, 27, tzinfo=UTC)
    summary = summarize_usage_records([], generated_at=generated)

    assert summary["overview"]["total_requests"] == 0
    assert summary["daily_trend"] == []
    assert summary["action_queue"] == []
    assert summary["expires_at_utc"] == "2026-11-25T00:00:00+00:00"
    assert "No production activity." in render_usage_summary_markdown(summary)

    raw_empty = summarize_usage_records(
        [], generated_at=generated, include_raw_questions=True
    )
    assert raw_empty["contains_raw_content"] is True
    assert raw_empty["top_exact_questions"] == []
    assert validate_usage_window(
        "2026-08-27T08:00:00+08:00",
        "2026-08-28T08:00:00+08:00",
    ) == ("2026-08-27T00:00:00+00:00", "2026-08-28T00:00:00+00:00")
    with pytest.raises(ValueError, match="timezone-aware"):
        validate_usage_window("2026-08-27T00:00:00", None)
    with pytest.raises(ValueError, match="earlier"):
        validate_usage_window(
            "2026-08-28T00:00:00+00:00",
            "2026-08-27T00:00:00+00:00",
        )


def test_review_template_selection_is_ordered_and_atomic() -> None:
    records = [
        usage_record("negative", question="negative", feedback="negative"),
        usage_record("failed", question="failed", status="failed"),
        usage_record("ok", question="ok"),
    ]

    selected = select_usage_review_templates(
        records,
        request_ids=["failed", "negative"],
    )

    assert [record["request_id"] for record in selected] == ["failed", "negative"]
    with pytest.raises(ValueError, match="unknown, already confirmed, or ineligible"):
        select_usage_review_templates(records, request_ids=["failed", "ok"])
    with pytest.raises(ValueError, match="must not repeat"):
        select_usage_review_templates(records, request_ids=["failed", "failed"])


def test_eval_candidate_export_excludes_nonproduction_traffic() -> None:
    record = {
        **usage_record(
            "assistant-eval",
            question="evaluation question",
            traffic_kind="assistant_eval",
        ),
        "review_status": "user_confirmed",
        "eval_candidate": 1,
        "candidate_case_id": "candidate-1",
        "candidate_category": "chat",
        "expected_route": "chat",
        "answerability": None,
    }

    assert eval_candidate_records([record]) == []


def test_usage_artifact_prune_uses_embedded_raw_content_expiry(
    tmp_path: Path,
) -> None:
    usage_directory = tmp_path / "usage"
    reports = usage_directory / "reports"
    reviews = usage_directory / "reviews"
    backups = usage_directory / "backups"
    reports.mkdir(parents=True)
    text_to_sql = usage_directory / "text_to_sql"
    text_to_sql.mkdir()
    reviews.mkdir()
    backups.mkdir()
    expired_at = "2026-08-01T00:00:00+00:00"
    report = reports / "usage_summary_20260801T000000Z.json"
    report.write_text(
        json.dumps({"expires_at_utc": expired_at}), encoding="utf-8"
    )
    markdown = report.with_suffix(".md")
    markdown.write_text("private report", encoding="utf-8")
    query = text_to_sql / "query_20260801T000000Z.json"
    query.write_text(
        json.dumps({"expires_at_utc": expired_at}), encoding="utf-8"
    )
    query_markdown = query.with_suffix(".md")
    query_markdown.write_text("private query", encoding="utf-8")
    review = reviews / "usage_review_20260801T000000Z.jsonl"
    review.write_text(
        json.dumps({"expires_at_utc": expired_at}) + "\n", encoding="utf-8"
    )
    backup = backups / "usage_20260801T000000Z.sqlite3"
    repository = SQLiteUsageRepository(backup)
    repository.initialize()
    repository.create_request(
        "old",
        endpoint="assistant_stream",
        question="old question",
        service_version="1.2.0",
        profile_id="profile",
    )
    connection = sqlite3.connect(backup)
    try:
        connection.execute(
            "UPDATE assistant_requests SET started_at_utc = ? WHERE request_id = 'old'",
            ("2026-05-01T00:00:00+00:00",),
        )
        connection.commit()
    finally:
        connection.close()
    release_evidence = backups / "usage_v1_2_release_20260801T000000Z.sqlite3"
    release_evidence.write_bytes(b"preserved")

    deleted = prune_expired_usage_artifacts(
        usage_directory,
        90,
        now=datetime(2026, 8, 27, tzinfo=UTC),
    )

    assert set(deleted) == {
        report,
        markdown,
        query,
        query_markdown,
        review,
        backup,
    }
    assert release_evidence.exists()


def test_review_templates_select_negative_failed_and_slow_requests() -> None:
    records = [
        usage_record("ok", question="ok"),
        usage_record("negative", question="negative", feedback="negative"),
        usage_record("failed", question="failed", status="failed"),
        usage_record("slow", question="slow", total_ms=7000),
    ]

    templates = usage_review_template_records(records, slow_ms=6000)

    assert [record["request_id"] for record in templates] == [
        "negative",
        "failed",
        "slow",
    ]
    assert templates[0]["review_signals"] == ["negative_feedback"]
    assert templates[1]["review_signals"] == ["execution_not_completed"]
    assert templates[2]["review_signals"] == ["slow_request"]
    assert templates[0]["contains_raw_content"] is True
    assert templates[0]["review_status"] == "pending"


def test_review_to_eval_candidate_flow_preserves_frozen_dataset(tmp_path: Path) -> None:
    database = tmp_path / "usage.sqlite3"
    repository = SQLiteUsageRepository(database)
    repository.initialize()
    repository.create_request(
        "request-1",
        endpoint="assistant_stream",
        question="怎样保存？",
        service_version="1.2.0",
        profile_id="profile",
    )
    repository.finalize_request(
        "request-1",
        execution_status="completed",
        route="rag",
        answer="遗漏了一步。",
        total_duration_ms=1000,
    )
    repository.set_feedback("request-1", "negative")
    template = usage_review_template_records(repository.fetch_usage_records())[0]
    template.update(
        {
            "review_status": "user_confirmed",
            "failure_type": "generation_correctness",
            "severity": "major",
            "expected_route": "rag",
            "answerability": True,
            "reason": "The answer omitted a required step.",
            "eval_candidate": True,
            "candidate_case_id": "usage-assistant-001",
            "candidate_category": "rag_answerable",
            "retrieval_case_id": "usage-retrieval-001",
            "expected_answer": "Complete both documented steps.",
            "relevant_evidence": [
                {"source": "guide.md", "evidence_text": "先选择，再保存。"}
            ],
        }
    )
    review_file = tmp_path / "review.jsonl"
    review_file.write_text(
        json.dumps(template, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for review in load_usage_reviews(review_file):
        repository.save_review(review)

    candidates = eval_candidate_records(repository.fetch_usage_records())

    assert candidates == [
        {
            "schema_version": "assistant-eval-v1",
            "case_id": "usage-assistant-001",
            "category": "rag_answerable",
            "question": "怎样保存？",
            "expected_route": "rag",
            "answerability": True,
            "retrieval_case_id": "usage-retrieval-001",
            "expected_answer": "Complete both documented steps.",
            "relevant_evidence": [
                {"source": "guide.md", "evidence_text": "先选择，再保存。"}
            ],
            "source_request_id": "request-1",
        }
    ]
    with pytest.raises(ValueError, match="existing question"):
        eval_candidate_records(
            repository.fetch_usage_records(), existing_questions=["怎样保存？"]
        )


def test_review_import_rejects_pending_or_duplicate_records(tmp_path: Path) -> None:
    template = usage_review_template_records(
        [usage_record("request-1", question="bad", feedback="negative")]
    )[0]
    path = tmp_path / "review.jsonl"
    path.write_text(json.dumps(template) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="user_confirmed"):
        load_usage_reviews(path)

    template["review_status"] = "user_confirmed"
    path.write_text(
        json.dumps(template) + "\n" + json.dumps(template) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="repeat request_id"):
        load_usage_reviews(path)


def test_answerable_eval_candidate_requires_strict_evidence(tmp_path: Path) -> None:
    database = tmp_path / "usage.sqlite3"
    repository = SQLiteUsageRepository(database)
    repository.initialize()
    repository.create_request(
        "request-1",
        endpoint="assistant_stream",
        question="question",
        service_version="1.2.0",
        profile_id="profile",
    )
    repository.finalize_request(
        "request-1",
        execution_status="completed",
        route="rag",
        answer="answer",
    )
    template = usage_review_template_records(
        [usage_record("request-1", question="question", feedback="negative")]
    )[0]
    template.update(
        {
            "review_status": "user_confirmed",
            "failure_type": "retrieval",
            "severity": "major",
            "expected_route": "rag",
            "answerability": True,
            "reason": "Evidence was not retrieved.",
            "eval_candidate": True,
            "candidate_case_id": "candidate-1",
            "candidate_category": "rag_answerable",
            "retrieval_case_id": "retrieval-1",
            "expected_answer": "expected",
            "relevant_evidence": [{"source": "guide.md"}],
        }
    )
    review_file = tmp_path / "review.jsonl"
    review_file.write_text(json.dumps(template) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="strict evidence"):
        repository.save_reviews(load_usage_reviews(review_file))

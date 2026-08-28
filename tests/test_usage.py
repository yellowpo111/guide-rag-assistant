import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
import sqlite3
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fiscal_rag.usage import (  # noqa: E402
    SQLiteUsageRepository,
    UsageFeedbackNotAllowed,
    UsageRequestNotFound,
    UsageReviewNotAllowed,
    UsageReview,
)


def make_repository(tmp_path: Path) -> SQLiteUsageRepository:
    repository = SQLiteUsageRepository(tmp_path / "usage" / "usage.sqlite3")
    repository.initialize()
    return repository


def create_request(
    repository: SQLiteUsageRepository,
    request_id: str = "request-1",
    *,
    endpoint: str = "assistant_stream",
    traffic_kind: str = "production",
) -> None:
    repository.create_request(
        request_id,
        endpoint=endpoint,
        question="怎样保存？",
        service_version="1.2.0",
        profile_id="profile",
        traffic_kind=traffic_kind,
        route="rag" if endpoint == "ask" else None,
    )


def complete_request(
    repository: SQLiteUsageRepository,
    request_id: str = "request-1",
) -> None:
    repository.finalize_request(
        request_id,
        execution_status="completed",
        route="rag",
        answer="点击保存。",
        trace={
            "retrieval_query": "如何保存？",
            "rewrite_query": "如何保存？",
            "query_rewrite_status": "accepted",
            "required_constraints": ["建设单位"],
            "missing_constraints": [],
            "sources": [
                {
                    "rank": 1,
                    "source": "guide.md",
                    "section": "填报",
                    "subsection": "步骤",
                    "dense_score": 0.7,
                    "rerank_score": 0.9,
                }
            ],
        },
        timings_ms={"server_total": 1200.0, "router": 100.0},
        total_duration_ms=1210.0,
    )


def test_repository_initializes_idempotently_and_persists_safe_trace(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    repository.initialize()
    create_request(repository)
    complete_request(repository)

    records = repository.fetch_usage_records()

    assert len(records) == 1
    record = records[0]
    assert record["execution_status"] == "completed"
    assert record["question"] == "怎样保存？"
    assert record["answer"] == "点击保存。"
    assert record["retrieval_query"] == "如何保存？"
    assert record["required_constraints"] == ["建设单位"]
    assert record["timings_ms"] == {"router": 100.0, "server_total": 1200.0}
    assert record["sources"][0]["source"] == "guide.md"

    with sqlite3.connect(repository.database_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(assistant_requests)")
        }
    assert "context" not in columns
    assert "prompt" not in columns
    assert "page_content" not in columns


def test_repository_rolls_back_failed_initial_migration(tmp_path: Path) -> None:
    database = tmp_path / "usage.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE request_sources (placeholder TEXT)")

    repository = SQLiteUsageRepository(database)
    with pytest.raises(sqlite3.OperationalError, match="already exists"):
        repository.initialize()

    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
    assert tables == {"request_sources"}


def test_repository_persists_traffic_kind_and_rejects_unknown_values(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    create_request(repository, "assistant-eval", traffic_kind="assistant_eval")
    complete_request(repository, "assistant-eval")

    record = repository.fetch_usage_records()[0]
    assert record["traffic_kind"] == "assistant_eval"

    with pytest.raises(ValueError, match="traffic_kind"):
        create_request(repository, "unknown", traffic_kind="browser_test")


def test_repository_supports_small_concurrent_writes(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)

    def write_request(index: int) -> None:
        request_id = f"concurrent-{index}"
        create_request(repository, request_id)
        complete_request(repository, request_id)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(write_request, range(12)))

    records = repository.fetch_usage_records()
    assert len(records) == 12
    assert {record["execution_status"] for record in records} == {"completed"}


def test_repository_recovers_started_requests_and_enforces_feedback_state(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    create_request(repository, "started")

    with pytest.raises(UsageFeedbackNotAllowed):
        repository.set_feedback("started", "negative")
    with pytest.raises(UsageFeedbackNotAllowed):
        repository.delete_feedback("started")
    assert repository.mark_started_interrupted() == 1

    create_request(repository, "completed")
    complete_request(repository, "completed")
    repository.set_feedback("completed", "negative")
    repository.set_feedback("completed", "positive")

    records = {record["request_id"]: record for record in repository.fetch_usage_records()}
    assert records["started"]["execution_status"] == "interrupted"
    assert records["started"]["error_code"] == "service_restart"
    assert records["completed"]["feedback_rating"] == "positive"
    assert repository.delete_feedback("completed") is True
    assert repository.delete_feedback("completed") is False
    with pytest.raises(UsageRequestNotFound):
        repository.set_feedback("unknown", "positive")


def test_feedback_and_reviews_reject_nonproduction_requests(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    create_request(repository, "assistant-eval", traffic_kind="assistant_eval")
    complete_request(repository, "assistant-eval")

    with pytest.raises(UsageFeedbackNotAllowed):
        repository.set_feedback("assistant-eval", "positive")
    with pytest.raises(UsageFeedbackNotAllowed):
        repository.delete_feedback("assistant-eval")
    with pytest.raises(UsageReviewNotAllowed):
        repository.save_review(
            UsageReview(
                request_id="assistant-eval",
                review_status="user_confirmed",
                failure_type="no_issue",
                severity="none",
                expected_route="rag",
                answerability=True,
                reason="Evaluation traffic is not a real usage case.",
                eval_candidate=False,
            )
        )


def test_eval_candidate_rejects_conflicting_category_contract(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    create_request(repository)
    complete_request(repository)

    with pytest.raises(ValueError, match="conflict"):
        repository.save_review(
            UsageReview(
                request_id="request-1",
                review_status="user_confirmed",
                failure_type="routing",
                severity="major",
                expected_route="rag",
                answerability=True,
                reason="The route should be checked.",
                eval_candidate=True,
                candidate_case_id="usage-chat-001",
                candidate_category="chat",
            )
        )


def test_repository_prunes_related_private_data_and_creates_consistent_backup(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    create_request(repository)
    complete_request(repository)
    repository.set_feedback("request-1", "negative")
    review = UsageReview(
        request_id="request-1",
        review_status="user_confirmed",
        failure_type="generation_correctness",
        severity="minor",
        expected_route="rag",
        answerability=True,
        reason="The answer omitted a step.",
        eval_candidate=False,
    )
    repository.save_review(review)
    backup_path = repository.backup(tmp_path / "backups" / "usage.sqlite3")

    with sqlite3.connect(backup_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM assistant_requests").fetchone()[0] == 1
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    with pytest.raises(FileExistsError):
        repository.backup(backup_path)

    future = datetime.now(UTC) + timedelta(days=91)
    assert repository.prune_expired(90, now=future) == 1
    assert repository.fetch_usage_records() == []
    with sqlite3.connect(repository.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM request_feedback").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM request_reviews").fetchone()[0] == 0


def test_repository_removes_incomplete_backup_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = make_repository(tmp_path)
    destination = tmp_path / "backups" / "failed.sqlite3"

    def fail_connect(*_args, **_kwargs):
        raise sqlite3.OperationalError("simulated backup failure")

    monkeypatch.setattr(sqlite3, "connect", fail_connect)
    with pytest.raises(sqlite3.OperationalError, match="simulated"):
        repository.backup(destination)

    assert not destination.exists()


def test_eval_candidate_review_requires_ground_truth_for_answerable_rag(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    create_request(repository)
    complete_request(repository)
    incomplete = UsageReview(
        request_id="request-1",
        review_status="user_confirmed",
        failure_type="retrieval",
        severity="major",
        expected_route="rag",
        answerability=True,
        reason="Correct evidence was missing.",
        eval_candidate=True,
        candidate_case_id="usage-001",
        candidate_category="rag_answerable",
    )

    with pytest.raises(ValueError, match="retrieval_case_id"):
        repository.save_review(incomplete)

    repository.save_review(
        UsageReview(
            **{
                **incomplete.__dict__,
                "retrieval_case_id": "usage-retrieval-001",
                "expected_answer": "Use the documented Save action.",
                "evidence_json": json.dumps(
                    [{"source": "guide.md", "evidence_text": "点击保存。"}],
                    ensure_ascii=False,
                ),
            }
        )
    )
    assert repository.fetch_usage_records()[0]["eval_candidate"] == 1


def test_review_batch_is_atomic_when_a_request_is_unknown(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    create_request(repository)
    complete_request(repository)
    reviews = [
        UsageReview(
            request_id=request_id,
            review_status="user_confirmed",
            failure_type="no_issue",
            severity="none",
            expected_route="rag",
            answerability=True,
            reason="Reviewed.",
            eval_candidate=False,
        )
        for request_id in ("request-1", "unknown")
    ]

    with pytest.raises(UsageRequestNotFound):
        repository.save_reviews(reviews)

    assert repository.fetch_usage_records()[0]["review_status"] is None

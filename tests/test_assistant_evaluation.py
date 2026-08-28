import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fiscal_rag.assistant_evaluation import (  # noqa: E402
    AssistantAdjudication,
    AssistantEvalCase,
    adjudication_template_records,
    load_assistant_eval_cases,
    merge_attempt_records,
    result_record,
    summarize_adjudications,
    summarize_assistant_results,
    write_jsonl_exclusive,
)


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def make_cases() -> list[AssistantEvalCase]:
    return [
        AssistantEvalCase("rag-1", "rag_answerable", "业务问题", "rag", True, "v2-001"),
        AssistantEvalCase("chat-1", "chat", "你好", "chat", None),
        AssistantEvalCase("oos-1", "out_of_scope", "写诗", "out_of_scope", None),
    ]


def test_load_assistant_cases_validates_schema_and_answerable_reference(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    write_jsonl(
        path,
        [
            {
                "schema_version": "assistant-eval-v1",
                "case_id": "case-1",
                "category": "rag_answerable",
                "question": "怎样操作？",
                "expected_route": "rag",
                "answerability": True,
                "retrieval_case_id": "v2-001",
            }
        ],
    )

    cases = load_assistant_eval_cases(path)

    assert cases[0].retrieval_case_id == "v2-001"


def test_load_assistant_cases_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    record = {
        "schema_version": "assistant-eval-v1",
        "case_id": "duplicate",
        "category": "chat",
        "question": "你好",
        "expected_route": "chat",
        "answerability": None,
    }
    write_jsonl(path, [record, record])

    with pytest.raises(ValueError, match="repeats case_id"):
        load_assistant_eval_cases(path)


def test_load_assistant_cases_rejects_conflicting_category_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cases.jsonl"
    write_jsonl(
        path,
        [
            {
                "schema_version": "assistant-eval-v1",
                "case_id": "bad-chat",
                "category": "chat",
                "question": "hello",
                "expected_route": "rag",
                "answerability": True,
            }
        ],
    )

    with pytest.raises(ValueError, match="conflicts"):
        load_assistant_eval_cases(path)


def test_assistant_summary_reports_routes_trace_and_critical_failure() -> None:
    cases = make_cases()
    trace = {
        "retrieval_query": "业务问题",
        "query_rewrite_status": "accepted",
        "sources": [{"rank": 1, "source": "guide.md", "rerank_score": 0.9}],
    }
    results = [
        {"case_id": "rag-1", "status": "completed", "actual_route": "rag", "trace": trace},
        {"case_id": "chat-1", "status": "completed", "actual_route": "rag", "trace": trace},
        {"case_id": "oos-1", "status": "failed", "actual_route": None, "trace": None},
    ]

    summary = summarize_assistant_results(cases, results)

    assert summary["completed_cases"] == 2
    assert summary["route_accuracy"] == pytest.approx(1 / 3)
    assert summary["rag_trace_completion_rate"] == 1.0
    assert summary["error_count"] == 1


def test_route_f1_counts_missing_results_as_false_negatives() -> None:
    cases = [
        AssistantEvalCase("rag-1", "rag_answerable", "Q1", "rag", True, "v2-001"),
        AssistantEvalCase("rag-2", "rag_answerable", "Q2", "rag", True, "v2-002"),
        AssistantEvalCase("chat-1", "chat", "Hi", "chat", None),
        AssistantEvalCase("oos-1", "out_of_scope", "Poem", "out_of_scope", None),
    ]
    results = [
        {"case_id": "rag-1", "status": "completed", "actual_route": "rag"},
        {"case_id": "rag-2", "status": "failed", "actual_route": None},
        {"case_id": "chat-1", "status": "completed", "actual_route": "chat"},
        {"case_id": "oos-1", "status": "completed", "actual_route": "out_of_scope"},
    ]

    summary = summarize_assistant_results(cases, results)

    assert summary["route_f1"]["rag"] == pytest.approx(2 / 3)


def test_result_records_attempt_without_losing_first_failure() -> None:
    case = make_cases()[1]
    stream_result = SimpleNamespace(
        route="chat",
        status="completed",
        request_id="request-1",
        events=("start", "route", "delta", "done"),
        trace=None,
        answer="你好",
        timings_ms={"server_total": 2.0},
        client_ttft_ms=1.0,
        client_total_ms=3.0,
        error_code=None,
        error_message=None,
    )

    record = result_record(case, attempt=2, run_id="run-1", result=stream_result)

    assert record["attempt"] == 2
    assert record["run_id"] == "run-1"
    assert record["answer"] == "你好"


def test_adjudication_template_stays_pending_until_human_review() -> None:
    cases = make_cases()
    templates = adjudication_template_records(
        cases,
        [{"case_id": "chat-1", "actual_route": "chat"}],
    )

    assert templates[1]["route_correct"] is True
    assert templates[1]["review_status"] == "pending"
    assert templates[0]["route_correct"] is None
    assert templates[1]["question"] == "你好"
    assert templates[1]["actual_route"] == "chat"


def test_adjudication_summary_requires_full_user_confirmation() -> None:
    cases = make_cases()
    adjudications = {
        case.case_id: AssistantAdjudication(
            case_id=case.case_id,
            review_status="user_confirmed",
            route_correct=True,
            answer_support="supported" if case.expected_route == "rag" else "not_applicable",
            answer_correctness="correct" if case.expected_route == "rag" else "not_applicable",
            abstention="not_applicable",
            boundary_compliance="compliant",
            source_trace_quality="sufficient" if case.expected_route == "rag" else "not_applicable",
            reason="Reviewed.",
        )
        for case in cases
    }

    summary = summarize_adjudications(cases, adjudications)

    assert summary["total_confirmed"] == 3
    assert summary["release_blockers"] == []


def test_adjudication_summary_rejects_pending_review() -> None:
    cases = make_cases()
    adjudications = {
        case.case_id: AssistantAdjudication(
            case_id=case.case_id,
            review_status="pending",
            route_correct=None,
            answer_support="pending",
            answer_correctness="pending",
            abstention="pending",
            boundary_compliance="pending",
            source_trace_quality="pending",
            reason="Pending human review.",
        )
        for case in cases
    }

    with pytest.raises(ValueError, match="not user-confirmed"):
        summarize_adjudications(cases, adjudications)


def test_jsonl_writer_never_overwrites_existing_results(tmp_path: Path) -> None:
    output = tmp_path / "results.jsonl"
    write_jsonl_exclusive(output, [{"attempt": 1, "status": "failed"}])

    with pytest.raises(FileExistsError):
        write_jsonl_exclusive(output, [{"attempt": 2, "status": "completed"}])

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "attempt": 1,
        "status": "failed",
    }


def test_merge_attempts_keeps_highest_attempt_and_case_order() -> None:
    cases = make_cases()
    attempt_one = [
        {
            "schema_version": "assistant-eval-result-v1",
            "run_id": "run-1",
            "attempt": 1,
            "case_id": case.case_id,
            "status": "failed" if case.case_id == "rag-1" else "completed",
        }
        for case in cases
    ]
    retry = [
        {
            "schema_version": "assistant-eval-result-v1",
            "run_id": "run-1",
            "attempt": 2,
            "case_id": "rag-1",
            "status": "completed",
        }
    ]

    merged = merge_attempt_records(cases, [retry, attempt_one])

    assert [record["case_id"] for record in merged] == [
        "rag-1",
        "chat-1",
        "oos-1",
    ]
    assert merged[0]["attempt"] == 2
    assert merged[0]["status"] == "completed"


def test_merge_attempts_requires_full_coverage() -> None:
    incomplete = [
        {
            "schema_version": "assistant-eval-result-v1",
            "run_id": "run-1",
            "attempt": 1,
            "case_id": "rag-1",
        }
    ]

    with pytest.raises(ValueError, match="missing cases"):
        merge_attempt_records(make_cases(), [incomplete])

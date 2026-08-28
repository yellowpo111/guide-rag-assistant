import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fiscal_rag.performance import percentile, summarize_performance_records  # noqa: E402
from scripts.run_performance_eval import load_performance_cases, parse_arguments  # noqa: E402


def test_parse_arguments_can_start_temporary_local_service() -> None:
    arguments = parse_arguments(["--start-local-service"])

    assert arguments.start_local_service is True


def test_percentile_uses_linear_interpolation() -> None:
    assert percentile([10.0, 20.0, 30.0], 0.5) == 20.0
    assert percentile([10.0, 20.0], 0.95) == pytest.approx(19.5)


def test_percentile_rejects_empty_or_invalid_quantile() -> None:
    with pytest.raises(ValueError, match="at least one"):
        percentile([], 0.5)
    with pytest.raises(ValueError, match="between 0 and 1"):
        percentile([1.0], 1.1)


def test_performance_summary_excludes_warmups_and_reports_queue_wait() -> None:
    records = [
        {
            "run_kind": "warmup",
            "case_id": "rag-1",
            "actual_route": "rag",
            "status": "completed",
            "client_total_ms": 999.0,
            "timings_ms": {"queue_wait": 0.1},
        },
        {
            "run_kind": "measured",
            "case_id": "rag-1",
            "actual_route": "rag",
            "status": "completed",
            "client_ttft_ms": 100.0,
            "client_total_ms": 300.0,
            "timings_ms": {"queue_wait": 0.2, "rerank": 50.0},
        },
        {
            "run_kind": "concurrency",
            "case_id": "rag-1",
            "actual_route": "rag",
            "status": "completed",
            "client_total_ms": 500.0,
            "timings_ms": {"queue_wait": 200.0},
        },
    ]

    summary = summarize_performance_records(records)

    assert summary["measured_requests"] == 1
    assert summary["first_request_success_rate"] == 1.0
    assert summary["overall"]["client_total_ms"]["p50"] == 300.0
    assert summary["concurrency_sanity"]["queue_wait_ms"]["p50"] == 200.0


def test_performance_cases_require_frozen_route_distribution(tmp_path: Path) -> None:
    records = []
    for route, count in (("rag", 8), ("chat", 4), ("out_of_scope", 4)):
        for index in range(count):
            records.append(
                {
                    "schema_version": "performance-eval-v1",
                    "case_id": f"{route}-{index}",
                    "question": "test question",
                    "expected_route": route,
                }
            )
    path = tmp_path / "performance.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    cases = load_performance_cases(path)

    assert len(cases) == 16


def test_performance_cases_reject_wrong_route_distribution(tmp_path: Path) -> None:
    path = tmp_path / "performance.jsonl"
    path.write_text(
        json.dumps(
            {
                "schema_version": "performance-eval-v1",
                "case_id": "rag-1",
                "question": "test question",
                "expected_route": "rag",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="route counts 8/4/4"):
        load_performance_cases(path)

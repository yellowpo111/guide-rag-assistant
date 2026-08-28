from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.verify_release import (  # noqa: E402
    EXPECTED_CHUNKS,
    EXPECTED_DOCUMENTS,
    EXPECTED_EMBEDDING_DIMENSION,
    EXPECTED_EMBEDDING_MODEL,
    EXPECTED_GENERATION_MODEL,
    EXPECTED_PRIVATE_EVAL_RECORDS,
    EXPECTED_RERANK_MODEL,
    FROZEN_BASELINE_VERSION,
    RELEASE_VERSION,
    _verify_assistant_summary,
    _verify_performance_summary,
    _verify_startup_result,
    _verify_static_release_contract,
    _verify_text_to_sql_contract,
    _verify_text_to_sql_summary,
    _verify_usage_store_contract,
    _verify_v1_2_assistant_summary,
)
from scripts.run_performance_eval import RELEASE_VERSION as PERFORMANCE_RELEASE_VERSION  # noqa: E402
from scripts import verify_release  # noqa: E402
from fiscal_rag.version import __version__  # noqa: E402


def test_v1_1_static_release_contract_is_frozen() -> None:
    _verify_static_release_contract()
    _verify_usage_store_contract()
    _verify_text_to_sql_contract()

    assert RELEASE_VERSION == "1.4.0"
    assert PERFORMANCE_RELEASE_VERSION == f"v{__version__}" == "v1.4.0"
    assert FROZEN_BASELINE_VERSION == "1.1.0"
    assert EXPECTED_DOCUMENTS == 29
    assert EXPECTED_CHUNKS == 1000
    assert EXPECTED_EMBEDDING_DIMENSION == 1024
    assert EXPECTED_EMBEDDING_MODEL == "qwen3.7-text-embedding"
    assert EXPECTED_RERANK_MODEL == "qwen3-rerank"
    assert EXPECTED_GENERATION_MODEL == "deepseek-v4-flash"
    assert EXPECTED_PRIVATE_EVAL_RECORDS["data_private/evals/assistant_eval_v1.jsonl"] == 54
    assert EXPECTED_PRIVATE_EVAL_RECORDS["data_private/evals/performance_eval_v1.jsonl"] == 16


def test_release_contract_checks_create_missing_private_data_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clean clone has no ignored data_private directory until verification runs."""
    monkeypatch.setattr(verify_release, "PROJECT_ROOT", tmp_path)

    verify_release._verify_usage_store_contract()
    verify_release._verify_text_to_sql_contract()

    assert (tmp_path / "data_private").is_dir()


def test_release_evidence_summary_contracts() -> None:
    _verify_assistant_summary(
        {
            "automatic": {
                "total_cases": 54,
                "completed_cases": 54,
                "sse_completion_rate": 1.0,
                "route_accuracy": 1.0,
                "route_macro_f1": 1.0,
                "rag_trace_completion_rate": 1.0,
                "error_count": 0,
            },
            "human_adjudication": {
                "total_confirmed": 54,
                "release_blockers": [],
            },
            "release_gate": "passed",
            "release_blockers": [],
        }
    )
    _verify_performance_summary(
        {
            "measured_requests": 48,
            "completion_rate": 1.0,
            "first_request_success_rate": 1.0,
            "routes": {
                "rag": {"requests": 24},
                "chat": {"requests": 12},
                "out_of_scope": {"requests": 12},
            },
            "concurrency_sanity": {"requests": 6, "completion_rate": 1.0},
        }
    )
    _verify_startup_result(
        {
            "release_version": "v1.1.0",
            "probe_path": "/health/ready",
            "startup_ready_ms": 4307.067,
        }
    )
    _verify_v1_2_assistant_summary(
        {
            "total_cases": 54,
            "completed_cases": 54,
            "sse_completion_rate": 1.0,
            "route_accuracy": 1.0,
            "route_macro_f1": 1.0,
            "rag_trace_completion_rate": 1.0,
            "error_count": 0,
        }
    )
    _verify_text_to_sql_summary(
        {
            "schema_version": "usage-text-to-sql-eval-summary-v1",
            "data_origin": "synthetic_usage_fixture_v1",
            "contains_real_usage_data": False,
            "total_cases": 12,
            "answerable_cases": 10,
            "refusal_cases": 2,
            "answerable_matches": 8,
            "refusal_matches": 2,
            "prototype_decision": "adopted",
        }
    )


@pytest.mark.parametrize(
    ("verifier", "value"),
    [
        (
            _verify_assistant_summary,
            {
                "automatic": {},
                "human_adjudication": {"total_confirmed": 54, "release_blockers": []},
                "release_gate": "passed",
                "release_blockers": [],
            },
        ),
        (
            _verify_performance_summary,
            {"routes": {}, "concurrency_sanity": {}},
        ),
        (
            _verify_startup_result,
            {
                "release_version": "v1.1.0",
                "probe_path": "/health/ready",
                "startup_ready_ms": 0,
            },
        ),
        (
            _verify_v1_2_assistant_summary,
            {
                "total_cases": 54,
                "completed_cases": 53,
                "sse_completion_rate": 1.0,
                "route_accuracy": 1.0,
                "route_macro_f1": 1.0,
                "rag_trace_completion_rate": 1.0,
                "error_count": 0,
            },
        ),
        (
            _verify_text_to_sql_summary,
            {
                "schema_version": "usage-text-to-sql-eval-summary-v1",
                "data_origin": "synthetic_usage_fixture_v1",
                "contains_real_usage_data": False,
                "total_cases": 12,
                "answerable_cases": 10,
                "refusal_cases": 2,
                "answerable_matches": 7,
                "refusal_matches": 2,
                "prototype_decision": "adopted",
            },
        ),
    ],
)
def test_release_evidence_summary_contracts_reject_incomplete_results(
    verifier, value: dict[str, object]
) -> None:
    with pytest.raises(RuntimeError):
        verifier(value)

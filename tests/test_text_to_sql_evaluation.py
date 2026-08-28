from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fiscal_rag.text_to_sql import GeneratedSQL  # noqa: E402
from fiscal_rag.text_to_sql_evaluation import (  # noqa: E402
    SYNTHETIC_DATA_ORIGIN,
    build_synthetic_usage_fixture,
    evaluate_text_to_sql_cases,
    load_text_to_sql_eval_cases,
    _results_match,
)


CASES_FILE = (
    Path(__file__).resolve().parent / "fixtures" / "text_to_sql_usage_eval_v1.jsonl"
)


class GoldGenerator:
    model_id = "gold-fixture"

    def __init__(self, cases) -> None:
        self.by_question = {case.question: case for case in cases}

    def generate(self, question: str) -> GeneratedSQL:
        case = self.by_question[question]
        if case.expected_refusal:
            return GeneratedSQL(sql=None, refusal_code="unsupported_query")
        return GeneratedSQL(case.gold_sql)


class UnsafeSensitiveGenerator(GoldGenerator):
    def generate(self, question: str) -> GeneratedSQL:
        case = self.by_question[question]
        if case.expected_refusal:
            return GeneratedSQL("SELECT question FROM assistant_requests")
        return GeneratedSQL(case.gold_sql)


def test_versioned_eval_cases_and_gold_queries_pass_on_synthetic_fixture(
    tmp_path: Path,
) -> None:
    cases = load_text_to_sql_eval_cases(CASES_FILE)
    database = build_synthetic_usage_fixture(tmp_path / "fixture.sqlite3")

    details, summary = evaluate_text_to_sql_cases(
        cases,
        generator=GoldGenerator(cases),
        database_path=database,
        run_id="fixture-run",
    )

    assert len(cases) == len(details) == 12
    assert all(record["correct"] for record in details)
    assert all(record["data_origin"] == SYNTHETIC_DATA_ORIGIN for record in details)
    assert summary["answerable_matches"] == 10
    assert summary["refusal_matches"] == 2
    assert summary["denotation_accuracy"] == 1.0
    assert summary["refusal_accuracy"] == 1.0
    assert summary["prototype_decision"] == "adopted"
    assert summary["contains_real_usage_data"] is False


def test_sensitive_eval_accepts_executor_rejection_as_safe_refusal(
    tmp_path: Path,
) -> None:
    cases = load_text_to_sql_eval_cases(CASES_FILE)
    details, summary = evaluate_text_to_sql_cases(
        cases,
        generator=UnsafeSensitiveGenerator(cases),
        database_path=build_synthetic_usage_fixture(tmp_path / "fixture.sqlite3"),
        run_id="unsafe-run",
    )

    sensitive = [record for record in details if record["expected_refusal"]]
    assert [record["error_code"] for record in sensitive] == [
        "sql_rejected",
        "sql_rejected",
    ]
    assert all(record["correct"] for record in sensitive)
    assert summary["refusal_matches"] == 2


def test_result_comparison_uses_float_tolerance_and_order_contract() -> None:
    assert _results_match([["rag", 1.0000004]], [["rag", 1.0]], order_sensitive=True)
    assert _results_match(
        [["chat", 2], ["rag", 1]],
        [["rag", 1], ["chat", 2]],
        order_sensitive=False,
    )
    assert not _results_match(
        [["chat", 2], ["rag", 1]],
        [["rag", 1], ["chat", 2]],
        order_sensitive=True,
    )

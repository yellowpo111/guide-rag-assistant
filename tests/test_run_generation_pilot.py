import importlib.util
import json
from pathlib import Path
import sys

import pytest
from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fiscal_rag.evaluation import Evidence, RetrievalEvalCase  # noqa: E402
from fiscal_rag.pipeline import RAGResult  # noqa: E402


def load_runner_module():
    script_path = PROJECT_ROOT / "scripts" / "run_generation_pilot.py"
    spec = importlib.util.spec_from_file_location("run_generation_pilot", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pilot_record(case_id: str = "case-001") -> dict[str, str]:
    return {
        "schema_version": "generation-pilot-v1",
        "case_id": case_id,
        "category": "strict_rank_1",
    }


def eval_case(case_id: str = "case-001") -> RetrievalEvalCase:
    return RetrievalEvalCase(
        case_id=case_id,
        question="How do I save the document?",
        expected_answer="Use Save.",
        relevant_evidence=(
            Evidence(
                source="guide.md",
                section="Section",
                subsection="Steps",
                evidence_text="Use Save.",
            ),
        ),
    )


def test_runner_defaults_to_live_experimental_profile() -> None:
    runner = load_runner_module()

    arguments = runner.parse_arguments(
        ["--pilot-file", "pilot.jsonl", "--details-file", "details.jsonl"]
    )

    assert arguments.retrieval_profile == "dense_rerank_live_rewrite_guard"
    assert arguments.candidate_k == 20


def test_runner_accepts_candidate_k_and_repeated_frozen_rewrite_sources() -> None:
    runner = load_runner_module()

    arguments = runner.parse_arguments(
        [
            "--pilot-file",
            "pilot.jsonl",
            "--details-file",
            "details.jsonl",
            "--candidate-k",
            "10",
            "--rewrite-source-details",
            "first.jsonl",
            "--rewrite-source-details",
            "second.jsonl",
        ]
    )

    assert arguments.candidate_k == 10
    assert arguments.rewrite_source_details == [Path("first.jsonl"), Path("second.jsonl")]


def test_runner_rejects_output_k_larger_than_candidate_k() -> None:
    runner = load_runner_module()
    arguments = runner.parse_arguments(
        [
            "--pilot-file",
            "pilot.jsonl",
            "--details-file",
            "details.jsonl",
            "--k",
            "5",
            "--candidate-k",
            "4",
        ]
    )

    with pytest.raises(ValueError, match="candidate-k"):
        runner.validate_experiment_arguments(arguments)


def test_pilot_categories_select_known_cases_in_manifest_order(tmp_path: Path) -> None:
    runner = load_runner_module()
    pilot_path = tmp_path / "pilot.jsonl"
    pilot_path.write_text(
        "\n".join(
            [
                json.dumps(pilot_record("case-002")),
                json.dumps(pilot_record("case-001")),
            ]
        ),
        encoding="utf-8",
    )

    selected = runner.select_pilot_cases(
        [eval_case("case-001"), eval_case("case-002")],
        runner.load_pilot_categories(pilot_path),
    )

    assert [(case.case_id, category) for case, category in selected] == [
        ("case-002", "strict_rank_1"),
        ("case-001", "strict_rank_1"),
    ]


def test_pilot_categories_reject_unknown_eval_case() -> None:
    runner = load_runner_module()

    with pytest.raises(ValueError, match="unknown eval case_ids"):
        runner.select_pilot_cases([eval_case()], {"unknown": "strict_rank_1"})


def test_generation_record_preserves_context_answer_and_retrieval_trace() -> None:
    runner = load_runner_module()
    document = Document(
        page_content="Use Save.",
        metadata={
            "source": "guide.md",
            "section": "Section",
            "subsection": "Steps",
            "_dense_score": 0.7,
            "_rerank_score": 0.9,
        },
    )
    result = RAGResult(
        question="How do I save the document?",
        retrieved_results=[(document, 0.9)],
        context="[Source: guide.md]\n\nUse Save.",
        answer="Use Save.",
        retrieval_query="How can I save?",
        rewrite_query="How can I save?",
        query_rewrite_status="accepted",
    )

    record = runner.rag_result_to_record(
        eval_case(), "strict_rank_1", "dense_rerank_live_rewrite_guard", result
    )

    assert record["expected_answer"] == "Use Save."
    assert record["retrieval_query"] == "How can I save?"
    assert record["query_rewrite_status"] == "accepted"
    assert record["retrieved_results"] == [
        {
            "rank": 1,
            "score": 0.9,
            "source": "guide.md",
            "section": "Section",
            "subsection": "Steps",
            "page_content": "Use Save.",
            "dense_score": 0.7,
            "rerank_score": 0.9,
        }
    ]
    assert record["context"] == "[Source: guide.md]\n\nUse Save."
    assert record["answer"] == "Use Save."


def test_generation_failure_record_preserves_case_and_error_without_answer() -> None:
    runner = load_runner_module()

    record = runner.generation_failure_to_record(
        eval_case(),
        "strict_rank_1",
        "dense_rerank_live_rewrite_guard",
        ValueError("DeepSeek returned an empty response."),
        candidate_k=10,
        frozen_rewrite_source_files=["frozen.jsonl"],
    )

    assert record["schema_version"] == "generation-pilot-failure-v1"
    assert record["case_id"] == "case-001"
    assert record["generation_error_type"] == "ValueError"
    assert "empty response" in record["generation_error_message"]
    assert "answer" not in record
    assert record["dense_candidate_k"] == 10


def test_frozen_rewrite_sources_merge_without_duplicate_questions(tmp_path: Path) -> None:
    runner = load_runner_module()
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(
        json.dumps({"question": "Question A", "retrieval_query": "Rewrite A"}),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps({"question": "Question B", "retrieval_query": "Rewrite B"}),
        encoding="utf-8",
    )

    rewrites = runner.load_frozen_query_rewrites_from_sources([first, second])

    assert rewrites == {"Question A": "Rewrite A", "Question B": "Rewrite B"}

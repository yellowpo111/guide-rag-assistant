import json
from pathlib import Path
import sys

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fiscal_rag.evaluation import (  # noqa: E402
    Evidence,
    RetrievalEvalCase,
    case_evaluation_to_record,
    evaluate_case,
    is_relevant,
    normalize_text,
    write_evaluation_details,
)


EVIDENCE = Evidence(
    source="guide.md",
    section="Section",
    subsection="Subsection",
    evidence_text="First line\nsecond line",
)
CASE = RetrievalEvalCase(
    case_id="case-001",
    question="How do I complete the task?",
    expected_answer="Use the documented steps.",
    relevant_evidence=(EVIDENCE,),
)


def document(content: str, source: str = "guide.md") -> Document:
    return Document(
        page_content=content,
        metadata={"source": source, "section": "Section", "subsection": "Subsection"},
    )


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text("  First\r\n\t second   line  ") == "First second line"


def test_relevant_when_evidence_is_fully_contained_in_same_source() -> None:
    assert is_relevant(document("Prefix\nFirst line   second line\nSuffix"), EVIDENCE)


def test_same_metadata_without_evidence_is_not_relevant() -> None:
    assert not is_relevant(document("The documented steps are elsewhere."), EVIDENCE)


def test_same_evidence_in_different_source_is_not_relevant() -> None:
    assert not is_relevant(document("First line second line", source="other.md"), EVIDENCE)


def test_rank_one_hit_has_perfect_metrics() -> None:
    evaluation = evaluate_case(
        CASE,
        [document("First line second line")],
        [(document("First line second line"), 0.9)],
    )

    assert evaluation.coverage_status == "OK"
    assert evaluation.first_relevant_rank == 1
    assert (evaluation.hit_at_1, evaluation.hit_at_3, evaluation.hit_at_5) == (1, 1, 1)
    assert evaluation.reciprocal_rank == 1


def test_rank_three_hit_has_expected_metrics() -> None:
    relevant_document = document("First line second line")
    evaluation = evaluate_case(
        CASE,
        [relevant_document],
        [
            (document("not relevant one"), 0.9),
            (document("not relevant two"), 0.8),
            (relevant_document, 0.7),
        ],
    )

    assert evaluation.first_relevant_rank == 3
    assert (evaluation.hit_at_1, evaluation.hit_at_3, evaluation.hit_at_5) == (0, 1, 1)
    assert evaluation.reciprocal_rank == 1 / 3


def test_top_five_miss_has_zero_metrics() -> None:
    covered_document = document("First line second line")
    evaluation = evaluate_case(
        CASE,
        [covered_document],
        [(document(f"not relevant {index}"), 1.0 - index / 10) for index in range(5)],
    )

    assert evaluation.coverage_status == "OK"
    assert evaluation.first_relevant_rank is None
    assert (evaluation.hit_at_1, evaluation.hit_at_3, evaluation.hit_at_5) == (0, 0, 0)
    assert evaluation.reciprocal_rank == 0
    assert evaluation.retrieval_miss


def test_missing_corpus_evidence_is_coverage_failure() -> None:
    evaluation = evaluate_case(
        CASE,
        [document("different corpus content")],
        [(document("different retrieval content"), 0.8)],
    )

    assert evaluation.coverage_status == "COVERAGE_FAILURE"
    assert evaluation.first_relevant_rank is None
    assert (evaluation.hit_at_1, evaluation.hit_at_3, evaluation.hit_at_5) == (0, 0, 0)
    assert evaluation.reciprocal_rank == 0
    assert not evaluation.retrieval_miss


def test_write_evaluation_details_preserves_ground_truth_and_results(
    tmp_path: Path,
) -> None:
    relevant_document = document("Prefix First line second line Suffix")
    evaluation = evaluate_case(
        CASE,
        [relevant_document],
        [
            (document("not relevant"), 0.9),
            (relevant_document, 0.8),
        ],
    )
    details_path = tmp_path / "results" / "details.jsonl"

    write_evaluation_details(details_path, [evaluation])

    records = [
        json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records == [case_evaluation_to_record(evaluation)]

    record = records[0]
    assert record["relevant_evidence"] == [
        {
            "source": "guide.md",
            "section": "Section",
            "subsection": "Subsection",
            "evidence_text": "First line\nsecond line",
        }
    ]
    assert record["retrieved_results"] == [
        {
            "rank": 1,
            "score": 0.9,
            "source": "guide.md",
            "section": "Section",
            "subsection": "Subsection",
            "page_content": "not relevant",
        },
        {
            "rank": 2,
            "score": 0.8,
            "source": "guide.md",
            "section": "Section",
            "subsection": "Subsection",
            "page_content": "Prefix First line second line Suffix",
        },
    ]


def test_detail_record_preserves_optional_dense_and_rerank_scores() -> None:
    reranked_document = Document(
        page_content="Prefix First line second line Suffix",
        metadata={
            "source": "guide.md",
            "section": "Section",
            "subsection": "Subsection",
            "_dense_score": 0.74,
            "_rerank_score": 0.93,
        },
    )
    evaluation = evaluate_case(CASE, [reranked_document], [(reranked_document, 0.93)])

    record = case_evaluation_to_record(evaluation)

    assert record["retrieved_results"] == [
        {
            "rank": 1,
            "score": 0.93,
            "source": "guide.md",
            "section": "Section",
            "subsection": "Subsection",
            "page_content": "Prefix First line second line Suffix",
            "dense_score": 0.74,
            "rerank_score": 0.93,
        }
    ]


def test_detail_record_preserves_optional_hybrid_candidate_scores() -> None:
    reranked_document = Document(
        page_content="Prefix First line second line Suffix",
        metadata={
            "source": "guide.md",
            "section": "Section",
            "subsection": "Subsection",
            "_dense_score": 0.74,
            "_bm25_score": 3.2,
            "_rrf_score": 0.03,
            "_rerank_score": 0.93,
        },
    )
    evaluation = evaluate_case(CASE, [reranked_document], [(reranked_document, 0.93)])

    record = case_evaluation_to_record(evaluation)

    assert record["retrieved_results"][0]["bm25_score"] == 3.2
    assert record["retrieved_results"][0]["rrf_score"] == 0.03


def test_detail_record_optionally_preserves_retrieval_query() -> None:
    relevant_document = document("Prefix First line second line Suffix")
    evaluation = evaluate_case(CASE, [relevant_document], [(relevant_document, 0.9)])

    record = case_evaluation_to_record(
        evaluation,
        retrieval_query="Clear rewritten retrieval question",
    )

    assert record["question"] == CASE.question
    assert record["retrieval_query"] == "Clear rewritten retrieval question"


def test_write_evaluation_details_preserves_retrieval_query(tmp_path: Path) -> None:
    relevant_document = document("Prefix First line second line Suffix")
    evaluation = evaluate_case(CASE, [relevant_document], [(relevant_document, 0.9)])
    details_path = tmp_path / "results" / "details.jsonl"

    write_evaluation_details(
        details_path,
        [evaluation],
        retrieval_queries={"case-001": "Clear rewritten retrieval question"},
    )

    record = json.loads(details_path.read_text(encoding="utf-8"))
    assert record["question"] == CASE.question
    assert record["retrieval_query"] == "Clear rewritten retrieval question"


def test_write_evaluation_details_preserves_query_rewrite_guard_record(
    tmp_path: Path,
) -> None:
    relevant_document = document("Prefix First line second line Suffix")
    evaluation = evaluate_case(CASE, [relevant_document], [(relevant_document, 0.9)])
    details_path = tmp_path / "results" / "details.jsonl"
    guard_record = {
        "rewrite_query": "Rewritten question without role",
        "query_rewrite_status": "fallback_to_original",
        "required_constraints": ["建设单位"],
        "missing_constraints": ["建设单位"],
    }

    write_evaluation_details(
        details_path,
        [evaluation],
        retrieval_queries={"case-001": CASE.question},
        query_rewrite_records={"case-001": guard_record},
    )

    record = json.loads(details_path.read_text(encoding="utf-8"))
    assert record["retrieval_query"] == CASE.question
    assert record["query_rewrite_status"] == "fallback_to_original"
    assert record["missing_constraints"] == ["建设单位"]


def test_write_evaluation_details_preserves_static_experiment_fields(
    tmp_path: Path,
) -> None:
    relevant_document = document("Prefix First line second line Suffix")
    evaluation = evaluate_case(CASE, [relevant_document], [(relevant_document, 0.9)])
    details_path = tmp_path / "results" / "details.jsonl"

    write_evaluation_details(
        details_path,
        [evaluation],
        detail_fields={"rerank_document_profile": "metadata_context"},
    )

    record = json.loads(details_path.read_text(encoding="utf-8"))
    assert record["rerank_document_profile"] == "metadata_context"


def test_write_evaluation_details_preserves_pre_rerank_dense_candidates(
    tmp_path: Path,
) -> None:
    relevant_document = document("Prefix First line second line Suffix")
    candidate_document = document("Dense candidate content")
    evaluation = evaluate_case(CASE, [relevant_document], [(relevant_document, 0.9)])
    details_path = tmp_path / "results" / "details.jsonl"

    write_evaluation_details(
        details_path,
        [evaluation],
        dense_candidate_results={
            "case-001": [(candidate_document, 0.81), (relevant_document, 0.72)]
        },
    )

    record = json.loads(details_path.read_text(encoding="utf-8"))
    assert record["dense_candidates"] == [
        {
            "dense_rank": 1,
            "dense_score": 0.81,
            "source": "guide.md",
            "section": "Section",
            "subsection": "Subsection",
            "page_content": "Dense candidate content",
        },
        {
            "dense_rank": 2,
            "dense_score": 0.72,
            "source": "guide.md",
            "section": "Section",
            "subsection": "Subsection",
            "page_content": "Prefix First line second line Suffix",
        },
    ]

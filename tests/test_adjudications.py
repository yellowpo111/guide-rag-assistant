import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fiscal_rag.adjudications import (  # noqa: E402
    SCHEMA_VERSION,
    load_retrieval_adjudications,
    summarize_saved_details,
    validate_adjudications_against_corpus,
)


def annotation(case_id: str = "case-001") -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "acceptable_evidence": [
            {
                "source": "guide.md",
                "section": "Section",
                "subsection": "Steps",
                "evidence_text": "Use the Save button.",
                "reason": "The passage directly provides the requested action.",
            }
        ],
    }


def details_record(
    *, strict_rank: int | None = 2, acceptable_rank: int = 1
) -> dict[str, object]:
    return {
        "case_id": "case-001",
        "first_relevant_rank": strict_rank,
        "retrieved_results": [
            {
                "rank": acceptable_rank,
                "source": "guide.md",
                "page_content": "First choose the item. Use the Save button.",
            },
            {
                "rank": 2,
                "source": "guide.md",
                "page_content": "The strict source evidence appears here.",
            },
        ],
    }


def test_load_adjudications_and_validate_raw_source(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "guide.md").write_text(
        "First choose the item. Use the Save button.", encoding="utf-8"
    )
    path = tmp_path / "adjudications.jsonl"
    path.write_text(json.dumps(annotation(), ensure_ascii=False) + "\n", encoding="utf-8")

    adjudications = load_retrieval_adjudications(path)

    assert list(adjudications) == ["case-001"]
    assert validate_adjudications_against_corpus(adjudications, corpus_dir) == []


def test_adjudications_reject_duplicate_case_ids(tmp_path: Path) -> None:
    path = tmp_path / "adjudications.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(annotation(), ensure_ascii=False),
                json.dumps(annotation(), ensure_ascii=False),
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="repeats case_id"):
        load_retrieval_adjudications(path)


def test_adjudications_report_missing_source_or_evidence(tmp_path: Path) -> None:
    path = tmp_path / "adjudications.jsonl"
    path.write_text(json.dumps(annotation(), ensure_ascii=False), encoding="utf-8")

    adjudications = load_retrieval_adjudications(path)

    assert validate_adjudications_against_corpus(adjudications, tmp_path / "missing") == [
        "case-001: missing source 'guide.md'"
    ]


def test_summary_adds_acceptable_evidence_without_changing_strict_rank(
    tmp_path: Path,
) -> None:
    path = tmp_path / "adjudications.jsonl"
    path.write_text(json.dumps(annotation(), ensure_ascii=False), encoding="utf-8")
    adjudications = load_retrieval_adjudications(path)

    summary = summarize_saved_details([details_record()], adjudications)

    result = summary.case_results[0]
    assert result.strict_first_relevant_rank == 2
    assert result.adjudicated_first_relevant_rank == 1
    assert summary.strict_hit_at_1 == 0.0
    assert summary.adjudicated_hit_at_1 == 1.0
    assert summary.strict_mrr == 0.5
    assert summary.adjudicated_mrr == 1.0
    assert [item.case_id for item in summary.strict_only_rank_1_overrides] == [
        "case-001"
    ]


def test_summary_keeps_strict_rank_when_no_adjudication_exists() -> None:
    summary = summarize_saved_details(
        [details_record(strict_rank=1)], adjudications={}
    )

    result = summary.case_results[0]
    assert result.strict_first_relevant_rank == 1
    assert result.adjudicated_first_relevant_rank == 1

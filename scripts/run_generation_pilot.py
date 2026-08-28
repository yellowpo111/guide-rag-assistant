"""Run a small, private, human-reviewed generation pilot without auto-scoring answers."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fiscal_rag.evaluation import RetrievalEvalCase, load_retrieval_eval_cases  # noqa: E402
from fiscal_rag.pipeline import (  # noqa: E402
    DENSE_RERANK_CANDIDATE_K,
    RAGResult,
    build_basic_rag_pipeline,
    build_dense_rerank_rag_pipeline,
)
from fiscal_rag.query_rewrite import FrozenQueryRewriter, load_frozen_query_rewrites  # noqa: E402


DEFAULT_EVAL_FILE = PROJECT_ROOT / "data_private" / "evals" / "retrieval_eval_v1.jsonl"
DEFAULT_CORPUS_DIR = PROJECT_ROOT / "data_private" / "corpus"
RETRIEVAL_PROFILES = ("dense_baseline", "dense_rerank_live_rewrite_guard")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate private, human-reviewable RAG pilot records."
    )
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--pilot-file", type=Path, required=True)
    parser.add_argument("--details-file", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument(
        "--retrieval-profile",
        choices=RETRIEVAL_PROFILES,
        default="dense_rerank_live_rewrite_guard",
    )
    parser.add_argument("--k", type=positive_integer, default=5)
    parser.add_argument(
        "--candidate-k",
        type=positive_integer,
        default=DENSE_RERANK_CANDIDATE_K,
        help=(
            "Dense candidates sent to the reranker for the experimental profile "
            f"(default: {DENSE_RERANK_CANDIDATE_K})."
        ),
    )
    parser.add_argument(
        "--rewrite-source-details",
        action="append",
        type=Path,
        default=[],
        help=(
            "Private generation-details JSONL containing frozen retrieval_query "
            "values. Repeat this option for multiple source files."
        ),
    )
    return parser.parse_args(argv)


def positive_integer(value: str) -> int:
    integer = int(value)
    if integer <= 0:
        raise argparse.ArgumentTypeError("k must be positive")
    return integer


def load_pilot_categories(path: str | Path) -> dict[str, str]:
    categories: dict[str, str] = {}
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, Mapping):
            raise ValueError(f"Pilot line {line_number} must be an object.")
        if record.get("schema_version") != "generation-pilot-v1":
            raise ValueError(
                f"Pilot line {line_number} must use schema_version "
                "'generation-pilot-v1'."
            )
        case_id = _required_string(record, "case_id", line_number)
        category = _required_string(record, "category", line_number)
        if case_id in categories:
            raise ValueError(f"Pilot line {line_number} repeats case_id: {case_id!r}")
        categories[case_id] = category
    if not categories:
        raise ValueError("Generation pilot JSONL contains no records.")
    return categories


def select_pilot_cases(
    cases: Sequence[RetrievalEvalCase], categories: Mapping[str, str]
) -> list[tuple[RetrievalEvalCase, str]]:
    cases_by_id = {case.case_id: case for case in cases}
    unknown_case_ids = sorted(set(categories) - set(cases_by_id))
    if unknown_case_ids:
        raise ValueError(
            "Generation pilot references unknown eval case_ids: "
            + ", ".join(unknown_case_ids)
        )
    return [(cases_by_id[case_id], category) for case_id, category in categories.items()]


def rag_result_to_record(
    case: RetrievalEvalCase,
    category: str,
    retrieval_profile: str,
    result: RAGResult,
    *,
    candidate_k: int | None = None,
    frozen_rewrite_source_files: Sequence[str] = (),
) -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": "generation-pilot-result-v1",
        "case_id": case.case_id,
        "category": category,
        "question": case.question,
        "expected_answer": case.expected_answer,
        "retrieval_profile": retrieval_profile,
        "retrieval_query": result.retrieval_query,
        "rewrite_query": result.rewrite_query,
        "query_rewrite_status": result.query_rewrite_status,
        "required_constraints": list(result.required_constraints),
        "missing_constraints": list(result.missing_constraints),
        "retrieved_results": [
            _retrieved_result_to_record(rank, document, score)
            for rank, (document, score) in enumerate(result.retrieved_results, start=1)
        ],
        "context": result.context,
        "answer": result.answer,
    }
    if candidate_k is not None:
        record["dense_candidate_k"] = candidate_k
    if frozen_rewrite_source_files:
        record["frozen_rewrite_source_files"] = list(frozen_rewrite_source_files)
    return record


def generation_failure_to_record(
    case: RetrievalEvalCase,
    category: str,
    retrieval_profile: str,
    error: Exception,
    *,
    candidate_k: int | None = None,
    frozen_rewrite_source_files: Sequence[str] = (),
) -> dict[str, object]:
    """Persist one failed generation call without treating it as an answer."""
    record: dict[str, object] = {
        "schema_version": "generation-pilot-failure-v1",
        "case_id": case.case_id,
        "category": category,
        "question": case.question,
        "expected_answer": case.expected_answer,
        "retrieval_profile": retrieval_profile,
        "generation_error_type": type(error).__name__,
        "generation_error_message": str(error),
    }
    if candidate_k is not None:
        record["dense_candidate_k"] = candidate_k
    if frozen_rewrite_source_files:
        record["frozen_rewrite_source_files"] = list(frozen_rewrite_source_files)
    return record


def write_pilot_details(path: str | Path, records: Sequence[Mapping[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False))
            output_file.write("\n")


def _retrieved_result_to_record(rank: int, document: object, score: float) -> dict[str, object]:
    metadata = getattr(document, "metadata")
    record: dict[str, object] = {
        "rank": rank,
        "score": score,
        "source": metadata.get("source"),
        "section": metadata.get("section"),
        "subsection": metadata.get("subsection"),
        "page_content": getattr(document, "page_content"),
    }
    if "_dense_score" in metadata:
        record["dense_score"] = metadata["_dense_score"]
    if "_rerank_score" in metadata:
        record["rerank_score"] = metadata["_rerank_score"]
    return record


def _required_string(record: Mapping[str, object], field: str, line_number: int) -> str:
    value = record.get(field)
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError(f"Pilot line {line_number} requires non-empty {field}.")


def load_frozen_query_rewrites_from_sources(
    source_paths: Sequence[str | Path],
) -> dict[str, str]:
    """Merge frozen rewrites while rejecting ambiguous duplicate questions."""
    merged: dict[str, str] = {}
    for source_path in source_paths:
        for question, retrieval_query in load_frozen_query_rewrites(source_path).items():
            if question in merged:
                raise ValueError(
                    "Frozen rewrite sources repeat question: " f"{question!r}"
                )
            merged[question] = retrieval_query
    return merged


def validate_experiment_arguments(args: argparse.Namespace) -> None:
    if args.k > args.candidate_k:
        raise ValueError("--candidate-k must be greater than or equal to --k.")
    if (
        args.retrieval_profile == "dense_baseline"
        and args.candidate_k != DENSE_RERANK_CANDIDATE_K
    ):
        raise ValueError("--candidate-k only applies to the dense-rerank profile.")
    if args.retrieval_profile == "dense_baseline" and args.rewrite_source_details:
        raise ValueError(
            "--rewrite-source-details only applies to the dense-rerank profile."
        )


def main() -> None:
    args = parse_arguments()
    validate_experiment_arguments(args)
    categories = load_pilot_categories(args.pilot_file)
    selected_cases = select_pilot_cases(
        load_retrieval_eval_cases(args.eval_file), categories
    )
    frozen_rewrites = (
        load_frozen_query_rewrites_from_sources(args.rewrite_source_details)
        if args.rewrite_source_details
        else None
    )
    if frozen_rewrites is not None:
        missing_questions = [
            case.question for case, _category in selected_cases if case.question not in frozen_rewrites
        ]
        if missing_questions:
            raise ValueError(
                "Frozen rewrite sources do not cover selected questions: "
                + "; ".join(missing_questions)
            )
    pipeline = (
        build_basic_rag_pipeline(args.corpus_dir)
        if args.retrieval_profile == "dense_baseline"
        else build_dense_rerank_rag_pipeline(
            args.corpus_dir,
            candidate_k=args.candidate_k,
            query_rewriter=(FrozenQueryRewriter(frozen_rewrites) if frozen_rewrites else None),
        )
    )
    records = []
    failures = 0
    for index, (case, category) in enumerate(selected_cases, start=1):
        print(f"Running {index}/{len(selected_cases)}: {case.case_id}")
        candidate_k = (
            args.candidate_k
            if args.retrieval_profile == "dense_rerank_live_rewrite_guard"
            else None
        )
        frozen_rewrite_source_files = [
            str(path) for path in args.rewrite_source_details
        ]
        try:
            result = pipeline.answer(case.question, k=args.k)
        except Exception as error:  # API failures are experiment observations, not answers.
            failures += 1
            print(f"Generation failed for {case.case_id}: {type(error).__name__}: {error}")
            records.append(
                generation_failure_to_record(
                    case,
                    category,
                    args.retrieval_profile,
                    error,
                    candidate_k=candidate_k,
                    frozen_rewrite_source_files=frozen_rewrite_source_files,
                )
            )
            continue
        records.append(
            rag_result_to_record(
                case,
                category,
                args.retrieval_profile,
                result,
                candidate_k=candidate_k,
                frozen_rewrite_source_files=frozen_rewrite_source_files,
            )
        )
    write_pilot_details(args.details_file, records)
    print(f"Generation Pilot Cases: {len(records)}")
    print(f"Generation Failures: {failures}")
    print(f"Retrieval Profile: {args.retrieval_profile}")
    if args.retrieval_profile == "dense_rerank_live_rewrite_guard":
        print(f"Dense Candidate k: {args.candidate_k}")
    if args.rewrite_source_details:
        print("Rewrite Source Details: " + ", ".join(map(str, args.rewrite_source_details)))
    print(f"Details File: {args.details_file}")


if __name__ == "__main__":
    main()

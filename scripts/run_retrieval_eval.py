"""Run the V0 evidence-coverage evaluation against global dense retrieval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fiscal_rag.embeddings import QwenEmbeddings  # noqa: E402
from fiscal_rag.evaluation import (  # noqa: E402
    CaseEvaluation,
    RetrievalEvalSummary,
    evaluate_retrieval,
    load_retrieval_eval_cases,
    write_evaluation_details,
)
from fiscal_rag.ingestion import ingest_markdown_directory  # noqa: E402
from fiscal_rag.retrieval.dense import GlobalDenseRetriever  # noqa: E402
from fiscal_rag.vector_store import build_in_memory_vector_store  # noqa: E402


DEFAULT_EVAL_FILE = PROJECT_ROOT / "data_private" / "evals" / "pilot_retrieval_eval_v0.jsonl"
CORPUS_DIRECTORY = PROJECT_ROOT / "data_private" / "corpus"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V0 global dense retrieval evaluation.")
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=DEFAULT_EVAL_FILE,
        help="Private JSONL eval dataset path.",
    )
    parser.add_argument("--k", type=positive_integer, default=5, help="Retrieval depth.")
    parser.add_argument(
        "--details-file",
        type=Path,
        default=None,
        help="Optional private JSONL path for full per-case retrieval details.",
    )
    return parser.parse_args()


def positive_integer(value: str) -> int:
    integer = int(value)
    if integer <= 0:
        raise argparse.ArgumentTypeError("k must be positive")
    return integer


def print_case_evaluation(evaluation: CaseEvaluation) -> None:
    print(f"\nCase: {evaluation.case.case_id}")
    print(f"Question: {evaluation.case.question}")
    print(f"Coverage Status: {evaluation.coverage_status}")
    print(f"First Relevant Rank: {evaluation.first_relevant_rank}")
    print(f"Hit@1: {evaluation.hit_at_1}")
    print(f"Hit@3: {evaluation.hit_at_3}")
    print(f"Hit@5: {evaluation.hit_at_5}")
    print(f"RR: {evaluation.reciprocal_rank:.6f}")

    if evaluation.needs_diagnostics:
        print_failure_diagnostics(evaluation)


def print_failure_diagnostics(evaluation: CaseEvaluation) -> None:
    print("\nGround Truth")
    for evidence in evaluation.case.relevant_evidence:
        print(f"Source: {evidence.source}")
        print(f"Section: {evidence.section}")
        print(f"Subsection: {evidence.subsection}")
        print(f"Evidence Preview: {preview(evidence.evidence_text)}")

    print(f"\nActual Top-{len(evaluation.retrieved_results)}")
    for rank, (document, score) in enumerate(evaluation.retrieved_results, start=1):
        metadata = document.metadata
        print(f"\nRank {rank}")
        print(f"Score: {score:.6f}")
        print(f"Source: {metadata.get('source')}")
        print(f"Section: {metadata.get('section')}")
        print(f"Subsection: {metadata.get('subsection')}")
        print("Content Preview:")
        print(preview(document.page_content))


def print_summary(summary: RetrievalEvalSummary) -> None:
    print("\nSummary")
    print(f"Total Cases: {summary.total_cases}")
    print(f"Coverage Failures: {summary.coverage_failures}")
    print(f"Hit@1: {summary.hit_at_1:.6f}")
    print(f"Hit@3: {summary.hit_at_3:.6f}")
    print(f"Hit@5: {summary.hit_at_5:.6f}")
    print(f"MRR: {summary.mrr:.6f}")
    print(f"Retrieval Misses: {summary.retrieval_misses}")


def preview(text: str, limit: int = 300) -> str:
    return text[:limit] + ("..." if len(text) > limit else "")


def main() -> None:
    args = parse_arguments()
    cases = load_retrieval_eval_cases(args.eval_file)
    raw_documents, chunks = ingest_markdown_directory(CORPUS_DIRECTORY)
    vector_store = build_in_memory_vector_store(chunks, QwenEmbeddings())
    retriever = GlobalDenseRetriever(vector_store)
    summary = evaluate_retrieval(cases, chunks, retriever, k=args.k)
    if args.details_file is not None:
        write_evaluation_details(args.details_file, summary.case_evaluations)

    print(f"Eval File: {args.eval_file}")
    print(f"Markdown Files: {len(raw_documents)}")
    print(f"Chunks Indexed: {len(chunks)}")
    print(f"Retrieval k: {args.k}")
    if args.details_file is not None:
        print(f"Details File: {args.details_file}")
    for evaluation in summary.case_evaluations:
        print_case_evaluation(evaluation)
    print_summary(summary)


if __name__ == "__main__":
    main()

"""Evaluate a Dense candidate pool plus qwen3-rerank Top-5 experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fiscal_rag.embeddings import QwenEmbeddings  # noqa: E402
from fiscal_rag.evaluation import (  # noqa: E402
    CaseEvaluation,
    RetrievalEvalSummary,
    evaluate_retrieval,
    load_retrieval_eval_cases,
    write_evaluation_details,
)
from fiscal_rag.ingestion import ingest_markdown_directory  # noqa: E402
from fiscal_rag.query_rewrite import (  # noqa: E402
    CONSERVATIVE_DEEPSEEK_PROFILE,
    CONSERVATIVE_REWRITE_PROMPT,
    GUARDED_CONSERVATIVE_DEEPSEEK_PROFILE,
    ConstraintPreservationGuard,
    DeepSeekQueryRewriter,
    FrozenQueryRewriter,
    QueryRewriteRetriever,
    load_frozen_query_rewrites,
)
from fiscal_rag.reranker import (  # noqa: E402
    RERANK_INSTRUCTION_PROFILES,
    DashScopeReranker,
)
from fiscal_rag.retrieval.dense import GlobalDenseRetriever  # noqa: E402
from fiscal_rag.retrieval.dense_rerank import (  # noqa: E402
    PAGE_CONTENT_DOCUMENT_PROFILE,
    RERANK_DOCUMENT_PROFILES,
    DenseRerankRetriever,
)
from fiscal_rag.vector_store import build_in_memory_vector_store  # noqa: E402


DEFAULT_EVAL_FILE = PROJECT_ROOT / "data_private" / "evals" / "retrieval_eval_v1.jsonl"
CORPUS_DIRECTORY = PROJECT_ROOT / "data_private" / "corpus"
DENSE_CANDIDATE_K = 20
QUERY_REWRITE_PROFILES = (
    "none",
    CONSERVATIVE_DEEPSEEK_PROFILE,
    GUARDED_CONSERVATIVE_DEEPSEEK_PROFILE,
)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Dense candidates plus qwen3-rerank retrieval."
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=DEFAULT_EVAL_FILE,
        help="Private JSONL eval dataset path.",
    )
    parser.add_argument("--k", type=positive_integer, default=5, help="Output depth.")
    parser.add_argument(
        "--candidate-k",
        type=positive_integer,
        default=DENSE_CANDIDATE_K,
        help="Dense candidate count sent to the reranker (default: 20).",
    )
    parser.add_argument(
        "--instruction-profile",
        choices=sorted(RERANK_INSTRUCTION_PROFILES),
        default="default",
        help="Fixed reranker instruction profile for a reproducible experiment.",
    )
    parser.add_argument(
        "--query-rewrite-profile",
        choices=QUERY_REWRITE_PROFILES,
        default="none",
        help="Optional query rewrite profile for an isolated retrieval experiment.",
    )
    parser.add_argument(
        "--rewrite-source-details",
        type=Path,
        default=None,
        help="Private details JSONL that freezes rewrites for the guarded experiment.",
    )
    parser.add_argument(
        "--rerank-document-profile",
        choices=RERANK_DOCUMENT_PROFILES,
        default=PAGE_CONTENT_DOCUMENT_PROFILE,
        help="Candidate text representation sent to the reranker.",
    )
    parser.add_argument(
        "--details-file",
        type=Path,
        default=None,
        help="Optional private JSONL path for full per-case retrieval details.",
    )
    return parser.parse_args(argv)


def positive_integer(value: str) -> int:
    integer = int(value)
    if integer <= 0:
        raise argparse.ArgumentTypeError("k must be positive")
    return integer


def validate_candidate_k(output_k: int, candidate_k: int) -> None:
    if output_k > candidate_k:
        raise ValueError(
            f"--k cannot exceed the Dense Top-{candidate_k} candidate pool."
        )


def validate_experiment_profiles(
    instruction_profile: str,
    query_rewrite_profile: str,
    rewrite_source_details: Path | None,
) -> None:
    if (
        query_rewrite_profile != "none"
        and instruction_profile != "default"
    ):
        raise ValueError(
            "Query rewrite experiments require --instruction-profile default "
            "to keep a single retrieval variable."
        )
    if (
        query_rewrite_profile == GUARDED_CONSERVATIVE_DEEPSEEK_PROFILE
        and rewrite_source_details is None
    ):
        raise ValueError(
            "Guarded query rewrite requires --rewrite-source-details so the "
            "previous rewrites are frozen."
        )
    if (
        query_rewrite_profile != GUARDED_CONSERVATIVE_DEEPSEEK_PROFILE
        and rewrite_source_details is not None
    ):
        raise ValueError(
            "--rewrite-source-details is only valid for the guarded query rewrite "
            "experiment."
        )


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
        print(f"Rerank Score: {score:.6f}")
        print(f"Dense Score: {metadata.get('_dense_score'):.6f}")
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
    validate_experiment_profiles(
        args.instruction_profile,
        args.query_rewrite_profile,
        args.rewrite_source_details,
    )
    validate_candidate_k(args.k, args.candidate_k)

    instruction = RERANK_INSTRUCTION_PROFILES[args.instruction_profile]
    # Validate reranker configuration before spending an embedding API call.
    reranker = DashScopeReranker(instruction=instruction)
    if args.query_rewrite_profile == CONSERVATIVE_DEEPSEEK_PROFILE:
        query_rewriter = DeepSeekQueryRewriter()
    elif args.query_rewrite_profile == GUARDED_CONSERVATIVE_DEEPSEEK_PROFILE:
        query_rewriter = FrozenQueryRewriter(
            load_frozen_query_rewrites(args.rewrite_source_details)
        )
    else:
        query_rewriter = None
    cases = load_retrieval_eval_cases(args.eval_file)
    raw_documents, chunks = ingest_markdown_directory(CORPUS_DIRECTORY)
    vector_store = build_in_memory_vector_store(chunks, QwenEmbeddings())
    dense_retriever = GlobalDenseRetriever(vector_store)
    retriever = DenseRerankRetriever(
        dense_retriever,
        reranker,
        candidate_k=args.candidate_k,
        document_profile=args.rerank_document_profile,
    )
    rewrite_retriever: QueryRewriteRetriever | None = None
    if query_rewriter is not None:
        guard = (
            ConstraintPreservationGuard()
            if args.query_rewrite_profile == GUARDED_CONSERVATIVE_DEEPSEEK_PROFILE
            else None
        )
        rewrite_retriever = QueryRewriteRetriever(
            retriever, query_rewriter, guard=guard
        )
        active_retriever = rewrite_retriever
    else:
        active_retriever = retriever
    summary = evaluate_retrieval(cases, chunks, active_retriever, k=args.k)
    if args.details_file is not None:
        retrieval_queries = (
            None
            if rewrite_retriever is None
            else {
                evaluation.case.case_id: rewrite_retriever.retrieval_query_for(
                    evaluation.case.question
                )
                for evaluation in summary.case_evaluations
            }
        )
        query_rewrite_records = (
            None
            if args.query_rewrite_profile != GUARDED_CONSERVATIVE_DEEPSEEK_PROFILE
            else {
                evaluation.case.case_id: rewrite_retriever.rewrite_decision_for(
                    evaluation.case.question
                ).to_record()
                for evaluation in summary.case_evaluations
            }
        )
        dense_candidate_results = {
            evaluation.case.case_id: retriever.dense_candidates_for(
                evaluation.case.question
                if retrieval_queries is None
                else retrieval_queries[evaluation.case.case_id]
            )
            for evaluation in summary.case_evaluations
        }
        write_evaluation_details(
            args.details_file,
            summary.case_evaluations,
            retrieval_queries=retrieval_queries,
            query_rewrite_records=query_rewrite_records,
            dense_candidate_results=dense_candidate_results,
            detail_fields={
                "rerank_document_profile": args.rerank_document_profile,
                "dense_candidate_k": args.candidate_k,
            },
        )

    print("Experiment: Global Dense candidates + qwen3-rerank")
    print(f"Instruction Profile: {args.instruction_profile}")
    print(f"Reranker Instruction: {instruction}")
    print(f"Query Rewrite Profile: {args.query_rewrite_profile}")
    print(f"Rerank Document Profile: {args.rerank_document_profile}")
    if args.query_rewrite_profile != "none":
        print(f"Query Rewrite Prompt: {CONSERVATIVE_REWRITE_PROMPT}")
    if args.query_rewrite_profile == GUARDED_CONSERVATIVE_DEEPSEEK_PROFILE:
        print("Query Rewrite Guard: explicit role and state constraint preservation")
        print(f"Frozen Rewrite Source: {args.rewrite_source_details}")
    print(f"Eval File: {args.eval_file}")
    print(f"Markdown Files: {len(raw_documents)}")
    print(f"Chunks Indexed: {len(chunks)}")
    print(f"Dense Candidate k: {args.candidate_k}")
    print(f"Output k: {args.k}")
    if args.details_file is not None:
        print(f"Details File: {args.details_file}")
    for evaluation in summary.case_evaluations:
        print_case_evaluation(evaluation)
    print_summary(summary)


if __name__ == "__main__":
    main()

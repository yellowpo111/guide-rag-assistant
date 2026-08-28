"""Evaluate one Hybrid candidate-pool experiment before the unchanged reranker."""

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
    ConstraintPreservationGuard,
    FrozenQueryRewriter,
    QueryRewriteRetriever,
    load_frozen_query_rewrites,
)
from fiscal_rag.reranker import DashScopeReranker  # noqa: E402
from fiscal_rag.retrieval.bm25 import BM25Retriever  # noqa: E402
from fiscal_rag.retrieval.dense import GlobalDenseRetriever  # noqa: E402
from fiscal_rag.retrieval.hybrid_rerank import HybridRerankRetriever  # noqa: E402
from fiscal_rag.vector_store import build_in_memory_vector_store  # noqa: E402


DEFAULT_EVAL_FILE = PROJECT_ROOT / "data_private" / "evals" / "retrieval_eval_v1.jsonl"
CORPUS_DIRECTORY = PROJECT_ROOT / "data_private" / "corpus"
DENSE_CANDIDATE_K = 10
BM25_CANDIDATE_K = 10
RERANK_CANDIDATE_K = 10
RRF_K = 60


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Dense Top-10 + BM25 Top-10 RRF candidates plus qwen3-rerank."
    )
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--k", type=positive_integer, default=5)
    parser.add_argument("--details-file", type=Path, required=True)
    parser.add_argument(
        "--rewrite-source-details",
        type=Path,
        required=True,
        help="Frozen rewrite details from the default-reranker development baseline.",
    )
    parser.add_argument("--dense-candidate-k", type=positive_integer, default=DENSE_CANDIDATE_K)
    parser.add_argument("--bm25-candidate-k", type=positive_integer, default=BM25_CANDIDATE_K)
    parser.add_argument("--rerank-candidate-k", type=positive_integer, default=RERANK_CANDIDATE_K)
    parser.add_argument("--rrf-k", type=non_negative_integer, default=RRF_K)
    return parser.parse_args(argv)


def positive_integer(value: str) -> int:
    integer = int(value)
    if integer <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return integer


def non_negative_integer(value: str) -> int:
    integer = int(value)
    if integer < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return integer


def validate_arguments(args: argparse.Namespace) -> None:
    if args.k > args.rerank_candidate_k:
        raise ValueError("--k cannot exceed --rerank-candidate-k.")


def print_case_evaluation(evaluation: CaseEvaluation) -> None:
    print(f"\nCase: {evaluation.case.case_id}")
    print(f"Question: {evaluation.case.question}")
    print(f"Coverage Status: {evaluation.coverage_status}")
    print(f"First Relevant Rank: {evaluation.first_relevant_rank}")
    print(f"Hit@1: {evaluation.hit_at_1}")
    print(f"Hit@3: {evaluation.hit_at_3}")
    print(f"Hit@5: {evaluation.hit_at_5}")
    print(f"RR: {evaluation.reciprocal_rank:.6f}")


def print_summary(summary: RetrievalEvalSummary) -> None:
    print("\nSummary")
    print(f"Total Cases: {summary.total_cases}")
    print(f"Coverage Failures: {summary.coverage_failures}")
    print(f"Hit@1: {summary.hit_at_1:.6f}")
    print(f"Hit@3: {summary.hit_at_3:.6f}")
    print(f"Hit@5: {summary.hit_at_5:.6f}")
    print(f"MRR: {summary.mrr:.6f}")
    print(f"Retrieval Misses: {summary.retrieval_misses}")


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)
    cases = load_retrieval_eval_cases(args.eval_file)
    raw_documents, chunks = ingest_markdown_directory(CORPUS_DIRECTORY)
    vector_store = build_in_memory_vector_store(chunks, QwenEmbeddings())
    hybrid_retriever = HybridRerankRetriever(
        GlobalDenseRetriever(vector_store),
        BM25Retriever(chunks),
        DashScopeReranker(),
        dense_candidate_k=args.dense_candidate_k,
        bm25_candidate_k=args.bm25_candidate_k,
        rerank_candidate_k=args.rerank_candidate_k,
        rrf_k=args.rrf_k,
    )
    rewrite_retriever = QueryRewriteRetriever(
        hybrid_retriever,
        FrozenQueryRewriter(load_frozen_query_rewrites(args.rewrite_source_details)),
        guard=ConstraintPreservationGuard(),
    )
    summary = evaluate_retrieval(cases, chunks, rewrite_retriever, k=args.k)
    write_evaluation_details(
        args.details_file,
        summary.case_evaluations,
        retrieval_queries={
            evaluation.case.case_id: rewrite_retriever.retrieval_query_for(
                evaluation.case.question
            )
            for evaluation in summary.case_evaluations
        },
        query_rewrite_records={
            evaluation.case.case_id: rewrite_retriever.rewrite_decision_for(
                evaluation.case.question
            ).to_record()
            for evaluation in summary.case_evaluations
        },
        detail_fields={
            "candidate_strategy": "dense_top_n_plus_bm25_top_n_rrf_then_rerank",
            "dense_candidate_k": args.dense_candidate_k,
            "bm25_candidate_k": args.bm25_candidate_k,
            "rerank_candidate_k": args.rerank_candidate_k,
            "rrf_k": args.rrf_k,
        },
    )

    print("Experiment: Dense Top-N + BM25 Top-N -> RRF -> qwen3-rerank")
    print("Reranker Instruction Profile: default")
    print("Query Rewrite Profile: frozen conservative rewrite + Guard")
    print(f"Frozen Rewrite Source: {args.rewrite_source_details}")
    print(f"Eval File: {args.eval_file}")
    print(f"Markdown Files: {len(raw_documents)}")
    print(f"Chunks Indexed: {len(chunks)}")
    print(f"Dense Candidate k: {args.dense_candidate_k}")
    print(f"BM25 Candidate k: {args.bm25_candidate_k}")
    print(f"RRF k: {args.rrf_k}")
    print(f"Rerank Candidate k: {args.rerank_candidate_k}")
    print(f"Output k: {args.k}")
    print(f"Details File: {args.details_file}")
    for evaluation in summary.case_evaluations:
        print_case_evaluation(evaluation)
    print_summary(summary)


if __name__ == "__main__":
    main()

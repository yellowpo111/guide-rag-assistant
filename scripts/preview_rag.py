"""Local-only preview of the V0 retrieve-then-generate RAG baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIRECTORY = PROJECT_ROOT / "data_private" / "corpus"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fiscal_rag.pipeline import (  # noqa: E402
    build_basic_rag_pipeline,
    build_dense_rerank_rag_pipeline,
)


RETRIEVAL_PROFILES = ("dense_baseline", "dense_rerank_live_rewrite_guard")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview basic local RAG generation.")
    parser.add_argument("--query", required=True, help="Question for the RAG pipeline.")
    parser.add_argument("--k", type=positive_integer, default=5, help="Number of chunks.")
    parser.add_argument(
        "--retrieval-profile",
        choices=RETRIEVAL_PROFILES,
        default="dense_baseline",
        help="Dense baseline or the opt-in live Rewrite + Guard + Reranker preview.",
    )
    return parser.parse_args()


def positive_integer(value: str) -> int:
    integer = int(value)
    if integer <= 0:
        raise argparse.ArgumentTypeError("k must be positive")
    return integer


def main() -> None:
    args = parse_arguments()
    pipeline = (
        build_basic_rag_pipeline(CORPUS_DIRECTORY)
        if args.retrieval_profile == "dense_baseline"
        else build_dense_rerank_rag_pipeline(CORPUS_DIRECTORY)
    )
    result = pipeline.answer(args.query, k=args.k)

    print(f"Retrieval Profile: {args.retrieval_profile}")
    print(f"Question:\n{result.question}")
    print(f"Actual Retrieval Query: {result.retrieval_query}")
    if result.rewrite_query is not None:
        print(f"Raw Rewrite: {result.rewrite_query}")
        print(f"Guard Status: {result.query_rewrite_status}")
        print(f"Required Constraints: {list(result.required_constraints)}")
        print(f"Missing Constraints: {list(result.missing_constraints)}")
    print("\nRetrieved Results")
    for rank, (document, score) in enumerate(result.retrieved_results, start=1):
        metadata = document.metadata
        print(f"\nRank {rank}")
        print(f"Score: {score:.6f}")
        if "_dense_score" in metadata:
            print(f"Dense Score: {metadata['_dense_score']:.6f}")
        if "_rerank_score" in metadata:
            print(f"Rerank Score: {metadata['_rerank_score']:.6f}")
        print(f"Source: {metadata.get('source')}")
        print(f"Section: {metadata.get('section')}")
        print(f"Subsection: {metadata.get('subsection')}")
        print("\nContent Preview:")
        print(document.page_content[:500])

    print("\nFinal Answer")
    print(result.answer)


if __name__ == "__main__":
    main()

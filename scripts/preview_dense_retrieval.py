"""Local-only preview of the global dense retrieval baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIRECTORY = PROJECT_ROOT / "data_private" / "corpus"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fiscal_rag.embeddings import QwenEmbeddings  # noqa: E402
from fiscal_rag.ingestion import ingest_markdown_directory  # noqa: E402
from fiscal_rag.retrieval.dense import GlobalDenseRetriever  # noqa: E402
from fiscal_rag.vector_store import build_in_memory_vector_store  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview global dense retrieval over local Markdown guides."
    )
    parser.add_argument("--query", required=True, help="Query text to retrieve against.")
    parser.add_argument("--k", type=positive_integer, default=5, help="Number of results.")
    return parser.parse_args()


def positive_integer(value: str) -> int:
    integer = int(value)
    if integer <= 0:
        raise argparse.ArgumentTypeError("k must be positive")
    return integer


def main() -> None:
    args = parse_arguments()
    raw_documents, chunks = ingest_markdown_directory(CORPUS_DIRECTORY)
    embeddings = QwenEmbeddings()
    vector_store = build_in_memory_vector_store(chunks, embeddings)
    results = GlobalDenseRetriever(vector_store).retrieve(args.query, k=args.k)

    print(f"Markdown files read: {len(raw_documents)}")
    print(f"Chunks indexed: {len(chunks)}")
    print(f"Query: {args.query}")
    print(f"Top-{len(results)} results:")

    for rank, (document, score) in enumerate(results, start=1):
        metadata = document.metadata
        preview = document.page_content[:500]
        print(f"\nRank {rank}")
        print(f"Score: {score:.6f}")
        print(f"Source: {metadata.get('source')}")
        print(f"Section: {metadata.get('section')}")
        print(f"Subsection: {metadata.get('subsection')}")
        print("\nContent Preview:")
        print(preview)


if __name__ == "__main__":
    main()

"""Build or validate the private persistent Chroma index used by the CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_DIRECTORY = PROJECT_ROOT / "data_private" / "corpus"
DEFAULT_INDEX_DIRECTORY = (
    PROJECT_ROOT / "data_private" / "indexes" / "fiscal_guides_chroma_v1"
)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fiscal_rag.embeddings import QwenEmbeddings  # noqa: E402
from fiscal_rag.vector_store import (  # noqa: E402
    PersistentIndexError,
    build_persistent_chroma_index,
)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the private persistent Chroma index for Fiscal RAG."
    )
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIRECTORY)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIRECTORY)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Explicitly replace a stale or existing derived index.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_arguments()
    try:
        result = build_persistent_chroma_index(
            args.corpus_dir,
            args.index_dir,
            QwenEmbeddings(),
            rebuild=args.rebuild,
        )
    except PersistentIndexError as error:
        print(f"Index build failed: {error}")
        raise SystemExit(1) from error

    if result.created and args.rebuild:
        action = "Index rebuilt, validated, and activated"
    elif result.created:
        action = "Index created, validated, and activated"
    else:
        action = "Index already current"
    print(action)
    print(f"Corpus directory: {args.corpus_dir}")
    print(f"Index directory: {result.index_dir}")
    print(f"Embedding model: {result.manifest['embedding_model']}")
    print(f"Embedding dimension: {result.manifest['embedding_dimension']}")
    print(f"Markdown files: {result.raw_document_count}")
    print(f"Chunks indexed: {result.chunk_count}")
    if result.created:
        print("Validation: manifest, source coverage, stored chunks, dimensions, retrieval smoke")


if __name__ == "__main__":
    main()

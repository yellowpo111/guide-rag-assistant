"""Load and split local Markdown guides for the retrieval experiment."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100
_HEADERS_TO_SPLIT_ON = [("#", "section"), ("##", "subsection")]


def load_markdown_documents(data_dir: str | Path) -> list[Document]:
    """Read every Markdown file below ``data_dir`` as one raw document.

    The source metadata is always relative to ``data_dir`` so local absolute
    paths are never carried into chunks or later experiment outputs.
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Markdown directory does not exist: {root}")

    markdown_paths = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".md"),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    return [
        Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={"source": path.relative_to(root).as_posix()},
        )
        for path in markdown_paths
    ]


def split_markdown_documents(
    raw_documents: Sequence[Document],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split Markdown by headings first, then recursively split long sections."""
    _validate_chunk_parameters(chunk_size, chunk_overlap)

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    structured_documents: list[Document] = []

    for raw_document in raw_documents:
        for section_document in header_splitter.split_text(raw_document.page_content):
            structured_documents.append(
                Document(
                    page_content=section_document.page_content,
                    metadata={**raw_document.metadata, **section_document.metadata},
                )
            )

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return recursive_splitter.split_documents(structured_documents)


def ingest_markdown_directory(
    data_dir: str | Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[list[Document], list[Document]]:
    """Load a Markdown directory and return its raw documents and final chunks."""
    raw_documents = load_markdown_documents(data_dir)
    chunks = split_markdown_documents(
        raw_documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return raw_documents, chunks


def print_ingestion_preview(
    raw_documents: Sequence[Document],
    chunks: Sequence[Document],
    *,
    preview_count: int = 3,
    preview_characters: int = 240,
) -> None:
    """Print a compact, local-only summary for manual ingestion inspection."""
    print(f"Markdown files read: {len(raw_documents)}")
    print(f"Chunks created: {len(chunks)}")

    for index, chunk in enumerate(chunks[:preview_count], start=1):
        print(f"\nChunk {index}")
        print(f"  source: {chunk.metadata.get('source')}")
        print(f"  section: {chunk.metadata.get('section')}")
        print(f"  subsection: {chunk.metadata.get('subsection')}")
        preview = chunk.page_content[:preview_characters].replace("\n", " ")
        print(f"  page_content preview: {preview}")


def _validate_chunk_parameters(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")


def main() -> None:
    """Run a local-only preview against the private company-guide corpus."""
    project_root = Path(__file__).resolve().parents[2]
    raw_documents, chunks = ingest_markdown_directory(
        project_root / "data_private" / "corpus"
    )
    print_ingestion_preview(raw_documents, chunks)


if __name__ == "__main__":
    main()

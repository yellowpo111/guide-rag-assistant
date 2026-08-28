from pathlib import Path
import sys

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fiscal_rag.vector_store import build_in_memory_vector_store  # noqa: E402


class DirectionEmbeddings(Embeddings):
    """Deterministic vectors for local vector store tests."""

    _vectors = {
        "alpha document": [1.0, 0.0],
        "beta document": [0.0, 1.0],
        "query alpha": [1.0, 0.0],
    }

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectors[text]


def test_build_in_memory_vector_store_preserves_document_metadata() -> None:
    documents = [
        Document(
            page_content="alpha document",
            metadata={
                "source": "alpha.md",
                "section": "Alpha",
                "subsection": "Setup",
            },
        ),
        Document(
            page_content="beta document",
            metadata={
                "source": "beta.md",
                "section": "Beta",
                "subsection": "Usage",
            },
        ),
    ]

    vector_store = build_in_memory_vector_store(documents, DirectionEmbeddings())
    result, score = vector_store.similarity_search_with_score("query alpha", k=1)[0]

    assert result.page_content == "alpha document"
    assert result.metadata == documents[0].metadata
    assert isinstance(score, float)

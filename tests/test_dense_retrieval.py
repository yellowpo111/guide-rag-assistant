from pathlib import Path
import sys

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fiscal_rag.retrieval.dense import GlobalDenseRetriever  # noqa: E402
from fiscal_rag.vector_store import build_in_memory_vector_store  # noqa: E402


class DirectionEmbeddings(Embeddings):
    """Deterministic vectors whose cosine similarity produces a known ranking."""

    _vectors = {
        "alpha document": [1.0, 0.0],
        "beta document": [0.0, 1.0],
        "gamma document": [-1.0, 0.0],
        "query alpha": [1.0, 0.0],
    }

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectors[text]


def make_retriever() -> GlobalDenseRetriever:
    documents = [
        Document(
            page_content="alpha document",
            metadata={"source": "alpha.md", "section": "Alpha", "subsection": "One"},
        ),
        Document(
            page_content="beta document",
            metadata={"source": "beta.md", "section": "Beta", "subsection": "Two"},
        ),
        Document(
            page_content="gamma document",
            metadata={"source": "gamma.md", "section": "Gamma", "subsection": "Three"},
        ),
    ]
    return GlobalDenseRetriever(
        build_in_memory_vector_store(documents, DirectionEmbeddings())
    )


def test_retrieve_returns_requested_top_k_in_similarity_order() -> None:
    retriever = make_retriever()

    top_one = retriever.retrieve("query alpha", k=1)
    top_two = retriever.retrieve("query alpha", k=2)

    assert len(top_one) == 1
    assert len(top_two) == 2
    assert [document.metadata["source"] for document, _ in top_two] == [
        "alpha.md",
        "beta.md",
    ]
    assert top_two[0][1] > top_two[1][1]


def test_retrieve_returns_scores_and_preserves_all_metadata() -> None:
    document, score = make_retriever().retrieve("query alpha", k=1)[0]

    assert isinstance(score, float)
    assert document.metadata == {
        "source": "alpha.md",
        "section": "Alpha",
        "subsection": "One",
    }

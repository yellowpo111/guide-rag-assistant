"""Global dense retrieval over a store with similarity-search capability."""

from typing import Protocol

from langchain_core.documents import Document


class SimilaritySearchStore(Protocol):
    """Minimal store interface shared by in-memory and persistent indexes."""

    def similarity_search_with_score(
        self, query: str, *, k: int = 4
    ) -> list[tuple[Document, float]]: ...


class GlobalDenseRetriever:
    """Retrieve the top-k most similar documents without metadata filtering."""

    def __init__(self, vector_store: SimilaritySearchStore) -> None:
        self._vector_store = vector_store

    def retrieve(self, query: str, *, k: int = 5) -> list[tuple[Document, float]]:
        """Return globally ranked ``(Document, similarity_score)`` pairs."""
        if k <= 0:
            raise ValueError("k must be positive")
        return self._vector_store.similarity_search_with_score(query, k=k)

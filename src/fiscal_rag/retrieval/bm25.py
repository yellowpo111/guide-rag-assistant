"""Transparent lexical BM25 retrieval for the Hybrid experiment."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


Tokenizer = Callable[[str], list[str]]


def tokenize_chinese_text(text: str) -> list[str]:
    """Tokenize Chinese as characters and Latin/digits as words.

    Chinese Markdown normally has no whitespace between words. Character tokens
    make terms such as ``账号解冻`` and ``指标解冻`` distinguishable without
    introducing a second segmentation-model dependency into this experiment.
    """
    return re.findall(r"[A-Za-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff]", text.lower())


class BM25Retriever:
    """Global BM25 ranking over the same chunks used by dense retrieval."""

    def __init__(
        self,
        documents: Sequence[Document],
        *,
        tokenizer: Tokenizer = tokenize_chinese_text,
    ) -> None:
        if not documents:
            raise ValueError("BM25Retriever requires at least one document.")
        self._documents = list(documents)
        self._tokenizer = tokenizer
        self._index = BM25Okapi(
            [self._tokenizer(document.page_content) for document in self._documents]
        )

    def retrieve(self, query: str, *, k: int = 5) -> list[tuple[Document, float]]:
        """Return globally ranked ``(Document, BM25_score)`` pairs."""
        if k <= 0:
            raise ValueError("k must be positive")
        scores = self._index.get_scores(self._tokenizer(query))
        ranked_indices = sorted(
            range(len(self._documents)), key=lambda index: (-float(scores[index]), index)
        )
        return [
            (self._documents[index], float(scores[index]))
            for index in ranked_indices[:k]
        ]

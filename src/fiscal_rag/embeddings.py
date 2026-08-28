"""DashScope Qwen embedding adapter behind LangChain's Embeddings interface."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from openai import OpenAI

from fiscal_rag.settings import non_negative_integer_setting, positive_float_setting


MAX_EMBEDDING_BATCH_SIZE = 10
DEFAULT_EMBEDDING_TIMEOUT_SECONDS = 60.0
DEFAULT_EMBEDDING_MAX_RETRIES = 0


class QwenEmbeddings(Embeddings):
    """Create Qwen text embeddings through DashScope's OpenAI-compatible API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        client: Any | None = None,
    ) -> None:
        _load_project_env()

        self.api_key = _required_setting("DASHSCOPE_API_KEY", api_key)
        self.base_url = _required_setting("DASHSCOPE_BASE_URL", base_url)
        self.model = _required_setting("DASHSCOPE_EMBEDDING_MODEL", model)
        self.timeout_seconds = positive_float_setting(
            "DASHSCOPE_EMBEDDING_TIMEOUT_SECONDS",
            DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
        )
        self.max_retries = non_negative_integer_setting(
            "DASHSCOPE_EMBEDDING_MAX_RETRIES",
            DEFAULT_EMBEDDING_MAX_RETRIES,
        )
        self._client = client or OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            max_retries=self.max_retries,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents in API-sized batches while preserving input order."""
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), MAX_EMBEDDING_BATCH_SIZE):
            batch = texts[start : start + MAX_EMBEDDING_BATCH_SIZE]
            response = self._client.embeddings.create(model=self.model, input=batch)
            vectors.extend(
                [float(value) for value in item.embedding] for item in response.data
            )

        if len(vectors) != len(texts):
            raise RuntimeError(
                "Embedding API returned a different number of vectors than input texts."
            )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed one query string."""
        vectors = self.embed_documents([text])
        if not vectors:
            raise RuntimeError("Embedding API returned no vector for the query.")
        return vectors[0]


def _load_project_env() -> None:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env", override=False)


def _required_setting(name: str, explicit_value: str | None) -> str:
    value = explicit_value if explicit_value is not None else os.getenv(name)
    if value:
        return value
    raise ValueError(
        f"Missing required DashScope embedding configuration: {name}. "
        "Set it in .env or as an environment variable."
    )


def main() -> None:
    """Run a minimal real API smoke test using only non-sensitive example text."""
    embeddings = QwenEmbeddings()
    query_vector = embeddings.embed_query("单位基础信息如何填写？")
    document_vectors = embeddings.embed_documents(
        ["单位基础信息填写说明", "数据填报保存操作"]
    )

    print(f"embed_query vector dimension: {len(query_vector)}")
    print(f"embed_documents vector count: {len(document_vectors)}")
    print(
        "embed_documents vector dimensions: "
        + ", ".join(str(len(vector)) for vector in document_vectors)
    )


if __name__ == "__main__":
    main()

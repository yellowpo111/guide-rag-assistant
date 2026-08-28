"""DashScope qwen3-rerank adapter for the retrieval experiments."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from fiscal_rag.settings import non_negative_integer_setting, positive_float_setting


DEFAULT_RERANK_MODEL = "qwen3-rerank"
DEFAULT_RERANK_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query."
)
FISCAL_OPERATION_RERANK_INSTRUCTION = (
    "Given a fiscal software operation question, rank passages by their ability "
    "to directly and completely answer the requested information. When the "
    "question asks for a procedure, prioritize the correct business variant and "
    "concrete role, menu path, buttons, fields, prerequisites, and next steps. "
    "For factual questions, prioritize a passage that explicitly states the "
    "answer. Deprioritize generic introductions, broad permission summaries, "
    "and similar but different workflows."
)
RERANK_INSTRUCTION_PROFILES = {
    "default": DEFAULT_RERANK_INSTRUCTION,
    "fiscal_operation": FISCAL_OPERATION_RERANK_INSTRUCTION,
}
MAX_RERANK_DOCUMENTS = 500
DEFAULT_RERANK_TIMEOUT_SECONDS = 60.0
DEFAULT_RERANK_MAX_RETRIES = 0


@dataclass(frozen=True)
class RerankResult:
    """One score returned by the rerank API for an input document index."""

    index: int
    relevance_score: float


class RerankRequestError(RuntimeError):
    """A sanitized rerank transport failure with retry eligibility."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


PostJson = Callable[[str, dict[str, str], dict[str, object]], object]


class DashScopeReranker:
    """Call DashScope's qwen3-rerank HTTP endpoint with an explicit endpoint."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        instruction: str | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
        post_json: PostJson | None = None,
    ) -> None:
        _load_project_env()

        self.api_key = _required_setting("DASHSCOPE_API_KEY", api_key)
        self.base_url = _required_setting("DASHSCOPE_RERANK_BASE_URL", base_url)
        self.model = model or os.getenv(
            "DASHSCOPE_RERANK_MODEL", DEFAULT_RERANK_MODEL
        )
        self.instruction = instruction or DEFAULT_RERANK_INSTRUCTION
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else positive_float_setting(
                "DASHSCOPE_RERANK_TIMEOUT_SECONDS", DEFAULT_RERANK_TIMEOUT_SECONDS
            )
        )
        self.max_retries = (
            max_retries
            if max_retries is not None
            else non_negative_integer_setting(
                "DASHSCOPE_RERANK_MAX_RETRIES", DEFAULT_RERANK_MAX_RETRIES
            )
        )
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self._post_json = post_json or (
            lambda endpoint, headers, payload: _post_json(
                endpoint,
                headers,
                payload,
                timeout_seconds=self.timeout_seconds,
            )
        )

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_n: int,
    ) -> list[RerankResult]:
        """Return API-ranked candidate indexes and their relative relevance scores."""
        if not documents:
            return []
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        if len(documents) > MAX_RERANK_DOCUMENTS:
            raise ValueError(
                f"qwen3-rerank accepts at most {MAX_RERANK_DOCUMENTS} documents per request."
            )

        payload: dict[str, object] = {
            "model": self.model,
            "documents": list(documents),
            "query": query,
            "top_n": top_n,
            "instruct": self.instruction,
        }
        response = self._request_with_retries(payload)
        return _parse_results(response, document_count=len(documents), top_n=top_n)

    def _request_with_retries(self, payload: dict[str, object]) -> object:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(self.max_retries + 1):
            try:
                return self._post_json(
                    _rerank_endpoint(self.base_url), headers, payload
                )
            except RerankRequestError as error:
                if not error.retryable or attempt >= self.max_retries:
                    raise
                time.sleep(min(2**attempt, 5))
        raise RuntimeError("Rerank request retry loop exited unexpectedly.")


def _load_project_env() -> None:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env", override=False)


def _required_setting(name: str, explicit_value: str | None) -> str:
    value = explicit_value if explicit_value is not None else os.getenv(name)
    if value:
        return value
    raise ValueError(
        f"Missing required DashScope rerank configuration: {name}. "
        "Set it in .env or as an environment variable."
    )


def _rerank_endpoint(base_url: str) -> str:
    """Accept either the compatible base URL or the full reranks endpoint."""
    normalized = base_url.rstrip("/")
    return normalized if normalized.endswith("/reranks") else f"{normalized}/reranks"


def _post_json(
    endpoint: str,
    headers: dict[str, str],
    payload: dict[str, object],
    *,
    timeout_seconds: float,
) -> object:
    request = Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(
            request, timeout=timeout_seconds
        ) as response:  # noqa: S310 - configured API URL
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RerankRequestError(
            f"DashScope rerank API request failed with HTTP {error.code}: {body}",
            retryable=error.code == 429 or error.code >= 500,
        ) from error
    except URLError as error:
        raise RerankRequestError(
            f"DashScope rerank API request could not reach endpoint: {error}",
            retryable=True,
        ) from error
    except json.JSONDecodeError as error:
        raise RerankRequestError(
            "DashScope rerank API returned invalid JSON.", retryable=False
        ) from error


def _parse_results(
    response: object, *, document_count: int, top_n: int
) -> list[RerankResult]:
    if not isinstance(response, Mapping):
        raise RuntimeError("DashScope rerank API returned a non-object response.")

    results = response.get("results")
    if not isinstance(results, list):
        code = response.get("code")
        message = response.get("message")
        if isinstance(code, str) or isinstance(message, str):
            raise RuntimeError(f"DashScope rerank API error ({code}): {message}")
        raise RuntimeError("DashScope rerank API response is missing a results list.")

    expected_count = min(top_n, document_count)
    if len(results) != expected_count:
        raise RuntimeError(
            "DashScope rerank API returned a different number of results than requested."
        )

    parsed_results: list[RerankResult] = []
    seen_indexes: set[int] = set()
    for result in results:
        if not isinstance(result, Mapping):
            raise RuntimeError("DashScope rerank API returned a malformed result item.")
        index = result.get("index")
        relevance_score = result.get("relevance_score")
        if isinstance(index, bool) or not isinstance(index, int):
            raise RuntimeError("DashScope rerank API result is missing an integer index.")
        if index < 0 or index >= document_count or index in seen_indexes:
            raise RuntimeError("DashScope rerank API returned an invalid document index.")
        if isinstance(relevance_score, bool) or not isinstance(
            relevance_score, (int, float)
        ):
            raise RuntimeError("DashScope rerank API result is missing a numeric relevance_score.")
        seen_indexes.add(index)
        parsed_results.append(
            RerankResult(index=index, relevance_score=float(relevance_score))
        )
    return parsed_results


def main() -> None:
    """Run an independent smoke test with non-sensitive example text."""
    reranker = DashScopeReranker()
    documents = ["单位基础信息填写说明", "数据填报保存操作"]
    results = reranker.rerank("单位基础信息如何填写？", documents, top_n=2)

    print(f"rerank result count: {len(results)}")
    print(
        "rerank document indexes: "
        + ", ".join(str(result.index) for result in results)
    )
    print(
        "rerank relevance scores: "
        + ", ".join(f"{result.relevance_score:.6f}" for result in results)
    )


if __name__ == "__main__":
    main()

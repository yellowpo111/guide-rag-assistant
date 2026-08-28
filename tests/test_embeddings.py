from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fiscal_rag.embeddings as embeddings_module  # noqa: E402
from fiscal_rag.embeddings import QwenEmbeddings  # noqa: E402


class FakeEmbeddingsEndpoint:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
        self.calls.append((model, input))
        vectors_by_text = {
            "query": (0, 0.5),
            "first": (1, 1.5),
            "second": (2, 2.5),
        }
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=vectors_by_text[text])
                for text in input
            ]
        )


class FakeClient:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddingsEndpoint()


def make_adapter(client: FakeClient) -> QwenEmbeddings:
    return QwenEmbeddings(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="fake-embedding-model",
        client=client,
    )


def test_embed_query_returns_a_float_vector() -> None:
    client = FakeClient()

    vector = make_adapter(client).embed_query("query")

    assert vector == [0.0, 0.5]
    assert all(isinstance(value, float) for value in vector)
    assert client.embeddings.calls == [("fake-embedding-model", ["query"])]


def test_embed_documents_returns_vectors_in_input_order() -> None:
    client = FakeClient()

    vectors = make_adapter(client).embed_documents(["first", "second"])

    assert vectors == [[1.0, 1.5], [2.0, 2.5]]
    assert client.embeddings.calls == [
        ("fake-embedding-model", ["first", "second"])
    ]


def test_embed_documents_splits_large_inputs_into_api_sized_batches() -> None:
    client = FakeClient()
    texts = ["first"] * 10 + ["second"] * 2

    vectors = make_adapter(client).embed_documents(texts)

    assert len(vectors) == 12
    assert client.embeddings.calls == [
        ("fake-embedding-model", ["first"] * 10),
        ("fake-embedding-model", ["second"] * 2),
    ]


def test_missing_required_environment_setting_has_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "DASHSCOPE_API_KEY",
        "DASHSCOPE_BASE_URL",
        "DASHSCOPE_EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(embeddings_module, "load_dotenv", lambda *args, **kwargs: False)

    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        QwenEmbeddings(client=FakeClient())


def test_default_client_uses_explicit_timeout_and_zero_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeConfiguredClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(embeddings_module, "OpenAI", FakeConfiguredClient)
    monkeypatch.setenv("DASHSCOPE_EMBEDDING_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("DASHSCOPE_EMBEDDING_MAX_RETRIES", "0")

    QwenEmbeddings(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="test-model",
    )

    assert captured["timeout"] == 45.0
    assert captured["max_retries"] == 0

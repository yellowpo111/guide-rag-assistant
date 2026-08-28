from pathlib import Path
from types import SimpleNamespace
import sys

from langchain_core.documents import Document


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fiscal_rag.assistant import (  # noqa: E402
    CHAT_PROMPT,
    OUT_OF_SCOPE_RESPONSE,
    ROUTER_PROMPT,
    AssistantRouter,
    AssistantWorkflow,
)
from fiscal_rag.pipeline import PreparedRAGRequest  # noqa: E402


class FakeChatModel:
    def __init__(
        self,
        *,
        invoke_content: str = "rag",
        stream_chunks: tuple[str, ...] = ("你", "好"),
        invoke_error: Exception | None = None,
    ) -> None:
        self.invoke_content = invoke_content
        self.stream_chunks = stream_chunks
        self.invoke_error = invoke_error
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        if self.invoke_error is not None:
            raise self.invoke_error
        return SimpleNamespace(content=self.invoke_content)

    def stream(self, prompt: str):
        self.prompts.append(prompt)
        for chunk in self.stream_chunks:
            yield SimpleNamespace(content=chunk)


class FakeRAGPipeline:
    def __init__(self) -> None:
        self.prepare_calls: list[tuple[str, int]] = []
        self.stream_calls: list[PreparedRAGRequest] = []

    def prepare(self, question: str, *, k: int = 5) -> PreparedRAGRequest:
        self.prepare_calls.append((question, k))
        return PreparedRAGRequest(
            question=question,
            retrieved_results=[
                (Document(page_content="private", metadata={"source": "guide.md"}), 0.9)
            ],
            context="private context",
            prompt="rag prompt",
            retrieval_query="rewritten query",
        )

    def stream_prepared(self, prepared: PreparedRAGRequest):
        self.stream_calls.append(prepared)
        yield "知识"
        yield "回答"


def test_router_accepts_only_supported_exact_labels() -> None:
    for label in ("rag", "chat", "out_of_scope"):
        model = FakeChatModel(invoke_content=label)

        assert AssistantRouter(model).route("测试") == label
        assert "含问候但同时提出此类问题时也选 rag" in model.prompts[0]
        assert ROUTER_PROMPT.format(question="测试") == model.prompts[0]


def test_router_falls_back_to_rag_for_invalid_output_or_model_failure() -> None:
    assert AssistantRouter(FakeChatModel(invoke_content="chat。因为是问候")).route("你好") == "rag"
    assert AssistantRouter(
        FakeChatModel(invoke_error=RuntimeError("private upstream detail"))
    ).route("你好") == "rag"


def test_rag_route_reuses_prepared_existing_pipeline() -> None:
    rag = FakeRAGPipeline()
    workflow = AssistantWorkflow(
        rag,
        router=AssistantRouter(FakeChatModel(invoke_content="rag")),
        chat_model=FakeChatModel(),
    )

    prepared = workflow.prepare("你好，单位信息怎样填？", k=5)

    assert prepared.route == "rag"
    assert rag.prepare_calls == [("你好，单位信息怎样填？", 5)]
    assert list(workflow.stream_prepared(prepared)) == ["知识", "回答"]


def test_chat_route_streams_lightweight_assistant_prompt_without_rag() -> None:
    rag = FakeRAGPipeline()
    chat_model = FakeChatModel(stream_chunks=("你好", "，请问有什么财政业务问题？"))
    workflow = AssistantWorkflow(
        rag,
        router=AssistantRouter(FakeChatModel(invoke_content="chat")),
        chat_model=chat_model,
    )

    prepared = workflow.prepare("你好")

    assert prepared.route == "chat"
    assert rag.prepare_calls == []
    assert list(workflow.stream_prepared(prepared)) == [
        "你好",
        "，请问有什么财政业务问题？",
    ]
    assert chat_model.prompts == [CHAT_PROMPT.format(question="你好")]


def test_out_of_scope_route_uses_fixed_response_without_model_or_rag() -> None:
    rag = FakeRAGPipeline()
    chat_model = FakeChatModel()
    workflow = AssistantWorkflow(
        rag,
        router=AssistantRouter(FakeChatModel(invoke_content="out_of_scope")),
        chat_model=chat_model,
    )

    prepared = workflow.prepare("写一首诗")

    assert list(workflow.stream_prepared(prepared)) == [OUT_OF_SCOPE_RESPONSE]
    assert rag.prepare_calls == []
    assert chat_model.prompts == []

"""Single-turn routing workflow layered above the verified RAG pipeline."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from fiscal_rag.pipeline import (
    ChatModel,
    PreparedRAGRequest,
    create_deepseek_chat_model,
)
from fiscal_rag.timing import measure_stage


AssistantRoute = Literal["rag", "chat", "out_of_scope"]
LOGGER = logging.getLogger("fiscal_rag.assistant")
ROUTER_PROMPT = """你是财政业务知识助手的请求路由器。

只允许输出以下一个标签，不得输出标点、解释或其他文字：
- rag：财政、财政软件、内部业务系统、操作流程、菜单、字段、权限、审核、填报等知识问题；含问候但同时提出此类问题时也选 rag；不能确定时也选 rag。
- chat：只有问候、告别、致谢，或询问助手身份、能力和使用范围的轻量对话。
- out_of_scope：明确要求回答与财政业务和助手自身无关的知识、娱乐、创作或其他任务。

用户输入：
{question}
"""
CHAT_PROMPT = """你是公司内网中的财政业务知识助手。

只对用户的问候、告别、致谢或关于助手身份和能力的提问作简短、自然的中文回应。
可以说明你主要帮助查询财政业务和内部操作指南。
不要声称执行操作、调用工具或知道当前对话之外的信息，也不要扩展回答通用知识。

用户输入：
{question}

回答：
"""
OUT_OF_SCOPE_RESPONSE = (
    "我目前主要用于财政业务知识和内部操作指南问答，"
    "暂不处理与此无关的通用知识、娱乐或创作任务。"
)


class RAGStreamingPipeline(Protocol):
    def prepare(self, question: str, *, k: int = 5) -> PreparedRAGRequest: ...

    def stream_prepared(self, prepared: PreparedRAGRequest) -> Iterator[str]: ...


@dataclass(frozen=True)
class PreparedAssistantResponse:
    """One routed request ready for its final response stream."""

    route: AssistantRoute
    rag_request: PreparedRAGRequest | None = None
    chat_prompt: str | None = None
    fixed_response: str | None = None


class AssistantRouter:
    """Classify one request, failing closed to the existing RAG path."""

    def __init__(self, chat_model: ChatModel | None = None) -> None:
        self._chat_model = chat_model or create_deepseek_chat_model()

    def route(self, question: str) -> AssistantRoute:
        try:
            with measure_stage("router"):
                response = self._chat_model.invoke(ROUTER_PROMPT.format(question=question))
            content = getattr(response, "content", response)
            label = content.strip().lower() if isinstance(content, str) else ""
        except Exception as error:
            LOGGER.warning("assistant_route_failed error_type=%s", type(error).__name__)
            return "rag"
        if label in {"rag", "chat", "out_of_scope"}:
            return cast(AssistantRoute, label)
        LOGGER.warning("assistant_route_invalid")
        return "rag"


class AssistantWorkflow:
    """Route a single turn to RAG, lightweight chat, or a fixed boundary reply."""

    def __init__(
        self,
        rag_pipeline: RAGStreamingPipeline,
        *,
        router: AssistantRouter | None = None,
        chat_model: ChatModel | None = None,
    ) -> None:
        self._rag_pipeline = rag_pipeline
        self._router = router or AssistantRouter()
        self._chat_model = chat_model or create_deepseek_chat_model()

    def prepare(self, question: str, *, k: int = 5) -> PreparedAssistantResponse:
        route = self._router.route(question)
        if route == "rag":
            with measure_stage("rag_preparation"):
                return PreparedAssistantResponse(
                    route=route,
                    rag_request=self._rag_pipeline.prepare(question, k=k),
                )
        if route == "chat":
            return PreparedAssistantResponse(
                route=route,
                chat_prompt=CHAT_PROMPT.format(question=question),
            )
        return PreparedAssistantResponse(
            route=route,
            fixed_response=OUT_OF_SCOPE_RESPONSE,
        )

    def stream_prepared(self, prepared: PreparedAssistantResponse) -> Iterator[str]:
        if prepared.route == "rag":
            if prepared.rag_request is None:
                raise RuntimeError("RAG route is missing its prepared request.")
            yield from self._rag_pipeline.stream_prepared(prepared.rag_request)
            return
        if prepared.route == "out_of_scope":
            if prepared.fixed_response is None:
                raise RuntimeError("Out-of-scope route is missing its fixed response.")
            yield prepared.fixed_response
            return
        if prepared.chat_prompt is None:
            raise RuntimeError("Chat route is missing its prompt.")
        yield from _stream_chat_response(self._chat_model, prepared.chat_prompt)


def _stream_chat_response(chat_model: ChatModel, prompt: str) -> Iterator[str]:
    pieces: list[str] = []
    for response in chat_model.stream(prompt):
        content: Any = getattr(response, "content", response)
        text = content if isinstance(content, str) else str(content)
        if not text:
            continue
        pieces.append(text)
        yield text
    if not "".join(pieces).strip():
        raise ValueError("DeepSeek returned an empty streaming response.")

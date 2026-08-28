from pathlib import Path
from threading import Lock, Thread
from time import sleep
import sys

from langchain_core.documents import Document


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fiscal_rag.pipeline import RAGResult  # noqa: E402
from fiscal_rag.assistant import PreparedAssistantResponse  # noqa: E402
from fiscal_rag.pipeline import PreparedRAGRequest  # noqa: E402
from fiscal_rag.service import FINAL_CONTEXT_K, FiscalRAGService  # noqa: E402


def make_result(question: str = "怎样保存？") -> RAGResult:
    return RAGResult(
        question=question,
        retrieved_results=[
            (
                Document(
                    page_content="private chunk content",
                    metadata={
                        "source": "guide.md",
                        "section": "填报",
                        "subsection": "操作步骤",
                        "_dense_score": 0.75,
                        "_rerank_score": 0.0,
                    },
                ),
                0.5,
            )
        ],
        context="private full context",
        answer="点击保存。",
        retrieval_query="如何保存？",
        rewrite_query="如何保存？",
        query_rewrite_status="accepted",
        required_constraints=("建设单位",),
    )


def test_service_uses_fixed_top_five_and_returns_only_safe_citation_metadata() -> None:
    class FakePipeline:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        def answer(self, question: str, *, k: int) -> RAGResult:
            self.calls.append((question, k))
            return make_result(question)

    pipeline = FakePipeline()
    result = FiscalRAGService(pipeline).answer("怎样保存？")

    assert pipeline.calls == [("怎样保存？", FINAL_CONTEXT_K)]
    assert result.answer == "点击保存。"
    assert result.sources[0].source == "guide.md"
    assert result.sources[0].dense_score == 0.75
    assert result.sources[0].rerank_score == 0.0
    assert not hasattr(result, "context")
    assert not hasattr(result.sources[0], "page_content")


def test_service_serializes_pipeline_calls() -> None:
    class ConcurrentPipeline:
        def __init__(self) -> None:
            self.state_lock = Lock()
            self.active = 0
            self.maximum_active = 0

        def answer(self, question: str, *, k: int) -> RAGResult:
            assert k == FINAL_CONTEXT_K
            with self.state_lock:
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
            sleep(0.03)
            with self.state_lock:
                self.active -= 1
            return make_result(question)

    pipeline = ConcurrentPipeline()
    service = FiscalRAGService(pipeline)
    threads = [
        Thread(target=service.answer, args=(f"question-{index}",))
        for index in range(2)
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert pipeline.maximum_active == 1


def test_service_streams_safe_rag_trace_and_deltas_in_order() -> None:
    document = Document(
        page_content="private chunk content",
        metadata={
            "source": "guide.md",
            "section": "填报",
            "subsection": "操作步骤",
            "_dense_score": 0.75,
            "_rerank_score": 0.95,
        },
    )

    class FakeWorkflow:
        def prepare(self, question: str, *, k: int):
            assert question == "怎样保存？"
            assert k == FINAL_CONTEXT_K
            return PreparedAssistantResponse(
                route="rag",
                rag_request=PreparedRAGRequest(
                    question=question,
                    retrieved_results=[(document, 0.95)],
                    context="private full context",
                    prompt="private prompt",
                    retrieval_query="如何保存？",
                    rewrite_query="如何保存？",
                    query_rewrite_status="accepted",
                ),
            )

        def stream_prepared(self, _prepared):
            yield "点击"
            yield "保存。"

    service = FiscalRAGService(
        object(), assistant_workflow=FakeWorkflow()  # type: ignore[arg-type]
    )

    events = list(service.stream_answer("怎样保存？"))

    assert [event.event for event in events] == [
        "route",
        "trace",
        "delta",
        "delta",
        "done",
    ]
    assert events[1].data["sources"] == [
        {
            "rank": 1,
            "source": "guide.md",
            "section": "填报",
            "subsection": "操作步骤",
            "dense_score": 0.75,
            "rerank_score": 0.95,
        }
    ]
    assert "private chunk content" not in str(events)
    assert "private full context" not in str(events)
    timings = events[-1].data["timings_ms"]
    assert events[-1].data["route"] == "rag"
    assert timings["queue_wait"] >= 0
    assert timings["generation_ttft"] >= 0
    assert timings["generation"] >= timings["generation_ttft"]
    assert timings["server_total"] >= timings["server_ttft"]


def test_service_serializes_complete_assistant_streams() -> None:
    class ConcurrentWorkflow:
        def __init__(self) -> None:
            self.state_lock = Lock()
            self.active = 0
            self.maximum_active = 0

        def prepare(self, _question: str, *, k: int):
            assert k == FINAL_CONTEXT_K
            with self.state_lock:
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
            return PreparedAssistantResponse(route="out_of_scope", fixed_response="边界")

        def stream_prepared(self, _prepared):
            sleep(0.03)
            yield "边界"
            with self.state_lock:
                self.active -= 1

    workflow = ConcurrentWorkflow()
    service = FiscalRAGService(
        object(), assistant_workflow=workflow  # type: ignore[arg-type]
    )
    threads = [
        Thread(target=lambda: list(service.stream_answer("question")))
        for _ in range(2)
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert workflow.maximum_active == 1


def test_closing_assistant_stream_releases_service_lock() -> None:
    class ClosingWorkflow:
        def prepare(self, _question: str, *, k: int):
            assert k == FINAL_CONTEXT_K
            return PreparedAssistantResponse(route="out_of_scope", fixed_response="边界")

        def stream_prepared(self, _prepared):
            yield "first"
            yield "second"

    service = FiscalRAGService(
        object(), assistant_workflow=ClosingWorkflow()  # type: ignore[arg-type]
    )
    stream = service.stream_answer("question")

    assert next(stream).event == "route"
    assert next(stream).event == "delta"
    stream.close()

    assert service._answer_lock.acquire(blocking=False)  # noqa: SLF001
    service._answer_lock.release()  # noqa: SLF001

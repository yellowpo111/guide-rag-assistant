from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
from starlette.requests import Request


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fiscal_rag.api import (  # noqa: E402
    MAX_QUESTION_CHARACTERS,
    _assistant_event_stream,
    create_app,
)
from fiscal_rag.service import (  # noqa: E402
    ServiceAnswer,
    ServiceStreamEvent,
    SourceReference,
)
from fiscal_rag.settings import ServiceSettings  # noqa: E402
from fiscal_rag.usage import (  # noqa: E402
    UsageFeedbackNotAllowed,
    UsageRequestNotFound,
)


class FakeService:
    def __init__(self, error: Exception | None = None) -> None:
        self.questions: list[str] = []
        self.error = error

    def answer(self, question: str) -> ServiceAnswer:
        self.questions.append(question)
        if self.error is not None:
            raise self.error
        return ServiceAnswer(
            answer="点击保存。",
            retrieval_query="如何保存？",
            rewrite_query="如何保存？",
            query_rewrite_status="accepted",
            required_constraints=("建设单位",),
            missing_constraints=(),
            sources=(
                SourceReference(
                    rank=1,
                    source="guide.md",
                    section="填报",
                    subsection="操作步骤",
                    dense_score=0.75,
                    rerank_score=0.95,
                ),
            ),
        )

    def stream_answer(self, question: str):
        self.questions.append(question)
        if self.error is not None:
            raise self.error
        yield ServiceStreamEvent("route", {"route": "chat"})
        yield ServiceStreamEvent("delta", {"text": "你"})
        yield ServiceStreamEvent("delta", {"text": "好"})
        yield ServiceStreamEvent("done", {"route": "chat"})


class FakeUsageRepository:
    def __init__(self, *, create_error: Exception | None = None) -> None:
        self.requests: dict[str, dict[str, object]] = {}
        self.feedback: dict[str, str] = {}
        self.create_error = create_error

    def initialize(self) -> None:
        return None

    def mark_started_interrupted(self) -> int:
        return 0

    def prune_expired(self, _retention_days: int) -> int:
        return 0

    def create_request(self, request_id: str, **fields: object) -> None:
        if self.create_error is not None:
            raise self.create_error
        self.requests[request_id] = {**fields, "execution_status": "started"}

    def finalize_request(self, request_id: str, **fields: object) -> None:
        self.requests[request_id].update(fields)

    def set_feedback(self, request_id: str, rating: str) -> None:
        record = self.requests.get(request_id)
        if record is None:
            raise UsageRequestNotFound(request_id)
        if (
            record.get("endpoint") != "assistant_stream"
            or record.get("execution_status") != "completed"
            or record.get("traffic_kind", "production") != "production"
        ):
            raise UsageFeedbackNotAllowed(request_id)
        self.feedback[request_id] = rating

    def delete_feedback(self, request_id: str) -> bool:
        record = self.requests.get(request_id)
        if record is None:
            raise UsageRequestNotFound(request_id)
        if (
            record.get("endpoint") != "assistant_stream"
            or record.get("execution_status") != "completed"
            or record.get("traffic_kind", "production") != "production"
        ):
            raise UsageFeedbackNotAllowed(request_id)
        return self.feedback.pop(request_id, None) is not None


def make_client(
    service: FakeService,
    usage: FakeUsageRepository | None = None,
    *,
    client_host: str = "127.0.0.1",
) -> TestClient:
    settings = ServiceSettings()
    active_usage = usage or FakeUsageRepository()
    app = create_app(
        settings=settings,
        service_factory=lambda _settings: service,
        usage_factory=lambda _settings: active_usage,
    )
    return TestClient(app, client=(client_host, 50000))


def test_health_checks_are_public_and_do_not_call_pipeline() -> None:
    service = FakeService()
    with make_client(service) as client:
        assert client.get("/health/live").json() == {"status": "ok"}
        assert client.get("/health/ready").json() == {"status": "ready"}

    assert service.questions == []


def test_frontend_and_assets_are_served_without_calling_pipeline() -> None:
    service = FakeService()
    with make_client(service) as client:
        page = client.get("/")
        styles = client.get("/assets/styles.css")
        script = client.get("/assets/app.js")

    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert 'id="ask-form"' in page.text
    assert styles.status_code == 200
    assert ".composer" in styles.text
    assert script.status_code == 200
    assert 'fetch("/v1/assistant/stream"' in script.text
    assert 'Accept: "text/event-stream"' in script.text
    assert "AbortController" in script.text
    assert "response.body.getReader()" in script.text
    assert "/v1/assistant/feedback/" in script.text
    assert "原始记录保留 90 天" in page.text
    assert 'data-rating="positive"' in page.text
    assert 'data-rating="negative"' in page.text
    assert "innerHTML" not in script.text
    assert service.questions == []


def test_ask_does_not_require_authorization_header() -> None:
    service = FakeService()
    with make_client(service) as client:
        response = client.post("/v1/ask", json={"question": "怎样保存？"})

    assert response.status_code == 200
    assert service.questions == ["怎样保存？"]


def test_ask_returns_trace_and_citations_without_private_context() -> None:
    service = FakeService()
    usage = FakeUsageRepository()
    with make_client(service, usage) as client:
        response = client.post(
            "/v1/ask", json={"question": "  怎样保存？  "}
        )

    body = response.json()
    serialized = response.text
    assert response.status_code == 200
    assert service.questions == ["怎样保存？"]
    assert body["request_id"] == response.headers["x-request-id"]
    assert body["answer"] == "点击保存。"
    assert body["retrieval_query"] == "如何保存？"
    assert body["query_rewrite_status"] == "accepted"
    assert body["sources"] == [
        {
            "rank": 1,
            "source": "guide.md",
            "section": "填报",
            "subsection": "操作步骤",
            "dense_score": 0.75,
            "rerank_score": 0.95,
        }
    ]
    assert "page_content" not in serialized
    assert "context" not in serialized
    usage_record = usage.requests[body["request_id"]]
    assert usage_record["endpoint"] == "ask"
    assert usage_record["route"] == "rag"
    assert usage_record["execution_status"] == "completed"
    assert usage_record["answer"] == "点击保存。"
    assert usage_record["trace"]["sources"][0]["source"] == "guide.md"


def test_ask_rejects_empty_or_oversized_questions() -> None:
    with make_client(FakeService()) as client:
        empty = client.post(
            "/v1/ask", json={"question": "   "}
        )
        oversized = client.post(
            "/v1/ask",
            json={"question": "x" * (MAX_QUESTION_CHARACTERS + 1)},
        )

    assert empty.status_code == 422
    assert oversized.status_code == 422
    assert empty.json()["error"]["code"] == "validation_error"


def test_model_failure_is_sanitized_and_carries_request_id(caplog) -> None:
    private_error = "provider failed with private upstream response"
    with make_client(FakeService(RuntimeError(private_error))) as client:
        response = client.post(
            "/v1/ask", json={"question": "怎样保存？"}
        )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "model_request_failed"
    assert response.json()["error"]["request_id"] == response.headers["x-request-id"]
    assert private_error not in response.text
    assert private_error not in caplog.text


def test_openapi_does_not_declare_authentication_for_ask() -> None:
    with make_client(FakeService()) as client:
        schema = client.get("/openapi.json").json()

    assert "security" not in schema["paths"]["/v1/ask"]["post"]
    assert "securitySchemes" not in schema.get("components", {})


def test_assistant_stream_returns_ordered_utf8_sse_events() -> None:
    service = FakeService()
    usage = FakeUsageRepository()
    with make_client(service, usage) as client:
        response = client.post(
            "/v1/assistant/stream", json={"question": "你好"}
        )

    request_id = response.headers["x-request-id"]
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert service.questions == ["你好"]
    assert response.text.index("event: start") < response.text.index("event: route")
    assert response.text.index("event: route") < response.text.index("event: delta")
    assert response.text.index("event: delta") < response.text.index("event: done")
    assert f'"request_id":"{request_id}"' in response.text
    assert 'data: {"text":"你"}' in response.text
    assert usage.requests[request_id]["execution_status"] == "completed"
    assert usage.requests[request_id]["route"] == "chat"
    assert usage.requests[request_id]["answer"] == "你好"


def test_assistant_stream_records_internal_eval_traffic_kind() -> None:
    usage = FakeUsageRepository()
    with make_client(FakeService(), usage) as client:
        eval_response = client.post(
            "/v1/assistant/stream",
            json={"question": "你好"},
            headers={"X-Fiscal-RAG-Traffic-Kind": "assistant_eval"},
        )
        unknown_response = client.post(
            "/v1/assistant/stream",
            json={"question": "你好"},
            headers={"X-Fiscal-RAG-Traffic-Kind": "unsupported"},
        )
        eval_feedback = client.put(
            f"/v1/assistant/feedback/{eval_response.headers['x-request-id']}",
            json={"rating": "positive"},
        )
    remote_usage = FakeUsageRepository()
    with make_client(
        FakeService(), remote_usage, client_host="192.0.2.10"
    ) as remote_client:
        untrusted_response = remote_client.post(
            "/v1/assistant/stream",
            json={"question": "你好"},
            headers={"X-Fiscal-RAG-Traffic-Kind": "assistant_eval"},
        )

    assert usage.requests[eval_response.headers["x-request-id"]]["traffic_kind"] == (
        "assistant_eval"
    )
    assert usage.requests[unknown_response.headers["x-request-id"]]["traffic_kind"] == (
        "production"
    )
    assert remote_usage.requests[
        untrusted_response.headers["x-request-id"]
    ]["traffic_kind"] == "production"
    assert eval_feedback.status_code == 409


def test_assistant_stream_records_aborted_when_iterator_is_closed() -> None:
    usage = FakeUsageRepository()
    request_id = "aborted-request"
    usage.create_request(
        request_id,
        endpoint="assistant_stream",
        question="你好",
        service_version="1.2.0",
        profile_id="profile",
        traffic_kind="production",
    )
    app = SimpleNamespace(
        state=SimpleNamespace(rag_service=FakeService(), usage_repository=usage)
    )
    request = Request({"type": "http", "app": app})
    request.state.request_id = request_id
    stream = _assistant_event_stream(
        request,
        "你好",
        request_id=request_id,
        usage_created=True,
    )

    assert "event: start" in next(stream)
    assert "event: route" in next(stream)
    stream.close()

    record = usage.requests[request_id]
    assert record["execution_status"] == "aborted"
    assert record["route"] == "chat"
    assert record["error_code"] == "client_disconnected"


def test_assistant_stream_sanitizes_failure_after_stream_start(caplog) -> None:
    private_error = "provider failed with private upstream response"
    usage = FakeUsageRepository()
    with make_client(FakeService(RuntimeError(private_error)), usage) as client:
        response = client.post(
            "/v1/assistant/stream", json={"question": "你好"}
        )

    assert response.status_code == 200
    assert "event: start" in response.text
    assert "event: error" in response.text
    assert "model_request_failed" in response.text
    assert private_error not in response.text
    assert private_error not in caplog.text
    record = usage.requests[response.headers["x-request-id"]]
    assert record["execution_status"] == "failed"
    assert record["error_type"] == "RuntimeError"
    assert record["answer"] == ""


def test_assistant_stream_validates_request_before_opening_stream() -> None:
    with make_client(FakeService()) as client:
        response = client.post(
            "/v1/assistant/stream", json={"question": "   "}
        )

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["code"] == "validation_error"


def test_feedback_can_be_changed_and_revoked_for_completed_assistant_request() -> None:
    usage = FakeUsageRepository()
    with make_client(FakeService(), usage) as client:
        assistant = client.post("/v1/assistant/stream", json={"question": "你好"})
        request_id = assistant.headers["x-request-id"]

        positive = client.put(
            f"/v1/assistant/feedback/{request_id}", json={"rating": "positive"}
        )
        negative = client.put(
            f"/v1/assistant/feedback/{request_id}", json={"rating": "negative"}
        )
        removed = client.delete(f"/v1/assistant/feedback/{request_id}")

    assert positive.json() == {"request_id": request_id, "rating": "positive"}
    assert negative.status_code == 200
    assert removed.status_code == 204
    assert request_id not in usage.feedback


def test_feedback_rejects_unknown_incomplete_or_invalid_requests() -> None:
    usage = FakeUsageRepository()
    usage.requests["started"] = {
        "endpoint": "assistant_stream",
        "execution_status": "started",
    }
    usage.requests["ask"] = {
        "endpoint": "ask",
        "execution_status": "completed",
    }
    with make_client(FakeService(), usage) as client:
        unknown = client.put(
            "/v1/assistant/feedback/unknown", json={"rating": "negative"}
        )
        incomplete = client.put(
            "/v1/assistant/feedback/started", json={"rating": "negative"}
        )
        invalid = client.put(
            "/v1/assistant/feedback/started", json={"rating": "neutral"}
        )
        incomplete_delete = client.delete("/v1/assistant/feedback/started")
        ask_delete = client.delete("/v1/assistant/feedback/ask")

    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "request_not_found"
    assert incomplete.status_code == 409
    assert invalid.status_code == 422
    assert incomplete_delete.status_code == 409
    assert ask_delete.status_code == 409


def test_usage_create_failure_does_not_break_assistant_answer(caplog) -> None:
    usage = FakeUsageRepository(create_error=RuntimeError("private database detail"))

    with make_client(FakeService(), usage) as client:
        response = client.post("/v1/assistant/stream", json={"question": "你好"})

    assert response.status_code == 200
    assert "event: done" in response.text
    assert "private database detail" not in caplog.text
    assert "usage_create_failed" in caplog.text


def test_slow_usage_create_does_not_block_health_requests() -> None:
    class SlowUsageRepository(FakeUsageRepository):
        def __init__(self) -> None:
            super().__init__()
            self.create_started = threading.Event()
            self.create_release = threading.Event()

        def create_request(self, request_id: str, **fields: object) -> None:
            self.create_started.set()
            if not self.create_release.wait(timeout=2):
                raise TimeoutError("test did not release usage create")
            super().create_request(request_id, **fields)

    usage = SlowUsageRepository()
    response_holder = []
    with make_client(FakeService(), usage) as client:
        request_thread = threading.Thread(
            target=lambda: response_holder.append(
                client.post("/v1/assistant/stream", json={"question": "你好"})
            )
        )
        request_thread.start()
        assert usage.create_started.wait(timeout=1)
        started = time.perf_counter()
        health = client.get("/health/ready")
        health_elapsed = time.perf_counter() - started
        usage.create_release.set()
        request_thread.join(timeout=2)

    assert health.status_code == 200
    assert health_elapsed < 0.5
    assert not request_thread.is_alive()
    assert response_holder[0].status_code == 200

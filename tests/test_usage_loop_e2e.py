import json
import socket
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.request import Request, urlopen

import uvicorn


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fiscal_rag.api import create_app  # noqa: E402
from fiscal_rag.assistant_evaluation import (  # noqa: E402
    load_assistant_eval_cases,
    write_jsonl_exclusive,
)
from fiscal_rag.service import ServiceStreamEvent  # noqa: E402
from fiscal_rag.settings import ServiceSettings  # noqa: E402
from fiscal_rag.sse_client import post_assistant_stream  # noqa: E402
from fiscal_rag.usage import SQLiteUsageRepository  # noqa: E402
from fiscal_rag.usage_analysis import (  # noqa: E402
    eval_candidate_records,
    load_usage_reviews,
    render_usage_summary_markdown,
    summarize_usage_records,
    usage_review_template_records,
)


class ClosedLoopService:
    def stream_answer(self, _question: str):
        yield ServiceStreamEvent("route", {"route": "chat"})
        yield ServiceStreamEvent("delta", {"text": "Hello."})
        yield ServiceStreamEvent(
            "done",
            {"route": "chat", "timings_ms": {"server_total": 10.0}},
        )


@contextmanager
def running_server(app):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="critical")
    )
    thread = threading.Thread(
        target=server.run,
        kwargs={"sockets": [listener]},
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()
        raise RuntimeError("Test Uvicorn server did not start.")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()
        if thread.is_alive():
            raise RuntimeError("Test Uvicorn server did not stop.")


def test_real_usage_failure_can_be_promoted_and_run_as_assistant_eval(
    tmp_path: Path,
) -> None:
    repository = SQLiteUsageRepository(tmp_path / "usage.sqlite3")
    app = create_app(
        settings=ServiceSettings(usage_db_path=repository.database_path),
        service_factory=lambda _settings: ClosedLoopService(),
        usage_factory=lambda _settings: repository,
    )

    with running_server(app) as base_url:
        production = post_assistant_stream(base_url, "hello")
        assert production.status == "completed"
        assert production.route == "chat"
        feedback_request = Request(
            f"{base_url}/v1/assistant/feedback/{production.request_id}",
            data=json.dumps({"rating": "negative"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="PUT",
        )
        with urlopen(feedback_request, timeout=5) as response:  # noqa: S310
            assert response.status == 200

        before_review = summarize_usage_records(repository.fetch_usage_records())
        assert before_review["review_funnel"]["actionable_requests"] == 1
        assert before_review["feedback"]["negative"] == 1
        assert "hello" not in render_usage_summary_markdown(before_review)

        template = usage_review_template_records(
            repository.fetch_usage_records()
        )[0]
        template.update(
            {
                "review_status": "user_confirmed",
                "failure_type": "generation_correctness",
                "severity": "major",
                "expected_route": "chat",
                "answerability": None,
                "reason": "The response was not useful.",
                "eval_candidate": True,
                "candidate_case_id": "usage-chat-001",
                "candidate_category": "chat",
            }
        )
        review_file = tmp_path / "confirmed_review.jsonl"
        write_jsonl_exclusive(review_file, [template])
        repository.save_reviews(load_usage_reviews(review_file))

        after_review = summarize_usage_records(repository.fetch_usage_records())
        assert after_review["review_funnel"]["actionable_requests"] == 0
        assert after_review["review_funnel"]["user_confirmed"] == 1
        assert after_review["review_funnel"]["eval_candidates"] == 1

        candidate_file = tmp_path / "assistant_eval_usage_v2.jsonl"
        write_jsonl_exclusive(
            candidate_file,
            eval_candidate_records(repository.fetch_usage_records()),
        )
        candidate = load_assistant_eval_cases(candidate_file)[0]
        evaluated = post_assistant_stream(
            base_url,
            candidate.question,
            traffic_kind="assistant_eval",
        )

    assert candidate.case_id == "usage-chat-001"
    assert evaluated.status == "completed"
    assert evaluated.route == candidate.expected_route == "chat"
    traffic = [
        record["traffic_kind"] for record in repository.fetch_usage_records()
    ]
    assert traffic == ["production", "assistant_eval"]

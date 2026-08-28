from io import BytesIO
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fiscal_rag.sse_client import (  # noqa: E402, PLC2701
    _consume_sse_lines,
    post_assistant_stream,
)


def test_sse_client_uses_first_delta_for_ttft_and_reads_done_timings() -> None:
    body = BytesIO(
        (
            'event: start\ndata: {"request_id":"r1"}\n\n'
            'event: route\ndata: {"route":"chat"}\n\n'
            'event: delta\ndata: {"text":"你"}\n\n'
            'event: delta\ndata: {"text":"好"}\n\n'
            'event: done\ndata: {"route":"chat","timings_ms":{"server_total":12.5}}\n\n'
        ).encode("utf-8")
    )

    result = _consume_sse_lines(body, started=0.0)

    assert result["status"] == "completed"
    assert result["route"] == "chat"
    assert result["answer"] == "你好"
    assert result["timings_ms"] == {"server_total": 12.5}
    assert result["events"] == ("start", "route", "delta", "delta", "done")


def test_sse_client_rejects_unknown_traffic_kind_before_network_access() -> None:
    with pytest.raises(ValueError, match="traffic_kind"):
        post_assistant_stream(
            "http://127.0.0.1:1",
            "question",
            traffic_kind="production",
        )


def test_sse_client_requires_localhost_for_evaluation_traffic() -> None:
    with pytest.raises(ValueError, match="localhost"):
        post_assistant_stream(
            "http://192.0.2.10:8000",
            "question",
            traffic_kind="assistant_eval",
        )

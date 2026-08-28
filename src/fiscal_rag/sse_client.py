"""Small blocking client for the project's POST-based SSE endpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from ipaddress import ip_address
from time import perf_counter
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class AssistantSseResult:
    """Observable client-side result of one assistant stream."""

    status: str
    request_id: str | None
    route: str | None
    answer: str
    trace: dict[str, object] | None
    timings_ms: dict[str, float]
    client_ttft_ms: float | None
    client_total_ms: float
    events: tuple[str, ...]
    error_code: str | None = None
    error_message: str | None = None


def post_assistant_stream(
    base_url: str,
    question: str,
    *,
    timeout_seconds: float = 120.0,
    traffic_kind: str | None = None,
) -> AssistantSseResult:
    """POST one question and consume the SSE response without retrying."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    endpoint = base_url.rstrip("/") + "/v1/assistant/stream"
    headers = {
        "Accept": "text/event-stream",
        "Content-Type": "application/json; charset=utf-8",
    }
    if traffic_kind is not None:
        if traffic_kind not in {"assistant_eval", "performance_eval"}:
            raise ValueError("unsupported traffic_kind")
        if not _is_loopback_url(endpoint):
            raise ValueError("evaluation traffic must target localhost")
        headers["X-Fiscal-RAG-Traffic-Kind"] = traffic_kind
    request = Request(
        endpoint,
        data=json.dumps({"question": question}, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = perf_counter()
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            request_id = response.headers.get("X-Request-ID")
            parsed = _consume_sse_lines(response, started=started)
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        return _transport_failure(
            started,
            request_id=error.headers.get("X-Request-ID"),
            code=f"http_{error.code}",
            message=_safe_error_message(body),
        )
    except (URLError, TimeoutError) as error:
        return _transport_failure(
            started,
            request_id=None,
            code="transport_error",
            message=type(error).__name__,
        )
    return AssistantSseResult(request_id=request_id, **parsed)


def _consume_sse_lines(response: Any, *, started: float) -> dict[str, object]:
    event_name = "message"
    data_lines: list[str] = []
    events: list[str] = []
    route: str | None = None
    trace: dict[str, object] | None = None
    answer_parts: list[str] = []
    timings_ms: dict[str, float] = {}
    client_ttft_ms: float | None = None
    status = "incomplete"
    error_code: str | None = None
    error_message: str | None = None

    def dispatch() -> None:
        nonlocal route, trace, timings_ms, client_ttft_ms
        nonlocal status, error_code, error_message
        if not data_lines:
            return
        payload = json.loads("\n".join(data_lines))
        if not isinstance(payload, dict):
            raise ValueError("SSE data must be a JSON object.")
        events.append(event_name)
        if event_name == "route":
            value = payload.get("route")
            route = value if isinstance(value, str) else None
        elif event_name == "trace":
            trace = payload
        elif event_name == "delta":
            text = payload.get("text")
            if isinstance(text, str) and text:
                if client_ttft_ms is None:
                    client_ttft_ms = (perf_counter() - started) * 1000
                answer_parts.append(text)
        elif event_name == "done":
            status = "completed"
            raw_timings = payload.get("timings_ms")
            if isinstance(raw_timings, dict):
                timings_ms = {
                    str(name): float(value)
                    for name, value in raw_timings.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
        elif event_name == "error":
            status = "failed"
            code = payload.get("code")
            message = payload.get("message")
            error_code = code if isinstance(code, str) else "stream_error"
            error_message = message if isinstance(message, str) else "Stream failed."

    for raw_line in response:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not line:
            dispatch()
            event_name = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    dispatch()

    return {
        "status": status,
        "route": route,
        "answer": "".join(answer_parts),
        "trace": trace,
        "timings_ms": timings_ms,
        "client_ttft_ms": (
            None if client_ttft_ms is None else round(client_ttft_ms, 3)
        ),
        "client_total_ms": round((perf_counter() - started) * 1000, 3),
        "events": tuple(events),
        "error_code": error_code,
        "error_message": error_message,
    }


def _transport_failure(
    started: float,
    *,
    request_id: str | None,
    code: str,
    message: str,
) -> AssistantSseResult:
    return AssistantSseResult(
        status="failed",
        request_id=request_id,
        route=None,
        answer="",
        trace=None,
        timings_ms={},
        client_ttft_ms=None,
        client_total_ms=round((perf_counter() - started) * 1000, 3),
        events=(),
        error_code=code,
        error_message=message,
    )


def _safe_error_message(body: str) -> str:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return "HTTP request failed."
    if not isinstance(payload, dict):
        return "HTTP request failed."
    error = payload.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    return "HTTP request failed."


def _is_loopback_url(value: str) -> bool:
    hostname = urlsplit(value).hostname
    if hostname is None:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ip_address(hostname).is_loopback
    except ValueError:
        return False

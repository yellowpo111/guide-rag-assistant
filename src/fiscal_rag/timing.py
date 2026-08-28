"""Request-scoped stage timing without changing retrieval interfaces."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from time import perf_counter


_ACTIVE_RECORDER: ContextVar[RequestTimingRecorder | None] = ContextVar(
    "fiscal_rag_active_timing_recorder",
    default=None,
)


@dataclass
class RequestTimingRecorder:
    """Accumulate elapsed milliseconds for one request."""

    _durations_ms: dict[str, float] = field(default_factory=dict)

    def add(self, stage: str, duration_ms: float) -> None:
        if duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")
        self._durations_ms[stage] = self._durations_ms.get(stage, 0.0) + duration_ms

    def set(self, stage: str, duration_ms: float) -> None:
        if duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")
        self._durations_ms[stage] = duration_ms

    def snapshot(self) -> Mapping[str, float]:
        return {
            stage: round(duration_ms, 3)
            for stage, duration_ms in sorted(self._durations_ms.items())
        }


@contextmanager
def use_timing_recorder(recorder: RequestTimingRecorder) -> Iterator[None]:
    """Make a recorder visible to nested synchronous pipeline components."""
    token: Token[RequestTimingRecorder | None] = _ACTIVE_RECORDER.set(recorder)
    try:
        yield
    finally:
        _ACTIVE_RECORDER.reset(token)


@contextmanager
def measure_stage(stage: str) -> Iterator[None]:
    """Measure a stage only when the current request enabled timing."""
    recorder = _ACTIVE_RECORDER.get()
    if recorder is None:
        yield
        return

    started = perf_counter()
    try:
        yield
    finally:
        recorder.add(stage, (perf_counter() - started) * 1000)

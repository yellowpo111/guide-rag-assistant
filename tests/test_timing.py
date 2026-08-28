from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fiscal_rag.timing import (  # noqa: E402
    RequestTimingRecorder,
    measure_stage,
    use_timing_recorder,
)


def test_timing_recorder_accumulates_and_rounds_stages() -> None:
    recorder = RequestTimingRecorder()
    recorder.add("rewrite", 1.2344)
    recorder.add("rewrite", 2.0004)
    recorder.set("queue_wait", 0.4446)

    assert recorder.snapshot() == {"queue_wait": 0.445, "rewrite": 3.235}


def test_measure_stage_is_request_scoped(monkeypatch) -> None:
    timestamps = iter([10.0, 10.025])
    monkeypatch.setattr("fiscal_rag.timing.perf_counter", lambda: next(timestamps))
    recorder = RequestTimingRecorder()

    with use_timing_recorder(recorder):
        with measure_stage("router"):
            pass

    assert recorder.snapshot() == {"router": 25.0}


def test_measure_stage_without_recorder_is_a_noop() -> None:
    with measure_stage("unused"):
        pass

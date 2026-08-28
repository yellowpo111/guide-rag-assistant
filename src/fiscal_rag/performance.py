"""Aggregation helpers for the deployed assistant performance baseline."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from math import ceil, floor
from typing import Any


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sample."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = floor(position)
    upper = ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def metric_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": round(min(values), 3),
        "p50": round(percentile(values, 0.50), 3),
        "p95": round(percentile(values, 0.95), 3),
        "max": round(max(values), 3),
    }


def summarize_performance_records(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Summarize measured sequential runs and paired queue-wait checks."""
    measured = [record for record in records if record.get("run_kind") == "measured"]
    concurrency = [
        record for record in records if record.get("run_kind") == "concurrency"
    ]
    by_route: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in measured:
        route = record.get("actual_route")
        by_route[str(route) if isinstance(route, str) else "unknown"].append(record)
    first_requests = _first_sequential_request_per_case(records)

    return {
        "schema_version": "performance-eval-summary-v1",
        "measured_requests": len(measured),
        "completion_rate": _completion_rate(measured),
        "first_request_success_rate": _completion_rate(first_requests),
        "routes": {
            route: _summarize_group(group) for route, group in sorted(by_route.items())
        },
        "overall": _summarize_group(measured),
        "concurrency_sanity": {
            "requests": len(concurrency),
            "completion_rate": _completion_rate(concurrency),
            "queue_wait_ms": metric_summary(
                _timing_values(concurrency, "queue_wait")
            ),
        },
    }


def _first_sequential_request_per_case(
    records: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    first_requests: list[Mapping[str, object]] = []
    seen: set[str] = set()
    for record in records:
        if record.get("run_kind") not in {"warmup", "measured"}:
            continue
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or case_id in seen:
            continue
        seen.add(case_id)
        first_requests.append(record)
    return first_requests


def _summarize_group(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    stages = sorted(
        {
            str(stage)
            for record in records
            for stage in _timings(record)
        }
    )
    return {
        "requests": len(records),
        "completion_rate": _completion_rate(records),
        "client_ttft_ms": metric_summary(_numeric_values(records, "client_ttft_ms")),
        "client_total_ms": metric_summary(_numeric_values(records, "client_total_ms")),
        "server_stages_ms": {
            stage: metric_summary(_timing_values(records, stage)) for stage in stages
        },
    }


def _completion_rate(records: Sequence[Mapping[str, object]]) -> float:
    if not records:
        return 0.0
    return sum(record.get("status") == "completed" for record in records) / len(records)


def _numeric_values(
    records: Sequence[Mapping[str, object]], field: str
) -> list[float]:
    values = []
    for record in records:
        value = record.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return values


def _timings(record: Mapping[str, object]) -> Mapping[str, Any]:
    timings = record.get("timings_ms")
    return timings if isinstance(timings, Mapping) else {}


def _timing_values(
    records: Sequence[Mapping[str, object]], stage: str
) -> list[float]:
    values = []
    for record in records:
        value = _timings(record).get(stage)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return values

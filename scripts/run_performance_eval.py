"""Measure the deployed assistant without storing questions or answers."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fiscal_rag.assistant_evaluation import write_jsonl_exclusive  # noqa: E402
from fiscal_rag.local_service import local_service_if_requested  # noqa: E402
from fiscal_rag.performance import summarize_performance_records  # noqa: E402
from fiscal_rag.sse_client import AssistantSseResult, post_assistant_stream  # noqa: E402
from fiscal_rag.version import __version__  # noqa: E402


DEFAULT_CASES_FILE = PROJECT_ROOT / "data_private" / "evals" / "performance_eval_v1.jsonl"
DEFAULT_RESULTS_DIRECTORY = PROJECT_ROOT / "data_private" / "evals" / "results"
RELEASE_VERSION = f"v{__version__}"
PROFILE_ID = "rewrite-guard-dense20-rerank5"


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the deployed Assistant performance baseline.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases-file", type=Path, default=DEFAULT_CASES_FILE)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--summary-file", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--environment-label", default="internal-deployment")
    parser.add_argument("--warmups", type=non_negative_integer, default=1)
    parser.add_argument("--repetitions", type=positive_integer, default=3)
    parser.add_argument("--concurrency-pairs", type=non_negative_integer, default=3)
    parser.add_argument("--timeout-seconds", type=positive_float, default=120.0)
    parser.add_argument(
        "--start-local-service",
        action="store_true",
        help="Start a temporary localhost Uvicorn service for this run.",
    )
    return parser.parse_args(argv)


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def non_negative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def load_performance_cases(path: str | Path) -> list[dict[str, str]]:
    records = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict) or record.get("schema_version") != "performance-eval-v1":
            raise ValueError(f"Performance case line {line_number} has an invalid schema.")
        values = {}
        for field in ("case_id", "question", "expected_route"):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Performance case line {line_number} requires {field}.")
            values[field] = value
        if values["case_id"] in seen:
            raise ValueError(f"Performance cases repeat case_id: {values['case_id']!r}")
        seen.add(values["case_id"])
        records.append(values)
    route_counts = {
        route: sum(case["expected_route"] == route for case in records)
        for route in ("rag", "chat", "out_of_scope")
    }
    if route_counts != {"rag": 8, "chat": 4, "out_of_scope": 4}:
        raise ValueError(f"Performance Eval requires route counts 8/4/4, got {route_counts}.")
    return records


def performance_record(
    case: dict[str, str],
    result: AssistantSseResult,
    *,
    run_id: str,
    run_kind: str,
    iteration: int,
    environment_label: str,
    pair_id: int | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "performance-eval-result-v1",
        "release_version": RELEASE_VERSION,
        "profile_id": PROFILE_ID,
        "run_id": run_id,
        "run_kind": run_kind,
        "iteration": iteration,
        "pair_id": pair_id,
        "environment": {
            "label": environment_label,
            "python": platform.python_version(),
            "os": platform.system(),
            "os_release": platform.release(),
            "machine": platform.machine(),
        },
        "case_id": case["case_id"],
        "expected_route": case["expected_route"],
        "actual_route": result.route,
        "status": result.status,
        "request_id": result.request_id,
        "answer_characters": len(result.answer),
        "client_ttft_ms": result.client_ttft_ms,
        "client_total_ms": result.client_total_ms,
        "timings_ms": result.timings_ms,
        "error_code": result.error_code,
    }


def main() -> None:
    args = parse_arguments()
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_file = args.output_file or (
        DEFAULT_RESULTS_DIRECTORY / f"performance_eval_v1_{run_id}.jsonl"
    )
    summary_file = args.summary_file or output_file.with_suffix(".summary.json")
    cases = load_performance_cases(args.cases_file)
    records: list[dict[str, object]] = []

    with local_service_if_requested(args.start_local_service, args.base_url):
        for case in cases:
            for iteration in range(1, args.warmups + 1):
                result = post_assistant_stream(
                    args.base_url,
                    case["question"],
                    timeout_seconds=args.timeout_seconds,
                    traffic_kind="performance_eval",
                )
                records.append(
                    performance_record(
                        case,
                        result,
                        run_id=run_id,
                        run_kind="warmup",
                        iteration=iteration,
                        environment_label=args.environment_label,
                    )
                )
            for iteration in range(1, args.repetitions + 1):
                print(f"Measured {case['case_id']} repetition {iteration}/{args.repetitions}")
                result = post_assistant_stream(
                    args.base_url,
                    case["question"],
                    timeout_seconds=args.timeout_seconds,
                    traffic_kind="performance_eval",
                )
                records.append(
                    performance_record(
                        case,
                        result,
                        run_id=run_id,
                        run_kind="measured",
                        iteration=iteration,
                        environment_label=args.environment_label,
                    )
                )

        rag_cases = [case for case in cases if case["expected_route"] == "rag"]
        for pair_id in range(1, args.concurrency_pairs + 1):
            pair_cases = rag_cases[(pair_id - 1) * 2 : (pair_id - 1) * 2 + 2]
            if len(pair_cases) < 2:
                pair_cases = rag_cases[:2]
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(
                        post_assistant_stream,
                        args.base_url,
                        case["question"],
                        timeout_seconds=args.timeout_seconds,
                        traffic_kind="performance_eval",
                    )
                    for case in pair_cases
                ]
                for case, future in zip(pair_cases, futures):
                    records.append(
                        performance_record(
                            case,
                            future.result(),
                            run_id=run_id,
                            run_kind="concurrency",
                            iteration=1,
                            pair_id=pair_id,
                            environment_label=args.environment_label,
                        )
                    )

    write_jsonl_exclusive(output_file, records)
    summary = summarize_performance_records(records)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with summary_file.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Raw Results: {output_file}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()

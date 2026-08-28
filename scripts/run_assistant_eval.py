"""Run the private Assistant Eval through the deployed POST SSE endpoint."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fiscal_rag.assistant_evaluation import (  # noqa: E402
    load_assistant_eval_cases,
    result_record,
    summarize_assistant_results,
    write_jsonl_exclusive,
)
from fiscal_rag.local_service import (  # noqa: E402
    local_service_address,
    local_service_if_requested,
)
from fiscal_rag.sse_client import post_assistant_stream  # noqa: E402


DEFAULT_CASES_FILE = PROJECT_ROOT / "data_private" / "evals" / "assistant_eval_v1.jsonl"
DEFAULT_RESULTS_DIRECTORY = PROJECT_ROOT / "data_private" / "evals" / "results"


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end Assistant Eval over HTTP SSE.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases-file", type=Path, default=DEFAULT_CASES_FILE)
    parser.add_argument("--details-file", type=Path, default=None)
    parser.add_argument("--summary-file", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        default=None,
        help="Run only this case ID; may be repeated for a retry subset.",
    )
    parser.add_argument("--attempt", type=positive_integer, default=1)
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


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main() -> None:
    args = parse_arguments()
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    details_file = args.details_file or (
        DEFAULT_RESULTS_DIRECTORY / f"assistant_eval_v1_{run_id}_attempt_{args.attempt}.jsonl"
    )
    summary_file = args.summary_file or details_file.with_suffix(".summary.json")
    with local_service_if_requested(args.start_local_service, args.base_url):
        cases = select_cases(
            load_assistant_eval_cases(args.cases_file), args.case_ids
        )
        records = []
        for index, case in enumerate(cases, start=1):
            print(f"Running {index}/{len(cases)}: {case.case_id}")
            stream_result = post_assistant_stream(
                args.base_url,
                case.question,
                timeout_seconds=args.timeout_seconds,
                traffic_kind="assistant_eval",
            )
            records.append(
                result_record(
                    case,
                    attempt=args.attempt,
                    run_id=run_id,
                    result=stream_result,
                )
            )

    write_jsonl_exclusive(details_file, records)
    summary = summarize_assistant_results(cases, records)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with summary_file.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Details File: {details_file}")
    print(f"Summary File: {summary_file}")


def select_cases(cases, requested_case_ids: list[str] | None):
    if not requested_case_ids:
        return cases
    if len(requested_case_ids) != len(set(requested_case_ids)):
        raise ValueError("--case-id values must not repeat.")
    requested = set(requested_case_ids)
    available = {case.case_id for case in cases}
    unknown = sorted(requested - available)
    if unknown:
        raise ValueError("Unknown Assistant Eval case IDs: " + ", ".join(unknown))
    return [case for case in cases if case.case_id in requested]


if __name__ == "__main__":
    main()

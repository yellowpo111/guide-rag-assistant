"""Export non-overwriting private review templates from usage SQLite."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fiscal_rag.assistant_evaluation import write_jsonl_exclusive  # noqa: E402
from fiscal_rag.settings import (  # noqa: E402
    DEFAULT_USAGE_DATABASE,
    DEFAULT_USAGE_RETENTION_DAYS,
)
from fiscal_rag.usage import SQLiteUsageRepository  # noqa: E402
from fiscal_rag.usage_analysis import (  # noqa: E402
    select_usage_review_templates,
    validate_usage_window,
)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export private usage review templates.")
    parser.add_argument("--database", type=Path, default=DEFAULT_USAGE_DATABASE)
    parser.add_argument("--started-from", default=None, help="Inclusive UTC ISO timestamp.")
    parser.add_argument("--started-to", default=None, help="Exclusive UTC ISO timestamp.")
    parser.add_argument(
        "--request-id",
        action="append",
        dest="request_ids",
        default=None,
        help="Export this actionable request ID; repeatable.",
    )
    parser.add_argument("--slow-ms", type=positive_float, default=6000.0)
    parser.add_argument(
        "--retention-days", type=positive_integer, default=DEFAULT_USAGE_RETENTION_DAYS
    )
    parser.add_argument("--output-file", type=Path, default=None)
    return parser.parse_args(argv)


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main() -> None:
    args = parse_arguments()
    started_from, started_to = validate_usage_window(
        args.started_from,
        args.started_to,
    )
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_file = args.output_file or (
        PROJECT_ROOT / "data_private" / "usage" / "reviews" / f"usage_review_{run_id}.jsonl"
    )
    repository = SQLiteUsageRepository(args.database)
    repository.initialize()
    records = select_usage_review_templates(
        repository.fetch_usage_records(
            started_from=started_from,
            started_to=started_to,
        ),
        request_ids=args.request_ids or (),
        slow_ms=args.slow_ms,
        retention_days=args.retention_days,
    )
    write_jsonl_exclusive(output_file, records)
    print(f"Review candidates: {len(records)}")
    print(f"Private output: {output_file}")


if __name__ == "__main__":
    main()

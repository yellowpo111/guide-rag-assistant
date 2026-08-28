"""Create a non-overwriting private usage summary from SQLite."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fiscal_rag.settings import (  # noqa: E402
    DEFAULT_USAGE_DATABASE,
    DEFAULT_USAGE_RETENTION_DAYS,
)
from fiscal_rag.usage import SQLiteUsageRepository  # noqa: E402
from fiscal_rag.usage_analysis import (  # noqa: E402
    render_usage_summary_markdown,
    summarize_usage_records,
    validate_usage_window,
)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize private assistant usage.")
    parser.add_argument("--database", type=Path, default=DEFAULT_USAGE_DATABASE)
    parser.add_argument("--started-from", default=None, help="Inclusive UTC ISO timestamp.")
    parser.add_argument("--started-to", default=None, help="Exclusive UTC ISO timestamp.")
    parser.add_argument("--slow-ms", type=positive_float, default=6000.0)
    parser.add_argument("--top-n", type=positive_integer, default=10)
    parser.add_argument("--queue-limit", type=positive_integer, default=20)
    parser.add_argument("--include-raw-questions", action="store_true")
    parser.add_argument(
        "--retention-days", type=positive_integer, default=DEFAULT_USAGE_RETENTION_DAYS
    )
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--markdown-file", type=Path, default=None)
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


def write_report_pair(
    json_path: Path,
    markdown_path: Path,
    summary: dict[str, object],
) -> None:
    """Exclusively create both report formats or leave neither new file behind."""
    if json_path.resolve() == markdown_path.resolve():
        raise ValueError("JSON and Markdown report paths must be different")
    if json_path.exists() or markdown_path.exists():
        existing = json_path if json_path.exists() else markdown_path
        raise FileExistsError(existing)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        with json_path.open("x", encoding="utf-8", newline="\n") as output:
            created.append(json_path)
            json.dump(summary, output, ensure_ascii=False, indent=2)
            output.write("\n")
        with markdown_path.open("x", encoding="utf-8", newline="\n") as output:
            created.append(markdown_path)
            output.write(render_usage_summary_markdown(summary))
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def main() -> None:
    args = parse_arguments()
    started_from, started_to = validate_usage_window(
        args.started_from,
        args.started_to,
    )
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_file = args.output_file or (
        PROJECT_ROOT / "data_private" / "usage" / "reports" / f"usage_summary_{run_id}.json"
    )
    markdown_file = args.markdown_file or output_file.with_suffix(".md")
    repository = SQLiteUsageRepository(args.database)
    repository.initialize()
    records = repository.fetch_usage_records(
        started_from=started_from,
        started_to=started_to,
    )
    summary = summarize_usage_records(
        records,
        top_n=args.top_n,
        retention_days=args.retention_days,
        slow_ms=args.slow_ms,
        queue_limit=args.queue_limit,
        include_raw_questions=args.include_raw_questions,
        started_from=started_from,
        started_to=started_to,
    )
    write_report_pair(output_file, markdown_file, summary)
    overview = summary["overview"]
    review_funnel = summary["review_funnel"]
    print(f"Production requests summarized: {overview['total_requests']}")
    print(f"Actionable requests: {review_funnel['actionable_requests']}")
    print(f"Private JSON: {output_file}")
    print(f"Private Markdown: {markdown_file}")


if __name__ == "__main__":
    main()

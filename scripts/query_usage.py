"""Run one local-only natural-language query over safe usage views."""

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
from fiscal_rag.text_to_sql import (  # noqa: E402
    DEFAULT_MAX_ROWS,
    DEFAULT_QUERY_TIMEOUT_SECONDS,
    HARD_MAX_ROWS,
    HARD_QUERY_TIMEOUT_SECONDS,
    DeepSeekSQLGenerator,
    ReadOnlyUsageQueryExecutor,
    render_text_to_sql_markdown,
    run_usage_text_to_sql,
)


DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "data_private" / "usage" / "text_to_sql"


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query production usage metadata with natural language."
    )
    parser.add_argument("--question", required=True)
    parser.add_argument("--database", type=Path, default=DEFAULT_USAGE_DATABASE)
    parser.add_argument(
        "--max-rows",
        type=bounded_rows,
        default=DEFAULT_MAX_ROWS,
    )
    parser.add_argument(
        "--timeout-seconds",
        type=bounded_timeout,
        default=DEFAULT_QUERY_TIMEOUT_SECONDS,
    )
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--markdown-file", type=Path, default=None)
    return parser.parse_args(argv)


def bounded_rows(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= HARD_MAX_ROWS:
        raise argparse.ArgumentTypeError(
            f"value must be between 1 and {HARD_MAX_ROWS}"
        )
    return parsed


def bounded_timeout(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= HARD_QUERY_TIMEOUT_SECONDS:
        raise argparse.ArgumentTypeError(
            f"value must be greater than 0 and at most {HARD_QUERY_TIMEOUT_SECONDS}"
        )
    return parsed


def write_query_artifact_pair(
    json_path: Path,
    markdown_path: Path,
    record: dict[str, object],
) -> None:
    """Create both private result formats or remove newly created partial files."""
    if json_path.resolve() == markdown_path.resolve():
        raise ValueError("JSON and Markdown output paths must be different")
    if json_path.exists() or markdown_path.exists():
        existing = json_path if json_path.exists() else markdown_path
        raise FileExistsError(existing)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        with json_path.open("x", encoding="utf-8", newline="\n") as output:
            created.append(json_path)
            json.dump(record, output, ensure_ascii=False, indent=2)
            output.write("\n")
        with markdown_path.open("x", encoding="utf-8", newline="\n") as output:
            created.append(markdown_path)
            output.write(render_text_to_sql_markdown(record))
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def main() -> None:
    args = parse_arguments()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_file = args.output_file or DEFAULT_OUTPUT_DIRECTORY / f"query_{run_id}.json"
    markdown_file = args.markdown_file or output_file.with_suffix(".md")
    record = run_usage_text_to_sql(
        args.question,
        generator=DeepSeekSQLGenerator(),
        executor=ReadOnlyUsageQueryExecutor(
            args.database,
            max_rows=args.max_rows,
            timeout_seconds=args.timeout_seconds,
        ),
        retention_days=DEFAULT_USAGE_RETENTION_DAYS,
    )
    write_query_artifact_pair(output_file, markdown_file, record)
    print(f"Query status: {record['status']}")
    print(f"Result rows: {record['row_count']}")
    print(f"Private JSON: {output_file}")
    print(f"Private Markdown: {markdown_file}")


if __name__ == "__main__":
    main()

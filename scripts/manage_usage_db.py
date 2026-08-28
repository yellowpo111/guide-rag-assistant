"""Explicit backup and retention maintenance for private usage SQLite."""

from __future__ import annotations

import argparse
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
from fiscal_rag.usage_analysis import prune_expired_usage_artifacts  # noqa: E402


DEFAULT_USAGE_DIRECTORY = PROJECT_ROOT / "data_private" / "usage"


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Maintain private usage SQLite.")
    parser.add_argument("--database", type=Path, default=DEFAULT_USAGE_DATABASE)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prune = subparsers.add_parser("prune", help="Delete records beyond retention.")
    prune.add_argument(
        "--retention-days", type=positive_integer, default=DEFAULT_USAGE_RETENTION_DAYS
    )
    prune.add_argument("--artifacts-directory", type=Path, default=DEFAULT_USAGE_DIRECTORY)
    backup = subparsers.add_parser("backup", help="Create a consistent SQLite backup.")
    backup.add_argument("--output-file", type=Path, default=None)
    backup.add_argument(
        "--retention-days", type=positive_integer, default=DEFAULT_USAGE_RETENTION_DAYS
    )
    return parser.parse_args(argv)


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main() -> None:
    args = parse_arguments()
    repository = SQLiteUsageRepository(args.database)
    repository.initialize()
    if args.command == "prune":
        deleted = repository.prune_expired(args.retention_days)
        deleted_artifacts = prune_expired_usage_artifacts(
            args.artifacts_directory,
            args.retention_days,
        )
        print(f"Expired requests deleted: {deleted}")
        print(f"Expired usage artifacts deleted: {len(deleted_artifacts)}")
        return
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_file = args.output_file or (
        PROJECT_ROOT / "data_private" / "usage" / "backups" / f"usage_{run_id}.sqlite3"
    )
    deleted = repository.prune_expired(args.retention_days)
    repository.backup(output_file)
    if deleted:
        print(f"Expired requests deleted before backup: {deleted}")
    print(f"Backup created: {output_file}")


if __name__ == "__main__":
    main()

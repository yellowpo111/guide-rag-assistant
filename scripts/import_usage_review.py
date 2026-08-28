"""Validate and import user-confirmed private usage reviews."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fiscal_rag.settings import DEFAULT_USAGE_DATABASE  # noqa: E402
from fiscal_rag.usage import SQLiteUsageRepository  # noqa: E402
from fiscal_rag.usage_analysis import load_usage_reviews  # noqa: E402


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import confirmed usage reviews.")
    parser.add_argument("--database", type=Path, default=DEFAULT_USAGE_DATABASE)
    parser.add_argument("--reviews-file", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_arguments()
    repository = SQLiteUsageRepository(args.database)
    repository.initialize()
    reviews = load_usage_reviews(args.reviews_file)
    repository.save_reviews(reviews)
    print(f"Reviews imported: {len(reviews)}")
    print(f"Source artifact preserved: {args.reviews_file}")


if __name__ == "__main__":
    main()

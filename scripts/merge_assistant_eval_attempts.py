"""Merge non-overwriting Assistant Eval attempts into one selected result set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fiscal_rag.assistant_evaluation import (  # noqa: E402
    load_assistant_eval_cases,
    load_jsonl_records,
    merge_attempt_records,
    summarize_assistant_results,
    write_jsonl_exclusive,
)


DEFAULT_CASES_FILE = PROJECT_ROOT / "data_private" / "evals" / "assistant_eval_v1.jsonl"


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Assistant Eval retry attempts.")
    parser.add_argument(
        "--details-file",
        action="append",
        type=Path,
        required=True,
        help="Attempt details file; repeat in any order.",
    )
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--summary-file", type=Path, default=None)
    parser.add_argument("--cases-file", type=Path, default=DEFAULT_CASES_FILE)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_arguments()
    cases = load_assistant_eval_cases(args.cases_file)
    merged = merge_attempt_records(
        cases,
        [load_jsonl_records(path) for path in args.details_file],
    )
    write_jsonl_exclusive(args.output_file, merged)
    summary_file = args.summary_file or args.output_file.with_suffix(".summary.json")
    write_jsonl_exclusive(summary_file, [summarize_assistant_results(cases, merged)])
    print(f"Merged records: {len(merged)}")
    print(f"Output File: {args.output_file}")
    print(f"Summary File: {summary_file}")


if __name__ == "__main__":
    main()

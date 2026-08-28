"""Create a non-overwriting human-review template from one Assistant Eval run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fiscal_rag.assistant_evaluation import (  # noqa: E402
    adjudication_template_records,
    load_assistant_eval_cases,
    load_jsonl_records,
    write_jsonl_exclusive,
)


DEFAULT_CASES_FILE = PROJECT_ROOT / "data_private" / "evals" / "assistant_eval_v1.jsonl"


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an Assistant Eval adjudication template.")
    parser.add_argument("--details-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--cases-file", type=Path, default=DEFAULT_CASES_FILE)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_arguments()
    records = adjudication_template_records(
        load_assistant_eval_cases(args.cases_file),
        load_jsonl_records(args.details_file),
    )
    write_jsonl_exclusive(args.output_file, records)
    print(f"Adjudication records: {len(records)}")
    print(f"Output File: {args.output_file}")


if __name__ == "__main__":
    main()

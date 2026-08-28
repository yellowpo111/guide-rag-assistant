"""Export confirmed usage reviews as a new Assistant Eval candidate file."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fiscal_rag.assistant_evaluation import write_jsonl_exclusive  # noqa: E402
from fiscal_rag.settings import DEFAULT_USAGE_DATABASE  # noqa: E402
from fiscal_rag.usage import SQLiteUsageRepository  # noqa: E402
from fiscal_rag.usage_analysis import (  # noqa: E402
    eval_candidate_records,
    load_existing_eval_identity,
)


DEFAULT_EXISTING_CASES_FILE = (
    PROJECT_ROOT / "data_private" / "evals" / "assistant_eval_v1.jsonl"
)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export reviewed Assistant Eval candidates.")
    parser.add_argument("--database", type=Path, default=DEFAULT_USAGE_DATABASE)
    parser.add_argument(
        "--existing-cases-file",
        action="append",
        type=Path,
        default=None,
        help=(
            "Additional Assistant Eval JSONL used for duplicate checks; repeatable. "
            "The frozen v1 dataset is always checked."
        ),
    )
    parser.add_argument("--output-file", type=Path, default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_arguments()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_file = args.output_file or (
        PROJECT_ROOT
        / "data_private"
        / "evals"
        / "candidates"
        / f"assistant_eval_usage_candidates_{run_id}.jsonl"
    )
    existing_paths = [
        DEFAULT_EXISTING_CASES_FILE,
        *(args.existing_cases_file or []),
    ]
    existing_questions, existing_case_ids = load_existing_eval_identity(
        list(dict.fromkeys(existing_paths))
    )
    repository = SQLiteUsageRepository(args.database)
    repository.initialize()
    candidates = eval_candidate_records(
        repository.fetch_usage_records(),
        existing_questions=existing_questions,
        existing_case_ids=existing_case_ids,
    )
    write_jsonl_exclusive(output_file, candidates)
    print(f"Eval candidates: {len(candidates)}")
    print(f"Private output: {output_file}")


if __name__ == "__main__":
    main()

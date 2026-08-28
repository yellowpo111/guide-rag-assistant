"""Run the Text-to-SQL prototype against a deterministic synthetic fixture."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fiscal_rag.text_to_sql import DeepSeekSQLGenerator  # noqa: E402
from fiscal_rag.text_to_sql_evaluation import (  # noqa: E402
    build_synthetic_usage_fixture,
    evaluate_text_to_sql_cases,
    load_text_to_sql_eval_cases,
)


DEFAULT_CASES_FILE = (
    PROJECT_ROOT / "tests" / "fixtures" / "text_to_sql_usage_eval_v1.jsonl"
)
DEFAULT_RESULTS_DIRECTORY = PROJECT_ROOT / "data_private" / "evals" / "results"


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate usage Text-to-SQL with synthetic data."
    )
    parser.add_argument("--cases-file", type=Path, default=DEFAULT_CASES_FILE)
    parser.add_argument("--details-file", type=Path, default=None)
    parser.add_argument("--summary-file", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args(argv)


def write_eval_artifact_pair(
    details_file: Path,
    summary_file: Path,
    details: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    if details_file.resolve() == summary_file.resolve():
        raise ValueError("Details and summary output paths must be different")
    if details_file.exists() or summary_file.exists():
        existing = details_file if details_file.exists() else summary_file
        raise FileExistsError(existing)
    details_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        with details_file.open("x", encoding="utf-8", newline="\n") as output:
            created.append(details_file)
            for record in details:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        with summary_file.open("x", encoding="utf-8", newline="\n") as output:
            created.append(summary_file)
            json.dump(summary, output, ensure_ascii=False, indent=2)
            output.write("\n")
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def main() -> None:
    args = parse_arguments()
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    details_file = args.details_file or (
        DEFAULT_RESULTS_DIRECTORY / f"text_to_sql_usage_v1_{run_id}.jsonl"
    )
    summary_file = args.summary_file or details_file.with_suffix(".summary.json")
    cases = load_text_to_sql_eval_cases(args.cases_file)
    with tempfile.TemporaryDirectory(
        prefix="text-to-sql-eval-", dir=PROJECT_ROOT / "data_private"
    ) as temporary_directory:
        database = build_synthetic_usage_fixture(
            Path(temporary_directory) / "synthetic_usage.sqlite3"
        )
        details, summary = evaluate_text_to_sql_cases(
            cases,
            generator=DeepSeekSQLGenerator(),
            database_path=database,
            run_id=run_id,
        )
    write_eval_artifact_pair(details_file, summary_file, details, summary)
    print(f"Synthetic cases: {summary['total_cases']}")
    print(f"Answerable matches: {summary['answerable_matches']}/10")
    print(f"Refusal matches: {summary['refusal_matches']}/2")
    print(f"Prototype decision: {summary['prototype_decision']}")
    print(f"Private details: {details_file}")
    print(f"Private summary: {summary_file}")


if __name__ == "__main__":
    main()

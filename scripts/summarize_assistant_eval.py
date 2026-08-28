"""Combine automatic Assistant Eval results with user-confirmed adjudications."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fiscal_rag.assistant_evaluation import (  # noqa: E402
    load_assistant_adjudications,
    load_assistant_eval_cases,
    load_jsonl_records,
    summarize_adjudications,
    summarize_assistant_results,
)


DEFAULT_CASES_FILE = PROJECT_ROOT / "data_private" / "evals" / "assistant_eval_v1.jsonl"


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize Assistant Eval after human review.")
    parser.add_argument("--details-file", type=Path, required=True)
    parser.add_argument("--adjudications-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--cases-file", type=Path, default=DEFAULT_CASES_FILE)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_arguments()
    cases = load_assistant_eval_cases(args.cases_file)
    automatic = summarize_assistant_results(cases, load_jsonl_records(args.details_file))
    human = summarize_adjudications(
        cases,
        load_assistant_adjudications(args.adjudications_file),
    )
    blockers = sorted(
        set(automatic["critical_rag_route_failures"])
        | set(human["release_blockers"])
    )
    if automatic["error_count"]:
        blockers.append("automatic_execution_errors")
    report = {
        "schema_version": "assistant-eval-final-summary-v1",
        "automatic": automatic,
        "human_adjudication": human,
        "release_gate": "passed" if not blockers else "blocked",
        "release_blockers": blockers,
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Output File: {args.output_file}")


if __name__ == "__main__":
    main()

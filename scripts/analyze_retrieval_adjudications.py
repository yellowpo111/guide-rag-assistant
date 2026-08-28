"""Report strict and adjudicated usefulness metrics from saved private details."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fiscal_rag.adjudications import (  # noqa: E402
    load_retrieval_adjudications,
    summarize_saved_details,
    validate_adjudications_against_corpus,
)


DEFAULT_CORPUS_DIR = PROJECT_ROOT / "data_private" / "corpus"


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare frozen strict metrics with supplementary adjudicated usefulness."
    )
    parser.add_argument("--details-file", type=Path, required=True)
    parser.add_argument("--adjudications-file", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    return parser.parse_args(argv)


def load_details_records(path: str | Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"Details line {line_number} must be an object.")
        records.append(record)
    if not records:
        raise ValueError("Details JSONL contains no records.")
    return records


def main() -> None:
    args = parse_arguments()
    adjudications = load_retrieval_adjudications(args.adjudications_file)
    validation_errors = validate_adjudications_against_corpus(
        adjudications, args.corpus_dir
    )
    if validation_errors:
        raise ValueError("Adjudication validation failed:\n- " + "\n- ".join(validation_errors))

    summary = summarize_saved_details(
        load_details_records(args.details_file), adjudications
    )
    print("Retrieval Relevance Adjudication Analysis")
    print(f"Details File: {args.details_file}")
    print(f"Adjudications File: {args.adjudications_file}")
    print(f"Corpus Directory: {args.corpus_dir}")
    print(f"Total Cases: {summary.total_cases}")
    print(f"Supplementary Adjudications: {len(adjudications)}")
    print(f"Strict Hit@1: {summary.strict_hit_at_1:.6f}")
    print(f"Adjudicated Useful Hit@1: {summary.adjudicated_hit_at_1:.6f}")
    print(f"Strict MRR: {summary.strict_mrr:.6f}")
    print(f"Adjudicated Useful MRR: {summary.adjudicated_mrr:.6f}")
    print(
        "Strict-only Rank-1 Overrides: "
        f"{len(summary.strict_only_rank_1_overrides)}"
    )
    for result in summary.strict_only_rank_1_overrides:
        print(f"- {result.case_id}")


if __name__ == "__main__":
    main()

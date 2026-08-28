"""Verify the private v1.4 code candidate and write an auditable manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fiscal_rag.embeddings import QwenEmbeddings  # noqa: E402
from fiscal_rag.ingestion import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE  # noqa: E402
from fiscal_rag.pipeline import DENSE_RERANK_CANDIDATE_K  # noqa: E402
from fiscal_rag.service import FINAL_CONTEXT_K  # noqa: E402
from fiscal_rag.settings import (  # noqa: E402
    DEFAULT_CORPUS_DIRECTORY,
    DEFAULT_INDEX_DIRECTORY,
    DEFAULT_USAGE_DATABASE,
    DEFAULT_USAGE_RETENTION_DAYS,
)
from fiscal_rag.text_to_sql import (  # noqa: E402
    SAFE_VIEW_NAMES,
    TEXT_TO_SQL_PROMPT_VERSION,
    TEXT_TO_SQL_SCHEMA_VERSION,
    ReadOnlyUsageQueryExecutor,
    TextToSQLError,
    build_text_to_sql_prompt,
)
from fiscal_rag.text_to_sql_evaluation import (  # noqa: E402
    SYNTHETIC_DATA_ORIGIN,
    build_synthetic_usage_fixture,
)
from fiscal_rag.usage import SQLiteUsageRepository, USAGE_SCHEMA_VERSION  # noqa: E402
from fiscal_rag.usage_analysis import USAGE_SUMMARY_SCHEMA_VERSION  # noqa: E402
from fiscal_rag.vector_store import build_persistent_chroma_index  # noqa: E402
from fiscal_rag.version import __version__  # noqa: E402


RELEASE_VERSION = "1.4.0"
V1_2_RELEASE_VERSION = "1.2.0"
FROZEN_BASELINE_VERSION = "1.1.0"
EXPECTED_PYTHON = (3, 13)
EXPECTED_DOCUMENTS = 29
EXPECTED_CHUNKS = 1000
EXPECTED_EMBEDDING_DIMENSION = 1024
EXPECTED_EMBEDDING_MODEL = "qwen3.7-text-embedding"
EXPECTED_RERANK_MODEL = "qwen3-rerank"
EXPECTED_GENERATION_MODEL = "deepseek-v4-flash"
REQUIRED_CONFIGURATION = (
    "DASHSCOPE_API_KEY",
    "DASHSCOPE_BASE_URL",
    "DASHSCOPE_EMBEDDING_MODEL",
    "DASHSCOPE_RERANK_BASE_URL",
    "DEEPSEEK_API_KEY",
)
REQUIRED_USAGE_EXAMPLE_CONFIGURATION = (
    "FISCAL_RAG_USAGE_DB_PATH",
    "FISCAL_RAG_USAGE_RETENTION_DAYS",
)
PRIVATE_EVAL_ARTIFACTS = (
    "data_private/evals/retrieval_eval_v1.jsonl",
    "data_private/evals/retrieval_eval_v2_holdout.jsonl",
    "data_private/evals/retrieval_eval_v4_general_holdout.jsonl",
    "data_private/evals/assistant_eval_v1.jsonl",
    "data_private/evals/performance_eval_v1.jsonl",
)
EXPECTED_PRIVATE_EVAL_RECORDS = {
    "data_private/evals/retrieval_eval_v1.jsonl": 50,
    "data_private/evals/retrieval_eval_v2_holdout.jsonl": 15,
    "data_private/evals/retrieval_eval_v4_general_holdout.jsonl": 16,
    "data_private/evals/assistant_eval_v1.jsonl": 54,
    "data_private/evals/performance_eval_v1.jsonl": 16,
}
RELEASE_EVIDENCE_JSONL = (
    "data_private/evals/results/assistant_eval_v1_20260826T160100Z_attempt_1.jsonl",
    "data_private/evals/results/assistant_eval_v1_20260826T160100Z_attempt_2.jsonl",
    "data_private/evals/results/assistant_eval_v1_20260826T160100Z_attempt_3.jsonl",
    "data_private/evals/results/assistant_eval_v1_20260826T160100Z_selected.jsonl",
    "data_private/evals/results/assistant_eval_v1_20260826T160100Z_user_confirmed.jsonl",
    "data_private/evals/results/performance_eval_v1_20260826T163218Z.jsonl",
)
EXPECTED_RELEASE_EVIDENCE_RECORDS = dict(
    zip(RELEASE_EVIDENCE_JSONL, (54, 1, 1, 54, 54, 70), strict=True)
)
ASSISTANT_FINAL_SUMMARY = (
    "data_private/evals/results/"
    "assistant_eval_v1_20260826T160100Z_final.summary.json"
)
PERFORMANCE_FINAL_SUMMARY = (
    "data_private/evals/results/performance_eval_v1_20260826T163218Z.summary.json"
)
STARTUP_READY_RESULT = (
    "data_private/evals/results/startup_ready_v1_20260826T083728Z.json"
)
V1_2_ASSISTANT_ATTEMPT_1 = (
    "data_private/evals/results/assistant_eval_v1_20260827T023500Z_attempt_1.jsonl"
)
V1_2_ASSISTANT_ATTEMPT_2 = (
    "data_private/evals/results/assistant_eval_v1_20260827T023500Z_attempt_2.jsonl"
)
V1_2_ASSISTANT_SELECTED = (
    "data_private/evals/results/assistant_eval_v1_20260827T023500Z_selected.jsonl"
)
V1_2_ASSISTANT_SUMMARY = (
    "data_private/evals/results/assistant_eval_v1_20260827T023500Z_selected.summary.json"
)
V1_2_PERFORMANCE_ATTEMPTS = (
    "data_private/evals/results/performance_eval_v1_20260827T023900Z.jsonl",
    "data_private/evals/results/performance_eval_v1_20260827T024500Z.jsonl",
    "data_private/evals/results/performance_eval_v1_20260827T025300Z.jsonl",
)
V1_2_PERFORMANCE_SUMMARY = (
    "data_private/evals/results/performance_eval_v1_20260827T025300Z.summary.json"
)
V1_2_STARTUP_RESULT = (
    "data_private/evals/results/startup_ready_v1_2_20260827T022954Z.json"
)
V1_2_USAGE_BACKUP = (
    "data_private/usage/backups/usage_v1_2_release_20260827T025800Z.sqlite3"
)
TEXT_TO_SQL_EVAL_GLOB = "text_to_sql_usage_v1_*.summary.json"


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the Fiscal RAG v1.4 code candidate.")
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_arguments()
    _verify_static_release_contract()
    _verify_usage_store_contract()
    _verify_text_to_sql_contract()
    configuration = _load_and_verify_configuration()
    pip_result = _run_checked([sys.executable, "-m", "pip", "check"])
    git_commit = _run_checked(["git", "rev-parse", "HEAD"])["stdout"].strip()
    tracked_private = _run_checked(
        ["git", "ls-files", "--", ".env", "data_private"]
    )["stdout"].strip()
    if tracked_private:
        raise RuntimeError("Private paths are tracked by Git: " + tracked_private)
    dirty = _run_checked(["git", "status", "--porcelain"])["stdout"].strip()
    if dirty and not args.allow_dirty:
        raise RuntimeError("Git worktree is not clean; use --allow-dirty only for preflight.")

    index_result = build_persistent_chroma_index(
        DEFAULT_CORPUS_DIRECTORY,
        DEFAULT_INDEX_DIRECTORY,
        QwenEmbeddings(),
    )
    if index_result.created:
        raise RuntimeError("Release verification must reuse an existing current index.")
    _verify_index_manifest(index_result.manifest)
    _verify_model_identifiers(configuration, index_result.manifest)
    release_evidence = _verify_release_evidence()
    v1_2_release_evidence = _verify_v1_2_release_evidence()
    text_to_sql_evidence = _verify_text_to_sql_evidence()

    test_result: dict[str, object]
    if args.skip_tests:
        test_result = {"status": "skipped"}
    else:
        with tempfile.TemporaryDirectory(
            prefix="release-pytest-", dir=PROJECT_ROOT / "data_private"
        ) as temporary_directory:
            test_result = _run_checked(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "tests",
                    f"--basetemp={Path(temporary_directory) / 'pytest'}",
                ]
            )
            test_result["status"] = "passed"

    created_at = datetime.now(UTC)
    output_file = args.output_file or (
        PROJECT_ROOT
        / "data_private"
        / "releases"
        / RELEASE_VERSION
        / f"release_manifest_{created_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    manifest = {
        "schema_version": "fiscal-rag-release-manifest-v1",
        "release_version": RELEASE_VERSION,
        "created_at_utc": created_at.isoformat(),
        "git_commit": git_commit,
        "worktree_clean": not bool(dirty),
        "python": sys.version.split()[0],
        "profile": {
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
            "dense_candidate_k": DENSE_RERANK_CANDIDATE_K,
            "final_context_k": FINAL_CONTEXT_K,
            "embedding_model": index_result.manifest["embedding_model"],
            "embedding_dimension": index_result.manifest["embedding_dimension"],
            "rerank_model": configuration.get("DASHSCOPE_RERANK_MODEL", "qwen3-rerank"),
            "generation_model": configuration.get("DEEPSEEK_MODEL")
            or EXPECTED_GENERATION_MODEL,
            "raw_document_count": index_result.raw_document_count,
            "chunk_count": index_result.chunk_count,
            "corpus_sha256": index_result.manifest["corpus_sha256"],
        },
        "usage_store": {
            "schema_version": USAGE_SCHEMA_VERSION,
            "summary_schema_version": USAGE_SUMMARY_SCHEMA_VERSION,
            "default_retention_days": DEFAULT_USAGE_RETENTION_DAYS,
            "default_path": DEFAULT_USAGE_DATABASE.relative_to(PROJECT_ROOT).as_posix(),
            "traffic_kinds": ["production", "assistant_eval", "performance_eval"],
        },
        "operational_trial_gate": {
            "status": "pending_manual_evidence",
            "minimum_business_days": 5,
            "requires_reviewed_production_request": True,
            "required_before_tag": "v1.3.0",
        },
        "text_to_sql_experiment": {
            "schema_version": TEXT_TO_SQL_SCHEMA_VERSION,
            "prompt_version": TEXT_TO_SQL_PROMPT_VERSION,
            "data_scope": "production_safe_views",
            "safe_views": sorted(SAFE_VIEW_NAMES),
            "online_integration": False,
            "evaluation": text_to_sql_evidence,
        },
        "requirements_lock_sha256": _sha256(PROJECT_ROOT / "requirements.lock.txt"),
        "private_eval_artifacts": [_artifact_record(path) for path in PRIVATE_EVAL_ARTIFACTS],
        "frozen_v1_1_release_evidence_artifacts": release_evidence,
        "v1_2_release_evidence_artifacts": v1_2_release_evidence,
        "pip_check": {"status": "passed", "output": pip_result["stdout"].strip()},
        "tests": test_result,
        "private_paths_tracked": False,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    print("Release verification passed.")
    print(f"Release Version: {RELEASE_VERSION}")
    print(f"Git Commit: {git_commit}")
    print(f"Manifest: {output_file}")


def _verify_static_release_contract() -> None:
    if __version__ != RELEASE_VERSION:
        raise RuntimeError("Package version does not match release verifier.")
    if sys.version_info[:2] != EXPECTED_PYTHON:
        raise RuntimeError(
            f"Expected Python {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]}, "
            f"got {sys.version_info.major}.{sys.version_info.minor}."
        )
    if (DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP) != (1000, 100):
        raise RuntimeError("Chunk profile is not the frozen 1000/100 configuration.")
    if DENSE_RERANK_CANDIDATE_K != 20 or FINAL_CONTEXT_K != 5:
        raise RuntimeError("Retrieval profile is not the frozen Dense-20 / Context-5 configuration.")
    if (
        DEFAULT_USAGE_RETENTION_DAYS != 90
        or USAGE_SCHEMA_VERSION != 1
        or USAGE_SUMMARY_SCHEMA_VERSION != "usage-summary-v2"
    ):
        raise RuntimeError("Usage persistence or analytics contract is not the v1.4 configuration.")
    if DEFAULT_USAGE_DATABASE.parent == DEFAULT_INDEX_DIRECTORY or DEFAULT_INDEX_DIRECTORY in DEFAULT_USAGE_DATABASE.parents:
        raise RuntimeError("Usage SQLite must be separate from the Chroma index directory.")


def _load_and_verify_configuration() -> dict[str, str | None]:
    example_keys = {
        line.split("=", 1)[0].strip()
        for line in (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
    }
    missing_example = sorted(
        (set(REQUIRED_CONFIGURATION) | set(REQUIRED_USAGE_EXAMPLE_CONFIGURATION))
        - example_keys
    )
    if missing_example:
        raise RuntimeError(".env.example is missing required keys: " + ", ".join(missing_example))
    file_values = dotenv_values(PROJECT_ROOT / ".env")
    configuration = {
        name: os.getenv(name) or file_values.get(name) for name in example_keys
    }
    missing_runtime = sorted(
        name for name in REQUIRED_CONFIGURATION if not configuration.get(name)
    )
    if missing_runtime:
        raise RuntimeError("Runtime configuration is missing: " + ", ".join(missing_runtime))
    return configuration


def _verify_index_manifest(manifest: object) -> None:
    if not isinstance(manifest, dict):
        raise RuntimeError("Index manifest is not an object.")
    expected = {
        "raw_document_count": EXPECTED_DOCUMENTS,
        "chunk_count": EXPECTED_CHUNKS,
        "embedding_dimension": EXPECTED_EMBEDDING_DIMENSION,
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
    }
    mismatches = [
        f"{field}={manifest.get(field)!r}, expected {value!r}"
        for field, value in expected.items()
        if manifest.get(field) != value
    ]
    if mismatches:
        raise RuntimeError("Index manifest mismatch: " + "; ".join(mismatches))


def _verify_usage_store_contract() -> None:
    with tempfile.TemporaryDirectory(
        prefix="release-usage-", dir=_private_data_directory()
    ) as temporary_directory:
        repository = SQLiteUsageRepository(Path(temporary_directory) / "usage.sqlite3")
        repository.initialize()
        with repository._connect() as connection:  # noqa: SLF001
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            required_tables = {
                "assistant_requests",
                "request_timings",
                "request_sources",
                "request_feedback",
                "request_reviews",
            }
            if not required_tables.issubset(tables):
                raise RuntimeError("Usage SQLite is missing required v1.2 tables.")
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(assistant_requests)")
            }
            if {"context", "page_content", "prompt"}.intersection(columns):
                raise RuntimeError("Usage SQLite contains prohibited private context fields.")


def _verify_text_to_sql_contract() -> None:
    expected_views = {
        "usage_requests",
        "usage_timings",
        "usage_feedback",
        "usage_reviews",
        "usage_source_stats",
    }
    if (
        TEXT_TO_SQL_SCHEMA_VERSION != "usage-text-to-sql-v1"
        or TEXT_TO_SQL_PROMPT_VERSION != "usage-text-to-sql-prompt-v1"
        or SAFE_VIEW_NAMES != expected_views
    ):
        raise RuntimeError("Text-to-SQL schema or prompt identity changed unexpectedly.")
    prompt = build_text_to_sql_prompt("Count completed requests by route.")
    if any(
        prohibited in prompt
        for prohibited in (
            "assistant_requests(",
            "request_sources(",
            "question TEXT",
            "answer TEXT",
            "reason TEXT",
            "source TEXT",
        )
    ):
        raise RuntimeError("Text-to-SQL prompt exposes a raw table or sensitive column.")
    with tempfile.TemporaryDirectory(
        prefix="release-text-to-sql-", dir=_private_data_directory()
    ) as temporary_directory:
        database = build_synthetic_usage_fixture(
            Path(temporary_directory) / "synthetic_usage.sqlite3"
        )
        before = _sha256(database)
        executor = ReadOnlyUsageQueryExecutor(database)
        result = executor.execute("SELECT COUNT(*) FROM usage_requests")
        if result.rows != ((11,),):
            raise RuntimeError("Text-to-SQL safe view does not isolate production traffic.")
        for sql in (
            "SELECT question FROM assistant_requests",
            "SELECT * FROM sqlite_master",
            "PRAGMA user_version",
            "DELETE FROM assistant_requests",
        ):
            try:
                executor.execute(sql)
            except TextToSQLError as error:
                if error.code != "sql_rejected":
                    raise RuntimeError(
                        "Text-to-SQL unsafe query returned an unstable error code."
                    ) from error
            else:
                raise RuntimeError("Text-to-SQL unsafe query was not rejected.")
        if _sha256(database) != before:
            raise RuntimeError("Text-to-SQL verification changed its SQLite fixture.")


def _private_data_directory() -> Path:
    """Return the ignored private-data root, creating it for verifier scratch files."""
    directory = PROJECT_ROOT / "data_private"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _verify_model_identifiers(
    configuration: dict[str, str | None],
    manifest: dict[str, object],
) -> None:
    actual = {
        "embedding_model": manifest.get("embedding_model"),
        "rerank_model": configuration.get("DASHSCOPE_RERANK_MODEL")
        or EXPECTED_RERANK_MODEL,
        "generation_model": configuration.get("DEEPSEEK_MODEL")
        or EXPECTED_GENERATION_MODEL,
    }
    expected = {
        "embedding_model": EXPECTED_EMBEDDING_MODEL,
        "rerank_model": EXPECTED_RERANK_MODEL,
        "generation_model": EXPECTED_GENERATION_MODEL,
    }
    mismatches = [
        f"{name}={actual[name]!r}, expected {value!r}"
        for name, value in expected.items()
        if actual[name] != value
    ]
    if mismatches:
        raise RuntimeError("Model identifier mismatch: " + "; ".join(mismatches))


def _verify_release_evidence() -> list[dict[str, object]]:
    jsonl_records = {
        path: _load_jsonl_objects(path) for path in RELEASE_EVIDENCE_JSONL
    }
    for path, records in jsonl_records.items():
        expected = EXPECTED_RELEASE_EVIDENCE_RECORDS[path]
        if len(records) != expected:
            raise RuntimeError(
                f"Release evidence {path} has {len(records)} records; expected {expected}."
            )

    attempt_1, attempt_2, attempt_3, selected, confirmed, performance = (
        jsonl_records[path] for path in RELEASE_EVIDENCE_JSONL
    )
    if not any(record.get("status") != "completed" for record in attempt_1):
        raise RuntimeError("Assistant attempt 1 no longer preserves its failed execution.")
    if not any(record.get("status") != "completed" for record in attempt_2):
        raise RuntimeError("Assistant attempt 2 no longer preserves its failed retry.")
    if any(record.get("status") != "completed" for record in attempt_3):
        raise RuntimeError("Assistant attempt 3 must contain the successful retry.")
    if any(record.get("status") != "completed" for record in selected):
        raise RuntimeError("Assistant selected results must all be completed.")
    if len({record.get("case_id") for record in selected}) != 54:
        raise RuntimeError("Assistant selected results must contain 54 unique case IDs.")
    if any(record.get("review_status") != "user_confirmed" for record in confirmed):
        raise RuntimeError("Assistant adjudications must all be user-confirmed.")
    if any(not str(record.get("reason", "")).strip() for record in confirmed):
        raise RuntimeError("Assistant adjudications must all include a review reason.")

    run_kind_counts = {
        kind: sum(record.get("run_kind") == kind for record in performance)
        for kind in ("warmup", "measured", "concurrency")
    }
    if run_kind_counts != {"warmup": 16, "measured": 48, "concurrency": 6}:
        raise RuntimeError(
            f"Performance evidence has unexpected run-kind counts: {run_kind_counts}."
        )
    prohibited_fields = {"question", "answer", "context", "Context"}
    if any(prohibited_fields.intersection(record) for record in performance):
        raise RuntimeError("Performance evidence contains question, answer, or context data.")
    if any(
        not isinstance(record.get("environment"), dict)
        or record["environment"].get("label")
        != "internal-deployment-i5-10400-32gb"
        for record in performance
    ):
        raise RuntimeError("Performance evidence does not match the deployment environment label.")

    assistant_summary = _load_json_object(ASSISTANT_FINAL_SUMMARY)
    performance_summary = _load_json_object(PERFORMANCE_FINAL_SUMMARY)
    startup_result = _load_json_object(STARTUP_READY_RESULT)
    _verify_assistant_summary(assistant_summary)
    _verify_performance_summary(performance_summary)
    _verify_startup_result(startup_result)

    artifacts = [_artifact_record(path) for path in RELEASE_EVIDENCE_JSONL]
    artifacts.extend(
        _json_artifact_record(path)
        for path in (
            ASSISTANT_FINAL_SUMMARY,
            PERFORMANCE_FINAL_SUMMARY,
            STARTUP_READY_RESULT,
        )
    )
    return artifacts


def _verify_assistant_summary(summary: dict[str, object]) -> None:
    automatic = summary.get("automatic")
    human = summary.get("human_adjudication")
    if not isinstance(automatic, dict) or not isinstance(human, dict):
        raise RuntimeError("Assistant final summary is missing automatic or human results.")
    expected_automatic = {
        "total_cases": 54,
        "completed_cases": 54,
        "sse_completion_rate": 1.0,
        "route_accuracy": 1.0,
        "route_macro_f1": 1.0,
        "rag_trace_completion_rate": 1.0,
        "error_count": 0,
    }
    if any(automatic.get(key) != value for key, value in expected_automatic.items()):
        raise RuntimeError("Assistant automatic summary does not meet the release contract.")
    if (
        human.get("total_confirmed") != 54
        or human.get("release_blockers") != []
        or summary.get("release_gate") != "passed"
        or summary.get("release_blockers") != []
    ):
        raise RuntimeError("Assistant human adjudication release gate has not passed.")


def _verify_v1_2_release_evidence() -> list[dict[str, object]]:
    attempt_1 = _load_jsonl_objects(V1_2_ASSISTANT_ATTEMPT_1)
    attempt_2 = _load_jsonl_objects(V1_2_ASSISTANT_ATTEMPT_2)
    selected = _load_jsonl_objects(V1_2_ASSISTANT_SELECTED)
    if len(attempt_1) != 54 or sum(
        record.get("status") == "completed" for record in attempt_1
    ) != 51:
        raise RuntimeError("v1.2 Assistant attempt 1 must preserve 51/54 completion.")
    if len(attempt_2) != 3 or any(
        record.get("status") != "completed" for record in attempt_2
    ):
        raise RuntimeError("v1.2 Assistant attempt 2 must preserve three successful retries.")
    if len(selected) != 54 or any(
        record.get("status") != "completed" for record in selected
    ):
        raise RuntimeError("v1.2 Assistant selected results must complete all 54 cases.")
    if len({record.get("case_id") for record in selected}) != 54:
        raise RuntimeError("v1.2 Assistant selected results must have unique case IDs.")
    _verify_v1_2_assistant_summary(_load_json_object(V1_2_ASSISTANT_SUMMARY))

    expected_measured_completed = (47, 44, 48)
    for path, expected_completed in zip(
        V1_2_PERFORMANCE_ATTEMPTS, expected_measured_completed, strict=True
    ):
        records = _load_jsonl_objects(path)
        run_kind_counts = {
            kind: sum(record.get("run_kind") == kind for record in records)
            for kind in ("warmup", "measured", "concurrency")
        }
        if run_kind_counts != {"warmup": 16, "measured": 48, "concurrency": 6}:
            raise RuntimeError(f"v1.2 performance evidence has invalid counts: {path}")
        measured_completed = sum(
            record.get("run_kind") == "measured"
            and record.get("status") == "completed"
            for record in records
        )
        if measured_completed != expected_completed:
            raise RuntimeError(f"v1.2 performance completion changed: {path}")
        if any(
            record.get("release_version") != f"v{V1_2_RELEASE_VERSION}"
            for record in records
        ):
            raise RuntimeError(f"v1.2 performance evidence has wrong version: {path}")
        prohibited_fields = {"question", "answer", "context", "Context"}
        if any(prohibited_fields.intersection(record) for record in records):
            raise RuntimeError(f"v1.2 performance evidence contains private text: {path}")
    _verify_performance_summary(_load_json_object(V1_2_PERFORMANCE_SUMMARY))

    startup = _load_json_object(V1_2_STARTUP_RESULT)
    startup_ready_ms = startup.get("startup_ready_ms")
    if (
        startup.get("release_version") != f"v{V1_2_RELEASE_VERSION}"
        or startup.get("probe_path") != "/health/ready"
        or not isinstance(startup_ready_ms, (int, float))
        or isinstance(startup_ready_ms, bool)
        or startup_ready_ms <= 0
    ):
        raise RuntimeError("v1.2 cold-start result does not meet the release contract.")
    _verify_v1_2_usage_backup()

    artifacts = [
        _artifact_record(path)
        for path in (
            V1_2_ASSISTANT_ATTEMPT_1,
            V1_2_ASSISTANT_ATTEMPT_2,
            V1_2_ASSISTANT_SELECTED,
            *V1_2_PERFORMANCE_ATTEMPTS,
        )
    ]
    artifacts.extend(
        _json_artifact_record(path)
        for path in (
            V1_2_ASSISTANT_SUMMARY,
            V1_2_PERFORMANCE_SUMMARY,
            V1_2_STARTUP_RESULT,
        )
    )
    artifacts.append(
        {
            "path": V1_2_USAGE_BACKUP,
            "sha256": _sha256(PROJECT_ROOT / V1_2_USAGE_BACKUP),
        }
    )
    return artifacts


def _verify_v1_2_assistant_summary(summary: dict[str, object]) -> None:
    expected = {
        "total_cases": 54,
        "completed_cases": 54,
        "sse_completion_rate": 1.0,
        "route_accuracy": 1.0,
        "route_macro_f1": 1.0,
        "rag_trace_completion_rate": 1.0,
        "error_count": 0,
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise RuntimeError("v1.2 Assistant selected summary does not meet the release contract.")


def _verify_v1_2_usage_backup() -> None:
    path = PROJECT_ROOT / V1_2_USAGE_BACKUP
    if not path.is_file():
        raise RuntimeError("Missing v1.2 usage backup.")
    with sqlite3.connect(path) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        statuses = dict(
            connection.execute(
                "SELECT execution_status, COUNT(1) FROM assistant_requests "
                "GROUP BY execution_status"
            ).fetchall()
        )
        traffic = dict(
            connection.execute(
                "SELECT traffic_kind, COUNT(1) FROM assistant_requests "
                "GROUP BY traffic_kind"
            ).fetchall()
        )
    if integrity != "ok" or version != USAGE_SCHEMA_VERSION:
        raise RuntimeError("v1.2 usage backup integrity or schema version is invalid.")
    if statuses != {"completed": 260, "failed": 9}:
        raise RuntimeError("v1.2 usage backup terminal states changed.")
    if traffic != {"assistant_eval": 59, "performance_eval": 210}:
        raise RuntimeError("v1.2 usage backup traffic isolation changed.")


def _verify_text_to_sql_evidence() -> dict[str, object]:
    results_directory = PROJECT_ROOT / "data_private" / "evals" / "results"
    summaries = sorted(results_directory.glob(TEXT_TO_SQL_EVAL_GLOB))
    if not summaries:
        raise RuntimeError("Missing private Text-to-SQL synthetic evaluation evidence.")
    summary_path = summaries[-1]
    details_path = summary_path.with_name(
        summary_path.name.removesuffix(".summary.json") + ".jsonl"
    )
    if not details_path.is_file():
        raise RuntimeError("Text-to-SQL evaluation details are missing.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise RuntimeError("Text-to-SQL evaluation summary is not an object.")
    _verify_text_to_sql_summary(summary)
    details = []
    for line_number, line in enumerate(
        details_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise RuntimeError(
                f"Text-to-SQL evidence line {line_number} is not an object."
            )
        details.append(record)
    if len(details) != 12 or len({record.get("case_id") for record in details}) != 12:
        raise RuntimeError("Text-to-SQL evaluation must contain 12 unique cases.")
    if any(
        record.get("run_id") != summary.get("run_id")
        or record.get("data_origin") != SYNTHETIC_DATA_ORIGIN
        or record.get("contains_real_usage_data") is not False
        for record in details
    ):
        raise RuntimeError("Text-to-SQL evidence origin or run identity is invalid.")
    refusal_count = sum(record.get("expected_refusal") is True for record in details)
    answerable_matches = sum(
        record.get("expected_refusal") is False and record.get("correct") is True
        for record in details
    )
    refusal_matches = sum(
        record.get("expected_refusal") is True and record.get("correct") is True
        for record in details
    )
    if (
        refusal_count != 2
        or answerable_matches != summary.get("answerable_matches")
        or refusal_matches != summary.get("refusal_matches")
    ):
        raise RuntimeError("Text-to-SQL detail and summary outcomes do not agree.")
    details_relative = details_path.relative_to(PROJECT_ROOT).as_posix()
    summary_relative = summary_path.relative_to(PROJECT_ROOT).as_posix()
    return {
        "data_origin": SYNTHETIC_DATA_ORIGIN,
        "contains_real_usage_data": False,
        "prototype_decision": summary["prototype_decision"],
        "answerable_matches": summary["answerable_matches"],
        "refusal_matches": summary["refusal_matches"],
        "details_artifact": _artifact_record(details_relative),
        "summary_artifact": _json_artifact_record(summary_relative),
    }


def _verify_text_to_sql_summary(summary: dict[str, object]) -> None:
    answerable_matches = summary.get("answerable_matches")
    refusal_matches = summary.get("refusal_matches")
    if (
        summary.get("schema_version") != "usage-text-to-sql-eval-summary-v1"
        or summary.get("data_origin") != SYNTHETIC_DATA_ORIGIN
        or summary.get("contains_real_usage_data") is not False
        or summary.get("total_cases") != 12
        or summary.get("answerable_cases") != 10
        or summary.get("refusal_cases") != 2
        or not isinstance(answerable_matches, int)
        or isinstance(answerable_matches, bool)
        or not 0 <= answerable_matches <= 10
        or not isinstance(refusal_matches, int)
        or isinstance(refusal_matches, bool)
        or not 0 <= refusal_matches <= 2
    ):
        raise RuntimeError("Text-to-SQL evaluation summary is invalid.")
    expected_decision = (
        "adopted"
        if answerable_matches >= 8 and refusal_matches == 2
        else "not_adopted"
    )
    if summary.get("prototype_decision") != expected_decision:
        raise RuntimeError("Text-to-SQL prototype decision does not match its results.")


def _verify_performance_summary(summary: dict[str, object]) -> None:
    concurrency = summary.get("concurrency_sanity")
    routes = summary.get("routes")
    if not isinstance(concurrency, dict) or not isinstance(routes, dict):
        raise RuntimeError("Performance summary is missing route or concurrency results.")
    expected_routes = {"rag": 24, "chat": 12, "out_of_scope": 12}
    route_counts = {
        route: details.get("requests") if isinstance(details, dict) else None
        for route, details in routes.items()
    }
    if (
        summary.get("measured_requests") != 48
        or summary.get("completion_rate") != 1.0
        or summary.get("first_request_success_rate") != 1.0
        or route_counts != expected_routes
        or concurrency.get("requests") != 6
        or concurrency.get("completion_rate") != 1.0
    ):
        raise RuntimeError("Performance summary does not meet the release contract.")


def _verify_startup_result(result: dict[str, object]) -> None:
    startup_ready_ms = result.get("startup_ready_ms")
    if (
        result.get("release_version") != f"v{FROZEN_BASELINE_VERSION}"
        or result.get("probe_path") != "/health/ready"
        or not isinstance(startup_ready_ms, (int, float))
        or isinstance(startup_ready_ms, bool)
        or startup_ready_ms <= 0
    ):
        raise RuntimeError("Cold-start result does not meet the release contract.")


def _run_checked(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            + completed.stdout
            + completed.stderr
        )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _artifact_record(relative_path: str) -> dict[str, object]:
    path = PROJECT_ROOT / relative_path
    if not path.is_file():
        raise RuntimeError(f"Missing private release artifact: {relative_path}")
    record_count = sum(
        bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines()
    )
    expected_count = EXPECTED_PRIVATE_EVAL_RECORDS.get(
        relative_path, EXPECTED_RELEASE_EVIDENCE_RECORDS.get(relative_path)
    )
    if expected_count is not None and record_count != expected_count:
        raise RuntimeError(
            f"Private release artifact {relative_path} has {record_count} records; "
            f"expected {expected_count}."
        )
    return {
        "path": relative_path,
        "sha256": _sha256(path),
        "records": record_count,
    }


def _json_artifact_record(relative_path: str) -> dict[str, object]:
    path = PROJECT_ROOT / relative_path
    if not path.is_file():
        raise RuntimeError(f"Missing private release artifact: {relative_path}")
    return {"path": relative_path, "sha256": _sha256(path)}


def _load_jsonl_objects(relative_path: str) -> list[dict[str, object]]:
    path = PROJECT_ROOT / relative_path
    if not path.is_file():
        raise RuntimeError(f"Missing private release artifact: {relative_path}")
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise RuntimeError(f"Release evidence {relative_path}:{line_number} is not an object.")
        records.append(record)
    return records


def _load_json_object(relative_path: str) -> dict[str, object]:
    path = PROJECT_ROOT / relative_path
    if not path.is_file():
        raise RuntimeError(f"Missing private release artifact: {relative_path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Release evidence {relative_path} is not an object.")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()

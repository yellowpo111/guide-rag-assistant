import importlib.util
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_runner_module():
    script_path = PROJECT_ROOT / "scripts" / "run_dense_rerank_eval.py"
    spec = importlib.util.spec_from_file_location("run_dense_rerank_eval", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_accepts_supported_instruction_profile() -> None:
    runner = load_runner_module()

    arguments = runner.parse_arguments(["--instruction-profile", "fiscal_operation"])

    assert arguments.instruction_profile == "fiscal_operation"


def test_runner_defaults_to_top_twenty_dense_candidates() -> None:
    runner = load_runner_module()

    arguments = runner.parse_arguments([])

    assert arguments.candidate_k == 20


def test_runner_accepts_explicit_dense_candidate_count() -> None:
    runner = load_runner_module()

    arguments = runner.parse_arguments(["--candidate-k", "10"])

    assert arguments.candidate_k == 10


def test_runner_rejects_output_k_above_candidate_count() -> None:
    runner = load_runner_module()

    with pytest.raises(ValueError, match="cannot exceed"):
        runner.validate_candidate_k(output_k=5, candidate_k=4)


def test_runner_rejects_unknown_instruction_profile() -> None:
    runner = load_runner_module()

    with pytest.raises(SystemExit):
        runner.parse_arguments(["--instruction-profile", "unknown"])


def test_runner_accepts_conservative_query_rewrite_profile() -> None:
    runner = load_runner_module()

    arguments = runner.parse_arguments(
        ["--query-rewrite-profile", "conservative_deepseek"]
    )

    assert arguments.query_rewrite_profile == "conservative_deepseek"


def test_runner_accepts_guarded_query_rewrite_profile() -> None:
    runner = load_runner_module()

    arguments = runner.parse_arguments(
        ["--query-rewrite-profile", "conservative_deepseek_guarded"]
    )

    assert arguments.query_rewrite_profile == "conservative_deepseek_guarded"


def test_runner_rejects_query_rewrite_with_non_default_instruction() -> None:
    runner = load_runner_module()

    with pytest.raises(ValueError, match="instruction-profile default"):
        runner.validate_experiment_profiles(
            "fiscal_operation", "conservative_deepseek", None
        )


def test_runner_requires_frozen_rewrites_for_guarded_profile() -> None:
    runner = load_runner_module()

    with pytest.raises(ValueError, match="rewrite-source-details"):
        runner.validate_experiment_profiles(
            "default", "conservative_deepseek_guarded", None
        )


def test_runner_accepts_guarded_profile_with_frozen_rewrites() -> None:
    runner = load_runner_module()

    runner.validate_experiment_profiles(
        "default",
        "conservative_deepseek_guarded",
        Path("data_private/evals/results/frozen_rewrites.jsonl"),
    )


def test_runner_defaults_to_page_content_rerank_documents() -> None:
    runner = load_runner_module()

    arguments = runner.parse_arguments([])

    assert arguments.rerank_document_profile == "page_content"


def test_runner_accepts_metadata_context_rerank_documents() -> None:
    runner = load_runner_module()

    arguments = runner.parse_arguments(
        ["--rerank-document-profile", "metadata_context"]
    )

    assert arguments.rerank_document_profile == "metadata_context"


def test_runner_rejects_unknown_rerank_document_profile() -> None:
    runner = load_runner_module()

    with pytest.raises(SystemExit):
        runner.parse_arguments(["--rerank-document-profile", "unknown"])

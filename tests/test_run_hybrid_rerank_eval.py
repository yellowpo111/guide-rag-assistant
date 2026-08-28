import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_runner_module():
    script_path = PROJECT_ROOT / "scripts" / "run_hybrid_rerank_eval.py"
    spec = importlib.util.spec_from_file_location("run_hybrid_rerank_eval", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_uses_fixed_current_candidate_pool_defaults() -> None:
    runner = load_runner_module()

    arguments = runner.parse_arguments(
        [
            "--details-file",
            "details.jsonl",
            "--rewrite-source-details",
            "rewrites.jsonl",
        ]
    )

    assert arguments.dense_candidate_k == 10
    assert arguments.bm25_candidate_k == 10
    assert arguments.rerank_candidate_k == 10
    assert arguments.rrf_k == 60


def test_runner_rejects_output_k_above_fused_candidate_pool() -> None:
    runner = load_runner_module()
    arguments = runner.parse_arguments(
        [
            "--details-file",
            "details.jsonl",
            "--rewrite-source-details",
            "rewrites.jsonl",
            "--k",
            "5",
            "--rerank-candidate-k",
            "4",
        ]
    )

    with pytest.raises(ValueError, match="rerank-candidate-k"):
        runner.validate_arguments(arguments)

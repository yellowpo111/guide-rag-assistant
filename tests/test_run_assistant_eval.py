from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_assistant_eval import (  # noqa: E402
    local_service_address,
    parse_arguments,
    select_cases,
)
from fiscal_rag.assistant_evaluation import AssistantEvalCase  # noqa: E402


def test_parse_arguments_can_start_temporary_local_service() -> None:
    arguments = parse_arguments(["--start-local-service"])

    assert arguments.start_local_service is True


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("http://127.0.0.1:8000", ("127.0.0.1", 8000)),
        ("http://localhost:9000/", ("127.0.0.1", 9000)),
    ],
)
def test_local_service_address_accepts_only_local_http(
    base_url: str, expected: tuple[str, int]
) -> None:
    assert local_service_address(base_url) == expected


@pytest.mark.parametrize(
    "base_url",
    [
        "https://127.0.0.1:8000",
        "http://0.0.0.0:8000",
        "http://127.0.0.1:8000/v1",
        "http://example.com:8000",
    ],
)
def test_local_service_address_rejects_nonlocal_or_path_urls(base_url: str) -> None:
    with pytest.raises(ValueError, match="requires"):
        local_service_address(base_url)


def test_select_cases_preserves_dataset_order_for_retry_subset() -> None:
    cases = [
        AssistantEvalCase("case-1", "chat", "one", "chat", None),
        AssistantEvalCase("case-2", "chat", "two", "chat", None),
        AssistantEvalCase("case-3", "chat", "three", "chat", None),
    ]

    selected = select_cases(cases, ["case-3", "case-1"])

    assert [case.case_id for case in selected] == ["case-1", "case-3"]

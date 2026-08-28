import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts import export_usage_review, summarize_usage  # noqa: E402
from fiscal_rag.usage import SQLiteUsageRepository  # noqa: E402


def make_failed_request(database: Path, request_id: str = "failed-request") -> None:
    repository = SQLiteUsageRepository(database)
    repository.initialize()
    repository.create_request(
        request_id,
        endpoint="assistant_stream",
        question="SECRET QUESTION",
        service_version="1.3.0",
        profile_id="profile",
    )
    repository.finalize_request(
        request_id,
        execution_status="failed",
        route="rag",
        answer="SECRET ANSWER",
        error_code="model_request_failed",
        error_type="RuntimeError",
        failure_stage="generation",
        total_duration_ms=7000,
    )


def test_summarize_usage_cli_creates_private_report_pair_without_raw_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "usage.sqlite3"
    make_failed_request(database)
    json_path = tmp_path / "reports" / "report.json"
    markdown_path = tmp_path / "reports" / "report.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_usage.py",
            "--database",
            str(database),
            "--output-file",
            str(json_path),
            "--markdown-file",
            str(markdown_path),
        ],
    )

    summarize_usage.main()

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    console = capsys.readouterr().out
    assert summary["overview"]["total_requests"] == 1
    assert summary["review_funnel"]["actionable_requests"] == 1
    assert "SECRET QUESTION" not in json_path.read_text(encoding="utf-8")
    assert "SECRET QUESTION" not in markdown
    assert "SECRET QUESTION" not in console
    assert "Actionable requests: 1" in console


def test_report_pair_removes_partial_outputs_when_rendering_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"

    def fail_render(_summary):
        raise RuntimeError("simulated render failure")

    monkeypatch.setattr(summarize_usage, "render_usage_summary_markdown", fail_render)
    with pytest.raises(RuntimeError, match="simulated"):
        summarize_usage.write_report_pair(json_path, markdown_path, {})

    assert not json_path.exists()
    assert not markdown_path.exists()


def test_report_pair_refuses_to_overwrite_either_output(tmp_path: Path) -> None:
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    markdown_path.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        summarize_usage.write_report_pair(json_path, markdown_path, {})

    assert not json_path.exists()
    assert markdown_path.read_text(encoding="utf-8") == "existing"


def test_export_review_cli_selects_ids_and_does_not_create_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "usage.sqlite3"
    make_failed_request(database)
    selected_path = tmp_path / "selected.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_usage_review.py",
            "--database",
            str(database),
            "--request-id",
            "failed-request",
            "--output-file",
            str(selected_path),
        ],
    )

    export_usage_review.main()

    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    assert selected["request_id"] == "failed-request"
    assert selected["contains_raw_content"] is True

    rejected_path = tmp_path / "rejected.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_usage_review.py",
            "--database",
            str(database),
            "--request-id",
            "unknown",
            "--output-file",
            str(rejected_path),
        ],
    )
    with pytest.raises(ValueError, match="unknown, already confirmed, or ineligible"):
        export_usage_review.main()

    assert not rejected_path.exists()

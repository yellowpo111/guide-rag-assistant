import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts import query_usage  # noqa: E402
from scripts.run_text_to_sql_eval import write_eval_artifact_pair  # noqa: E402
from fiscal_rag.text_to_sql import GeneratedSQL  # noqa: E402
from fiscal_rag.text_to_sql_evaluation import build_synthetic_usage_fixture  # noqa: E402


class FixedGenerator:
    model_id = "fixture-model"

    def generate(self, _question: str) -> GeneratedSQL:
        return GeneratedSQL(
            "SELECT request_id, route FROM usage_requests "
            "ORDER BY request_id LIMIT 1"
        )


def test_query_cli_creates_private_pair_and_redacts_console(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = build_synthetic_usage_fixture(tmp_path / "fixture.sqlite3")
    json_path = tmp_path / "result.json"
    markdown_path = tmp_path / "result.md"
    monkeypatch.setattr(query_usage, "DeepSeekSQLGenerator", FixedGenerator)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "query_usage.py",
            "--question",
            "PRIVATE OPERATOR QUESTION",
            "--database",
            str(database),
            "--output-file",
            str(json_path),
            "--markdown-file",
            str(markdown_path),
        ],
    )

    query_usage.main()

    record = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    console = capsys.readouterr().out
    assert record["status"] == "completed"
    assert record["question"] == "PRIVATE OPERATOR QUESTION"
    assert "PRIVATE OPERATOR QUESTION" in markdown
    assert "PRIVATE OPERATOR QUESTION" not in console
    assert "SELECT request_id" not in console
    assert "p-" not in console


def test_query_pair_is_atomic_on_render_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    json_path = tmp_path / "result.json"
    markdown_path = tmp_path / "result.md"
    monkeypatch.setattr(
        query_usage,
        "render_text_to_sql_markdown",
        lambda _record: (_ for _ in ()).throw(RuntimeError("render failed")),
    )

    with pytest.raises(RuntimeError, match="render failed"):
        query_usage.write_query_artifact_pair(json_path, markdown_path, {})

    assert not json_path.exists()
    assert not markdown_path.exists()


def test_query_pair_refuses_to_overwrite_either_path(tmp_path: Path) -> None:
    json_path = tmp_path / "result.json"
    markdown_path = tmp_path / "result.md"
    json_path.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        query_usage.write_query_artifact_pair(json_path, markdown_path, {})

    assert json_path.read_text(encoding="utf-8") == "existing"
    assert not markdown_path.exists()


def test_eval_artifact_pair_is_exclusive_and_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    details = tmp_path / "run.jsonl"
    summary = tmp_path / "run.summary.json"
    write_eval_artifact_pair(details, summary, [{"case_id": "one"}], {"total": 1})
    assert details.is_file() and summary.is_file()
    with pytest.raises(FileExistsError):
        write_eval_artifact_pair(details, summary, [], {})

    second_details = tmp_path / "second.jsonl"
    second_summary = tmp_path / "second.summary.json"
    original_open = Path.open

    def fail_summary(path: Path, *args, **kwargs):
        if path == second_summary:
            raise OSError("synthetic write failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_summary)
    with pytest.raises(OSError, match="synthetic write failure"):
        write_eval_artifact_pair(second_details, second_summary, [], {})
    assert not second_details.exists()
    assert not second_summary.exists()

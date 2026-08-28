import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fiscal_rag.text_to_sql import (  # noqa: E402
    DeepSeekSQLGenerator,
    GeneratedSQL,
    ModelOutputError,
    ModelTransportError,
    ReadOnlyUsageQueryExecutor,
    TextToSQLError,
    build_text_to_sql_prompt,
    parse_generated_sql,
    render_text_to_sql_markdown,
    run_usage_text_to_sql,
    validate_generated_sql,
)
from fiscal_rag.text_to_sql_evaluation import (  # noqa: E402
    build_synthetic_usage_fixture,
)


class FakeChatModel:
    def __init__(self, response: object = '{"sql":"SELECT COUNT(*) FROM usage_requests"}') -> None:
        self.response = response
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        if isinstance(self.response, Exception):
            raise self.response
        return SimpleNamespace(content=self.response)


class FixedGenerator:
    model_id = "fixture-model"

    def __init__(self, generated: GeneratedSQL) -> None:
        self.generated = generated

    def generate(self, _question: str) -> GeneratedSQL:
        return self.generated


def fixture_database(tmp_path: Path) -> Path:
    return build_synthetic_usage_fixture(tmp_path / "synthetic.sqlite3")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prompt_exposes_only_safe_schema_and_no_real_rows() -> None:
    prompt = build_text_to_sql_prompt("不同 route 有多少请求？")

    for view in (
        "usage_requests",
        "usage_timings",
        "usage_feedback",
        "usage_reviews",
        "usage_source_stats",
    ):
        assert view in prompt
    for prohibited in (
        "assistant_requests(",
        "request_sources(",
        "question TEXT",
        "answer TEXT",
        "reason TEXT",
        "source TEXT",
        "SYNTHETIC QUESTION",
    ):
        assert prohibited not in prompt
    assert "completed 只表示执行完成" in prompt
    assert "不能把阶段耗时相加" in prompt


def test_strict_model_output_accepts_sql_or_stable_refusal() -> None:
    assert parse_generated_sql('{"sql":"SELECT COUNT(*) FROM usage_requests"}') == GeneratedSQL(
        sql="SELECT COUNT(*) FROM usage_requests"
    )
    assert parse_generated_sql(
        '{"sql":null,"refusal_code":"unsupported_query"}'
    ) == GeneratedSQL(sql=None, refusal_code="unsupported_query")

    for invalid in (
        "SELECT 1",
        "```json\n{\"sql\":\"SELECT 1\"}\n```",
        '{"sql":""}',
        '{"sql":null}',
        '{"sql":null,"refusal_code":"other"}',
        '{"sql":"SELECT 1","reason":"extra"}',
    ):
        with pytest.raises(ModelOutputError):
            parse_generated_sql(invalid)


def test_deepseek_generator_uses_isolated_prompt_and_sanitizes_failures() -> None:
    chat_model = FakeChatModel()
    generator = DeepSeekSQLGenerator(chat_model, model_id="fixture-model")

    generated = generator.generate("统计 route")

    assert generated.sql == "SELECT COUNT(*) FROM usage_requests"
    assert "统计 route" in chat_model.prompts[0]
    assert generator.model_id == "fixture-model"

    with pytest.raises(ModelTransportError) as captured:
        DeepSeekSQLGenerator(
            FakeChatModel(RuntimeError("PRIVATE PROVIDER DETAIL")),
            model_id="fixture-model",
        ).generate("统计 route")
    assert str(captured.value) == "model_transport_error"


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM assistant_requests",
        "PRAGMA user_version",
        "ATTACH DATABASE 'other.sqlite3' AS other",
        "WITH RECURSIVE counter(x) AS (SELECT 1) SELECT x FROM counter",
        "SELECT 1; SELECT 2",
        "SELECT 1 -- comment",
        "SELECT /* comment */ 1",
    ],
)
def test_sql_shape_rejects_non_select_recursive_comments_and_multiple_statements(
    sql: str,
) -> None:
    with pytest.raises(TextToSQLError, match="sql_rejected"):
        validate_generated_sql(sql)


def test_read_only_executor_filters_nonproduction_and_never_changes_database(
    tmp_path: Path,
) -> None:
    database = fixture_database(tmp_path)
    before = file_hash(database)
    executor = ReadOnlyUsageQueryExecutor(database)

    result = executor.execute(
        "SELECT execution_status, COUNT(*) FROM usage_requests "
        "GROUP BY execution_status ORDER BY execution_status"
    )

    assert sum(int(row[1]) for row in result.rows) == 11
    assert all("eval" not in str(value) for row in result.rows for value in row)
    assert file_hash(database) == before


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT question FROM assistant_requests",
        "SELECT answer FROM assistant_requests",
        "SELECT source FROM request_sources",
        "SELECT reason FROM request_reviews",
        "SELECT * FROM sqlite_master",
        "SELECT random() FROM usage_requests",
        "SELECT load_extension('private')",
        "DROP TABLE assistant_requests",
    ],
)
def test_authorizer_blocks_raw_tables_sensitive_fields_functions_and_writes(
    tmp_path: Path,
    sql: str,
) -> None:
    database = fixture_database(tmp_path)
    before = file_hash(database)

    with pytest.raises(TextToSQLError, match="sql_rejected"):
        ReadOnlyUsageQueryExecutor(database).execute(sql)

    assert file_hash(database) == before


def test_missing_sensitive_columns_in_safe_views_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(TextToSQLError, match="sql_rejected"):
        ReadOnlyUsageQueryExecutor(fixture_database(tmp_path)).execute(
            "SELECT question FROM usage_requests"
        )


def test_executor_limits_rows_and_marks_truncation(tmp_path: Path) -> None:
    result = ReadOnlyUsageQueryExecutor(
        fixture_database(tmp_path), max_rows=2
    ).execute(
        "SELECT request_id FROM usage_requests ORDER BY request_id"
    )

    assert len(result.rows) == 2
    assert result.truncated is True


def test_executor_interrupts_expensive_safe_query(tmp_path: Path) -> None:
    executor = ReadOnlyUsageQueryExecutor(
        fixture_database(tmp_path), timeout_seconds=0.001
    )
    sql = """
    WITH ids AS (SELECT request_id FROM usage_requests)
    SELECT COUNT(*) FROM ids AS a, ids AS b, ids AS c, ids AS d,
         ids AS e, ids AS f, ids AS g, ids AS h
    """

    with pytest.raises(TextToSQLError, match="sql_timeout"):
        executor.execute(sql)


def test_run_record_and_markdown_keep_stable_result_contract(tmp_path: Path) -> None:
    record = run_usage_text_to_sql(
        "不同 route 有多少请求？",
        generator=FixedGenerator(
            GeneratedSQL(
                "SELECT route, COUNT(*) FROM usage_requests "
                "GROUP BY route ORDER BY route"
            )
        ),
        executor=ReadOnlyUsageQueryExecutor(fixture_database(tmp_path)),
    )
    markdown = render_text_to_sql_markdown(record)

    assert record["status"] == "completed"
    assert record["data_scope"] == "production_safe_views"
    assert record["contains_raw_request_content"] is False
    assert record["contains_request_identifiers"] is True
    assert record["expires_at_utc"]
    assert record["row_count"] == 4
    assert "SYNTHETIC QUESTION" not in markdown
    assert "SELECT route" in markdown


def test_markdown_treats_model_and_result_text_as_inert_content() -> None:
    markdown = render_text_to_sql_markdown(
        {
            "question": "![remote](https://invalid.example/image)",
            "sql": "SELECT '```\\n<script>alert(1)</script>' AS value",
            "columns": ["<script>"],
            "rows": [["<img src=x onerror=alert(1)>|value"]],
        }
    )

    assert "![remote]" not in markdown
    assert "<script>" not in markdown
    assert "<img" not in markdown
    assert "```sql" not in markdown
    assert "&lt;script&gt;" in markdown


def test_run_record_preserves_refusal_without_opening_database(tmp_path: Path) -> None:
    record = run_usage_text_to_sql(
        "列出原始问题和答案",
        generator=FixedGenerator(
            GeneratedSQL(sql=None, refusal_code="unsupported_query")
        ),
        executor=ReadOnlyUsageQueryExecutor(tmp_path / "missing.sqlite3"),
    )

    assert record["status"] == "refused"
    assert record["error_code"] == "unsupported_query"
    assert not (tmp_path / "missing.sqlite3").exists()

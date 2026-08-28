"""Local-only natural-language queries over privacy-filtered usage views."""

from __future__ import annotations

import html
import json
import os
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import monotonic, perf_counter
from typing import Any, Protocol

from fiscal_rag.pipeline import (
    DEFAULT_DEEPSEEK_MODEL,
    ChatModel,
    create_deepseek_chat_model,
)


TEXT_TO_SQL_SCHEMA_VERSION = "usage-text-to-sql-v1"
TEXT_TO_SQL_PROMPT_VERSION = "usage-text-to-sql-prompt-v1"
TEXT_TO_SQL_DATA_SCOPE = "production_safe_views"
DEFAULT_MAX_ROWS = 50
HARD_MAX_ROWS = 100
DEFAULT_QUERY_TIMEOUT_SECONDS = 2.0
HARD_QUERY_TIMEOUT_SECONDS = 5.0
MAX_SQL_CHARACTERS = 10_000
MAX_RESULT_BYTES = 65_536
MAX_CELL_CHARACTERS = 512
SUPPORTED_REFUSAL_CODE = "unsupported_query"
SAFE_VIEW_NAMES = frozenset(
    {
        "usage_requests",
        "usage_timings",
        "usage_feedback",
        "usage_reviews",
        "usage_source_stats",
    }
)
SAFE_SQL_FUNCTIONS = frozenset(
    {
        "abs",
        "avg",
        "coalesce",
        "count",
        "date",
        "datetime",
        "max",
        "min",
        "nullif",
        "round",
        "strftime",
        "sum",
    }
)
SQL_START_PATTERN = re.compile(r"^(SELECT|WITH)\b", re.IGNORECASE)
RECURSIVE_CTE_PATTERN = re.compile(r"^WITH\s+RECURSIVE\b", re.IGNORECASE)
SQL_COMMENT_PATTERN = re.compile(r"--|/\*|\*/")


TEXT_TO_SQL_PROMPT = """你是公司内网 Usage Analytics 的 SQLite 查询生成器。

你的唯一任务是把维护人员的问题转换为一条只读 SQLite 查询。不要回答问题，不要解释推理。
只能使用下面列出的临时安全视图；它们已经固定只包含 production 流量。

usage_requests(
  request_id TEXT, endpoint TEXT, started_at_utc TEXT, finished_at_utc TEXT,
  service_version TEXT, profile_id TEXT, route TEXT,
  execution_status TEXT, error_code TEXT, error_type TEXT,
  failure_stage TEXT, total_duration_ms REAL,
  has_retrieval_trace INTEGER, has_missing_constraints INTEGER
)

usage_timings(
  request_id TEXT, endpoint TEXT, route TEXT, execution_status TEXT,
  stage TEXT, duration_ms REAL
)

usage_feedback(
  request_id TEXT, route TEXT, rating TEXT,
  created_at_utc TEXT, updated_at_utc TEXT
)

usage_reviews(
  request_id TEXT, service_version TEXT, route TEXT, review_status TEXT,
  failure_type TEXT, severity TEXT, expected_route TEXT,
  answerability INTEGER, eval_candidate INTEGER, candidate_case_id TEXT,
  candidate_category TEXT, retrieval_case_id TEXT, reviewed_at_utc TEXT
)

usage_source_stats(request_id TEXT, route TEXT, source_count INTEGER)

字段语义：
- endpoint 只有 assistant_stream 或 ask。
- route 为 rag、chat、out_of_scope 或 NULL。统计 Assistant route 时必须限定 endpoint='assistant_stream'。
- execution_status 为 started、completed、failed、aborted、interrupted；completed 只表示执行完成，不表示回答正确。
- failed、aborted、interrupted 是不同终态。用户说“失败”时只使用 failed，除非问题明确要求其他终态。
- total_duration_ms 和 duration_ms 缺失时为 NULL，不能当作 0。
- usage_timings 的阶段可能嵌套，不能把阶段耗时相加为总耗时。
- 未出现在 usage_feedback 中表示没有反馈，不是中立或负面反馈。
- answerability 和 eval_candidate 使用 0/1；人工 failure 只来自 usage_reviews。
- UTC 时间以带时区 ISO 8601 文本保存，可以直接排序。用户说“最近”但没有给时间范围时，按 started_at_utc 降序并限制结果数量。
- p50/p95 不属于本实验的 SQL 承诺；只生成 count、avg、sum、min、max、列表和 JOIN 查询。

安全和格式规则：
- 只生成单条 SELECT 或非递归 WITH 查询。
- 不得使用原始表、系统表、PRAGMA、ATTACH、DDL、DML、注释或多语句。
- 不得查询原始问题、回答、retrieval/rewrite query、source 名、review reason、expected answer、evidence 或 topic。
- 列表查询最多 LIMIT 50，并使用确定性 ORDER BY。
- 如果问题要求未暴露的敏感字段、写操作、任意 SQL 执行或无法由这些视图回答，必须拒绝。

严格输出一个 JSON 对象，不要使用 Markdown code fence，也不要增加其他字段：
- 可回答：{{"sql":"SELECT ..."}}
- 必须拒绝：{{"sql":null,"refusal_code":"unsupported_query"}}

维护人员问题：
{question}
"""


SAFE_VIEW_SQL = """
CREATE TEMP VIEW usage_requests AS
SELECT
    request_id,
    endpoint,
    started_at_utc,
    finished_at_utc,
    service_version,
    profile_id,
    route,
    execution_status,
    error_code,
    error_type,
    failure_stage,
    total_duration_ms,
    CASE WHEN retrieval_query IS NULL THEN 0 ELSE 1 END AS has_retrieval_trace,
    CASE WHEN missing_constraints_json = '[]' THEN 0 ELSE 1 END AS has_missing_constraints
FROM assistant_requests
WHERE traffic_kind = 'production';

CREATE TEMP VIEW usage_timings AS
SELECT
    r.request_id,
    r.endpoint,
    r.route,
    r.execution_status,
    t.stage,
    t.duration_ms
FROM assistant_requests AS r
JOIN request_timings AS t ON t.request_id = r.request_id
WHERE r.traffic_kind = 'production';

CREATE TEMP VIEW usage_feedback AS
SELECT
    r.request_id,
    r.route,
    f.rating,
    f.created_at_utc,
    f.updated_at_utc
FROM assistant_requests AS r
JOIN request_feedback AS f ON f.request_id = r.request_id
WHERE r.traffic_kind = 'production';

CREATE TEMP VIEW usage_reviews AS
SELECT
    r.request_id,
    r.service_version,
    r.route,
    v.review_status,
    v.failure_type,
    v.severity,
    v.expected_route,
    v.answerability,
    v.eval_candidate,
    v.candidate_case_id,
    v.candidate_category,
    v.retrieval_case_id,
    v.reviewed_at_utc
FROM assistant_requests AS r
JOIN request_reviews AS v ON v.request_id = r.request_id
WHERE r.traffic_kind = 'production';

CREATE TEMP VIEW usage_source_stats AS
SELECT
    r.request_id,
    r.route,
    COUNT(s.rank) AS source_count
FROM assistant_requests AS r
LEFT JOIN request_sources AS s ON s.request_id = r.request_id
WHERE r.traffic_kind = 'production'
GROUP BY r.request_id, r.route;
"""


class TextToSQLError(RuntimeError):
    """Stable, content-free failure raised by the prototype."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class ModelTransportError(TextToSQLError):
    def __init__(self) -> None:
        super().__init__("model_transport_error")


class ModelOutputError(TextToSQLError):
    def __init__(self) -> None:
        super().__init__("model_output_invalid")


@dataclass(frozen=True)
class GeneratedSQL:
    sql: str | None
    refusal_code: str | None = None


@dataclass(frozen=True)
class SQLQueryResult:
    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]
    truncated: bool
    duration_ms: float


class SQLGenerator(Protocol):
    model_id: str

    def generate(self, question: str) -> GeneratedSQL: ...


class DeepSeekSQLGenerator:
    """Generate one strict SQL payload without exposing database rows."""

    def __init__(
        self,
        chat_model: ChatModel | None = None,
        *,
        model_id: str | None = None,
    ) -> None:
        self._chat_model = chat_model or create_deepseek_chat_model()
        self.model_id = model_id or os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)

    def generate(self, question: str) -> GeneratedSQL:
        prompt = build_text_to_sql_prompt(question)
        try:
            response = self._chat_model.invoke(prompt)
        except Exception as error:
            raise ModelTransportError() from error
        content: Any = getattr(response, "content", response)
        if not isinstance(content, str):
            raise ModelOutputError()
        return parse_generated_sql(content)


class ReadOnlyUsageQueryExecutor:
    """Execute generated SQL only through production-safe temporary views."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        max_rows: int = DEFAULT_MAX_ROWS,
        timeout_seconds: float = DEFAULT_QUERY_TIMEOUT_SECONDS,
    ) -> None:
        self.database_path = Path(database_path)
        if not 1 <= max_rows <= HARD_MAX_ROWS:
            raise ValueError(f"max_rows must be between 1 and {HARD_MAX_ROWS}")
        if not 0 < timeout_seconds <= HARD_QUERY_TIMEOUT_SECONDS:
            raise ValueError(
                f"timeout_seconds must be greater than 0 and at most "
                f"{HARD_QUERY_TIMEOUT_SECONDS}"
            )
        self.max_rows = max_rows
        self.timeout_seconds = timeout_seconds

    def execute(self, sql: str) -> SQLQueryResult:
        validated_sql = validate_generated_sql(sql)
        if not self.database_path.is_file():
            raise TextToSQLError("usage_database_not_found")
        started = perf_counter()
        connection = self._open_connection()
        try:
            deadline = monotonic() + self.timeout_seconds
            connection.set_progress_handler(
                lambda: int(monotonic() >= deadline),
                1_000,
            )
            connection.set_authorizer(_usage_query_authorizer)
            try:
                cursor = connection.execute(validated_sql)
                columns = tuple(
                    str(description[0]) for description in (cursor.description or ())
                )
                raw_rows = cursor.fetchmany(self.max_rows + 1)
            except sqlite3.DatabaseError as error:
                message = str(error).lower()
                code = "sql_timeout" if "interrupted" in message else "sql_rejected"
                if code == "sql_rejected" and not (
                    "not authorized" in message or "prohibited" in message
                    or "no such column" in message
                    or "no such table" in message
                ):
                    code = "sql_execution_error"
                raise TextToSQLError(code) from error
            rows, size_truncated = _bounded_rows(raw_rows[: self.max_rows])
            return SQLQueryResult(
                columns=columns,
                rows=tuple(rows),
                truncated=len(raw_rows) > self.max_rows or size_truncated,
                duration_ms=round((perf_counter() - started) * 1000, 3),
            )
        finally:
            connection.close()

    def _open_connection(self) -> sqlite3.Connection:
        uri = self.database_path.resolve().as_uri() + "?mode=ro"
        try:
            connection = sqlite3.connect(
                uri,
                uri=True,
                timeout=self.timeout_seconds,
            )
        except sqlite3.Error as error:
            raise TextToSQLError("usage_database_open_failed") from error
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA trusted_schema = OFF")
            connection.execute("PRAGMA temp_store = MEMORY")
            connection.executescript(SAFE_VIEW_SQL)
            connection.execute("PRAGMA query_only = ON")
            _apply_sqlite_limits(connection)
        except Exception:
            connection.close()
            raise
        return connection


def build_text_to_sql_prompt(question: str) -> str:
    normalized = question.strip()
    if not normalized or len(normalized) > 1_000:
        raise ValueError("question must contain between 1 and 1000 characters")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in normalized):
        raise ValueError("question contains invalid Unicode characters")
    return TEXT_TO_SQL_PROMPT.format(question=normalized)


def parse_generated_sql(content: str) -> GeneratedSQL:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise ModelOutputError() from error
    if not isinstance(payload, dict):
        raise ModelOutputError()
    if set(payload) == {"sql"}:
        sql = payload.get("sql")
        if isinstance(sql, str) and sql.strip():
            return GeneratedSQL(sql=sql.strip())
        raise ModelOutputError()
    if set(payload) == {"sql", "refusal_code"}:
        if payload.get("sql") is None and payload.get("refusal_code") == SUPPORTED_REFUSAL_CODE:
            return GeneratedSQL(sql=None, refusal_code=SUPPORTED_REFUSAL_CODE)
    raise ModelOutputError()


def validate_generated_sql(sql: str) -> str:
    normalized = sql.strip()
    if not normalized or len(normalized) > MAX_SQL_CHARACTERS:
        raise TextToSQLError("sql_rejected")
    if SQL_COMMENT_PATTERN.search(normalized):
        raise TextToSQLError("sql_rejected")
    if not SQL_START_PATTERN.match(normalized) or RECURSIVE_CTE_PATTERN.match(normalized):
        raise TextToSQLError("sql_rejected")
    if normalized.endswith(";"):
        normalized = normalized[:-1].rstrip()
    if ";" in normalized:
        raise TextToSQLError("sql_rejected")
    return normalized


def run_usage_text_to_sql(
    question: str,
    *,
    generator: SQLGenerator,
    executor: ReadOnlyUsageQueryExecutor,
    generated_at: datetime | None = None,
    retention_days: int = 90,
) -> dict[str, object]:
    active_generated_at = generated_at or datetime.now(UTC)
    if active_generated_at.tzinfo is None:
        raise ValueError("generated_at must be timezone-aware")
    if retention_days <= 0:
        raise ValueError("retention_days must be positive")
    generated_at_utc = active_generated_at.astimezone(UTC)
    base: dict[str, object] = {
        "schema_version": TEXT_TO_SQL_SCHEMA_VERSION,
        "prompt_version": TEXT_TO_SQL_PROMPT_VERSION,
        "generated_at_utc": generated_at_utc.isoformat(),
        "expires_at_utc": (generated_at_utc + timedelta(days=retention_days)).isoformat(),
        "data_scope": TEXT_TO_SQL_DATA_SCOPE,
        "contains_raw_request_content": False,
        "contains_request_identifiers": True,
        "model_id": generator.model_id,
        "question": question.strip(),
        "sql": None,
        "columns": [],
        "rows": [],
        "row_count": 0,
        "truncated": False,
        "generation_duration_ms": None,
        "execution_duration_ms": None,
        "status": "failed",
        "error_code": None,
    }
    generation_started = perf_counter()
    try:
        generated = generator.generate(question)
    except TextToSQLError as error:
        base["generation_duration_ms"] = round(
            (perf_counter() - generation_started) * 1000, 3
        )
        base["error_code"] = error.code
        return base
    base["generation_duration_ms"] = round(
        (perf_counter() - generation_started) * 1000, 3
    )
    if generated.refusal_code:
        base["status"] = "refused"
        base["error_code"] = generated.refusal_code
        return base
    base["sql"] = generated.sql
    try:
        result = executor.execute(generated.sql or "")
    except TextToSQLError as error:
        base["status"] = "rejected" if error.code == "sql_rejected" else "failed"
        base["error_code"] = error.code
        return base
    base.update(
        {
            "status": "completed",
            "columns": list(result.columns),
            "rows": [list(row) for row in result.rows],
            "row_count": len(result.rows),
            "truncated": result.truncated,
            "execution_duration_ms": result.duration_ms,
            "error_code": "result_truncated" if result.truncated else None,
        }
    )
    return base


def render_text_to_sql_markdown(record: Mapping[str, object]) -> str:
    lines = [
        "# Usage Text-to-SQL Result",
        "",
        f"- Generated at (UTC): `{record.get('generated_at_utc')}`",
        f"- Expires at (UTC): `{record.get('expires_at_utc')}`",
        f"- Status: `{record.get('status')}`",
        f"- Data scope: `{record.get('data_scope')}`",
        f"- Model: `{record.get('model_id')}`",
        f"- Contains raw request content: `false`",
        f"- Contains request identifiers: `true`",
        f"- Truncated: `{str(bool(record.get('truncated'))).lower()}`",
        "",
        "## Question",
        "",
        _escape_markdown(record.get("question")),
        "",
        "## Generated SQL",
        "",
    ]
    sql = record.get("sql")
    lines.extend([_indented_code(sql if isinstance(sql, str) else "-- no SQL"), ""])
    if record.get("error_code"):
        lines.extend(
            ["## Outcome", "", f"Error code: `{record.get('error_code')}`", ""]
        )
    lines.extend(["## Result", "", _markdown_result(record)])
    return "\n".join(lines).rstrip() + "\n"


def _usage_query_authorizer(
    action: int,
    arg1: str | None,
    arg2: str | None,
    _database: str | None,
    source: str | None,
) -> int:
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION:
        return (
            sqlite3.SQLITE_OK
            if (arg2 or "").lower() in SAFE_SQL_FUNCTIONS
            else sqlite3.SQLITE_DENY
        )
    if action == sqlite3.SQLITE_READ:
        if arg1 in SAFE_VIEW_NAMES or source in SAFE_VIEW_NAMES:
            return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def _apply_sqlite_limits(connection: sqlite3.Connection) -> None:
    connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, MAX_SQL_CHARACTERS)
    connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_RESULT_BYTES)
    connection.setlimit(sqlite3.SQLITE_LIMIT_COLUMN, 20)
    connection.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 50)
    connection.setlimit(sqlite3.SQLITE_LIMIT_COMPOUND_SELECT, 10)
    connection.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
    connection.setlimit(sqlite3.SQLITE_LIMIT_LIKE_PATTERN_LENGTH, 100)


def _bounded_rows(
    raw_rows: Sequence[sqlite3.Row],
) -> tuple[list[tuple[object, ...]], bool]:
    rows: list[tuple[object, ...]] = []
    used_bytes = 0
    for raw_row in raw_rows:
        row = tuple(_safe_cell(value) for value in raw_row)
        row_bytes = len(json.dumps(row, ensure_ascii=False).encode("utf-8"))
        if used_bytes + row_bytes > MAX_RESULT_BYTES:
            return rows, True
        used_bytes += row_bytes
        rows.append(row)
    return rows, False


def _safe_cell(value: object) -> object:
    if value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, str) and len(value) <= MAX_CELL_CHARACTERS:
        return value
    raise TextToSQLError("result_too_large")


def _markdown_result(record: Mapping[str, object]) -> str:
    columns = record.get("columns")
    rows = record.get("rows")
    if not isinstance(columns, list) or not columns:
        return "No result rows."
    if not isinstance(rows, list) or not rows:
        return "No result rows."
    header = "| " + " | ".join(_escape_markdown(column) for column in columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    body = []
    for row in rows:
        values = row if isinstance(row, list) else []
        body.append(
            "| " + " | ".join(_escape_markdown(value) for value in values) + " |"
        )
    return "\n".join([header, separator, *body])


def _escape_markdown(value: object) -> str:
    if value is None:
        return "-"
    text = html.escape(str(value), quote=False).replace("\r", " ").replace("\n", " ")
    for character in "\\`*_{}[]()#+-.!>|":
        text = text.replace(character, f"\\{character}")
    return text


def _indented_code(value: str) -> str:
    escaped = html.escape(value, quote=False)
    return "\n".join(f"    {line}" for line in escaped.splitlines())

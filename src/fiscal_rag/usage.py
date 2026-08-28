"""SQLite persistence for private assistant usage, feedback, and reviews."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator, Protocol


USAGE_SCHEMA_VERSION = 1
ACTIVE_PROFILE_ID = "rewrite-guard-dense20-rerank5"
EXECUTION_STATUSES = (
    "started",
    "completed",
    "failed",
    "aborted",
    "interrupted",
)
FEEDBACK_RATINGS = ("positive", "negative")
TRAFFIC_KINDS = ("production", "assistant_eval", "performance_eval")
FAILURE_TYPES = (
    "no_issue",
    "routing",
    "retrieval",
    "knowledge_coverage",
    "generation_support",
    "generation_correctness",
    "abstention",
    "boundary",
    "performance",
    "upstream_transport",
    "other",
)
REVIEW_SEVERITIES = ("none", "minor", "major")
REVIEW_STATUSES = ("pending", "user_confirmed")
EVAL_CATEGORIES = (
    "rag_answerable",
    "rag_unanswerable",
    "chat",
    "out_of_scope",
    "routing_boundary",
)


class UsagePersistenceError(RuntimeError):
    """Base error for the private usage store."""


class UsageRequestNotFound(UsagePersistenceError):
    """Raised when a request ID is unknown to the usage store."""


class UsageFeedbackNotAllowed(UsagePersistenceError):
    """Raised when feedback targets an ineligible request."""


class UsageReviewNotAllowed(UsagePersistenceError):
    """Raised when a review targets non-production traffic."""


@dataclass(frozen=True)
class UsageReview:
    request_id: str
    review_status: str
    failure_type: str
    severity: str
    expected_route: str | None
    answerability: bool | None
    reason: str
    eval_candidate: bool
    candidate_case_id: str | None = None
    candidate_category: str | None = None
    retrieval_case_id: str | None = None
    expected_answer: str | None = None
    evidence_json: str | None = None
    topic: str | None = None


class UsageRepository(Protocol):
    """Narrow persistence contract used by the HTTP and review layers."""

    def initialize(self) -> None: ...

    def mark_started_interrupted(self) -> int: ...

    def create_request(
        self,
        request_id: str,
        *,
        endpoint: str,
        question: str,
        service_version: str,
        profile_id: str,
        traffic_kind: str = "production",
        route: str | None = None,
    ) -> None: ...

    def finalize_request(
        self,
        request_id: str,
        *,
        execution_status: str,
        route: str | None,
        answer: str,
        trace: Mapping[str, object] | None = None,
        timings_ms: Mapping[str, object] | None = None,
        error_code: str | None = None,
        error_type: str | None = None,
        failure_stage: str | None = None,
        total_duration_ms: float | None = None,
    ) -> None: ...

    def set_feedback(self, request_id: str, rating: str) -> None: ...

    def delete_feedback(self, request_id: str) -> bool: ...

    def save_reviews(self, reviews: Sequence[UsageReview]) -> None: ...

    def fetch_usage_records(
        self,
        *,
        started_from: str | None = None,
        started_to: str | None = None,
    ) -> list[dict[str, object]]: ...

    def prune_expired(
        self, retention_days: int, *, now: datetime | None = None
    ) -> int: ...

    def backup(self, output_path: str | Path) -> Path: ...


class SQLiteUsageRepository:
    """Small, transaction-oriented SQLite implementation for one host."""

    def __init__(self, database_path: str | Path, *, busy_timeout_ms: int = 5000) -> None:
        if busy_timeout_ms <= 0:
            raise ValueError("busy_timeout_ms must be positive")
        self.database_path = Path(database_path)
        self._busy_timeout_ms = busy_timeout_ms

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > USAGE_SCHEMA_VERSION:
                raise UsagePersistenceError(
                    f"Usage database schema {version} is newer than supported "
                    f"schema {USAGE_SCHEMA_VERSION}."
                )
            if version == 0:
                connection.executescript(
                    "BEGIN IMMEDIATE;\n"
                    + _SCHEMA_V1
                    + f"\nPRAGMA user_version = {USAGE_SCHEMA_VERSION};\nCOMMIT;"
                )
            violations = connection.execute("PRAGMA foreign_key_check").fetchall()
            if violations:
                raise UsagePersistenceError("Usage database foreign key check failed.")

    def create_request(
        self,
        request_id: str,
        *,
        endpoint: str,
        question: str,
        service_version: str,
        profile_id: str,
        traffic_kind: str = "production",
        route: str | None = None,
    ) -> None:
        if endpoint not in {"assistant_stream", "ask"}:
            raise ValueError("endpoint must be assistant_stream or ask")
        if traffic_kind not in TRAFFIC_KINDS:
            raise ValueError("unsupported traffic_kind")
        if route not in {None, "rag", "chat", "out_of_scope"}:
            raise ValueError("unsupported route")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO assistant_requests (
                    request_id, endpoint, started_at_utc, service_version,
                    profile_id, traffic_kind, route, execution_status, question
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'started', ?)
                """,
                (
                    request_id,
                    endpoint,
                    utc_now_iso(),
                    service_version,
                    profile_id,
                    traffic_kind,
                    route,
                    question,
                ),
            )

    def finalize_request(
        self,
        request_id: str,
        *,
        execution_status: str,
        route: str | None,
        answer: str,
        trace: Mapping[str, object] | None = None,
        timings_ms: Mapping[str, object] | None = None,
        error_code: str | None = None,
        error_type: str | None = None,
        failure_stage: str | None = None,
        total_duration_ms: float | None = None,
    ) -> None:
        if execution_status not in set(EXECUTION_STATUSES) - {"started"}:
            raise ValueError("execution_status must be terminal")
        if route not in {None, "rag", "chat", "out_of_scope"}:
            raise ValueError("unsupported route")
        safe_trace = trace if isinstance(trace, Mapping) else {}
        required_constraints = _string_list(safe_trace.get("required_constraints"))
        missing_constraints = _string_list(safe_trace.get("missing_constraints"))
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE assistant_requests
                SET finished_at_utc = ?, route = ?, execution_status = ?, answer = ?,
                    retrieval_query = ?, rewrite_query = ?, query_rewrite_status = ?,
                    required_constraints_json = ?, missing_constraints_json = ?,
                    error_code = ?, error_type = ?, failure_stage = ?,
                    total_duration_ms = ?
                WHERE request_id = ?
                """,
                (
                    utc_now_iso(),
                    route,
                    execution_status,
                    answer,
                    _optional_string(safe_trace.get("retrieval_query")),
                    _optional_string(safe_trace.get("rewrite_query")),
                    _optional_string(safe_trace.get("query_rewrite_status")),
                    json.dumps(required_constraints, ensure_ascii=False),
                    json.dumps(missing_constraints, ensure_ascii=False),
                    error_code,
                    error_type,
                    failure_stage,
                    _optional_number(total_duration_ms),
                    request_id,
                ),
            )
            if cursor.rowcount != 1:
                raise UsageRequestNotFound(request_id)
            connection.execute(
                "DELETE FROM request_timings WHERE request_id = ?", (request_id,)
            )
            for stage, duration_ms in _numeric_items(timings_ms):
                connection.execute(
                    "INSERT INTO request_timings (request_id, stage, duration_ms) "
                    "VALUES (?, ?, ?)",
                    (request_id, stage, duration_ms),
                )
            connection.execute(
                "DELETE FROM request_sources WHERE request_id = ?", (request_id,)
            )
            for source in _source_records(safe_trace.get("sources")):
                connection.execute(
                    """
                    INSERT INTO request_sources (
                        request_id, rank, source, section, subsection,
                        dense_score, rerank_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (request_id, *source),
                )

    def mark_started_interrupted(self) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE assistant_requests
                SET execution_status = 'interrupted', finished_at_utc = ?,
                    error_code = 'service_restart', failure_stage = 'service'
                WHERE execution_status = 'started'
                """,
                (utc_now_iso(),),
            )
            return cursor.rowcount

    def set_feedback(self, request_id: str, rating: str) -> None:
        if rating not in FEEDBACK_RATINGS:
            raise ValueError("rating must be positive or negative")
        now = utc_now_iso()
        with self._connect() as connection:
            request_row = connection.execute(
                "SELECT endpoint, execution_status, traffic_kind "
                "FROM assistant_requests "
                "WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if request_row is None:
                raise UsageRequestNotFound(request_id)
            if (
                request_row[0] != "assistant_stream"
                or request_row[1] != "completed"
                or request_row[2] != "production"
            ):
                raise UsageFeedbackNotAllowed(request_id)
            connection.execute(
                """
                INSERT INTO request_feedback (
                    request_id, rating, created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(request_id) DO UPDATE SET
                    rating = excluded.rating,
                    updated_at_utc = excluded.updated_at_utc
                """,
                (request_id, rating, now, now),
            )

    def delete_feedback(self, request_id: str) -> bool:
        with self._connect() as connection:
            request_row = connection.execute(
                "SELECT endpoint, execution_status, traffic_kind "
                "FROM assistant_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if request_row is None:
                raise UsageRequestNotFound(request_id)
            if (
                request_row[0] != "assistant_stream"
                or request_row[1] != "completed"
                or request_row[2] != "production"
            ):
                raise UsageFeedbackNotAllowed(request_id)
            cursor = connection.execute(
                "DELETE FROM request_feedback WHERE request_id = ?", (request_id,)
            )
            return cursor.rowcount == 1

    def save_review(self, review: UsageReview) -> None:
        self.save_reviews([review])

    def save_reviews(self, reviews: Sequence[UsageReview]) -> None:
        for review in reviews:
            _validate_review(review)
        with self._connect() as connection:
            for review in reviews:
                self._save_review(connection, review)

    @staticmethod
    def _save_review(connection: sqlite3.Connection, review: UsageReview) -> None:
        request_row = connection.execute(
            "SELECT traffic_kind FROM assistant_requests WHERE request_id = ?",
            (review.request_id,),
        ).fetchone()
        if request_row is None:
            raise UsageRequestNotFound(review.request_id)
        if request_row[0] != "production":
            raise UsageReviewNotAllowed(review.request_id)
        connection.execute(
            """
            INSERT INTO request_reviews (
                request_id, review_status, failure_type, severity,
                expected_route, answerability, reason, eval_candidate,
                candidate_case_id, candidate_category, retrieval_case_id,
                expected_answer, evidence_json, topic, reviewed_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(request_id) DO UPDATE SET
                review_status = excluded.review_status,
                failure_type = excluded.failure_type,
                severity = excluded.severity,
                expected_route = excluded.expected_route,
                answerability = excluded.answerability,
                reason = excluded.reason,
                eval_candidate = excluded.eval_candidate,
                candidate_case_id = excluded.candidate_case_id,
                candidate_category = excluded.candidate_category,
                retrieval_case_id = excluded.retrieval_case_id,
                expected_answer = excluded.expected_answer,
                evidence_json = excluded.evidence_json,
                topic = excluded.topic,
                reviewed_at_utc = excluded.reviewed_at_utc
            """,
            (
                review.request_id,
                review.review_status,
                review.failure_type,
                review.severity,
                review.expected_route,
                None if review.answerability is None else int(review.answerability),
                review.reason,
                int(review.eval_candidate),
                review.candidate_case_id,
                review.candidate_category,
                review.retrieval_case_id,
                review.expected_answer,
                review.evidence_json,
                review.topic,
                utc_now_iso(),
            ),
        )

    def fetch_usage_records(
        self,
        *,
        started_from: str | None = None,
        started_to: str | None = None,
    ) -> list[dict[str, object]]:
        clauses: list[str] = []
        parameters: list[object] = []
        if started_from:
            clauses.append("r.started_at_utc >= ?")
            parameters.append(started_from)
        if started_to:
            clauses.append("r.started_at_utc < ?")
            parameters.append(started_to)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT r.*, f.rating AS feedback_rating,
                       v.review_status, v.failure_type, v.severity,
                       v.expected_route, v.answerability, v.reason,
                       v.eval_candidate, v.candidate_case_id,
                       v.candidate_category, v.retrieval_case_id,
                       v.expected_answer, v.evidence_json, v.topic
                FROM assistant_requests r
                LEFT JOIN request_feedback f ON f.request_id = r.request_id
                LEFT JOIN request_reviews v ON v.request_id = r.request_id
                """ + where + " ORDER BY r.started_at_utc, r.request_id",
                parameters,
            ).fetchall()
            records = [dict(row) for row in rows]
            for record in records:
                request_id = str(record["request_id"])
                record["timings_ms"] = {
                    str(row["stage"]): float(row["duration_ms"])
                    for row in connection.execute(
                        "SELECT stage, duration_ms FROM request_timings "
                        "WHERE request_id = ? ORDER BY stage",
                        (request_id,),
                    )
                }
                record["sources"] = [
                    dict(row)
                    for row in connection.execute(
                        """
                        SELECT rank, source, section, subsection,
                               dense_score, rerank_score
                        FROM request_sources WHERE request_id = ? ORDER BY rank
                        """,
                        (request_id,),
                    )
                ]
                record["required_constraints"] = _json_list(
                    record.pop("required_constraints_json")
                )
                record["missing_constraints"] = _json_list(
                    record.pop("missing_constraints_json")
                )
            return records

    def prune_expired(
        self, retention_days: int, *, now: datetime | None = None
    ) -> int:
        if retention_days <= 0:
            raise ValueError("retention_days must be positive")
        active_now = now or datetime.now(UTC)
        if active_now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        cutoff = (active_now.astimezone(UTC) - timedelta(days=retention_days)).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM assistant_requests WHERE started_at_utc < ?", (cutoff,)
            )
            return cursor.rowcount

    def backup(self, output_path: str | Path) -> Path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.open("xb").close()
        try:
            with self._connect() as source:
                target = sqlite3.connect(destination)
                try:
                    source.backup(target)
                finally:
                    target.close()
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        return destination

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self.database_path,
            timeout=self._busy_timeout_ms / 1000,
        )
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
            with connection:
                yield connection
        finally:
            connection.close()


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _validate_review(review: UsageReview) -> None:
    if review.review_status not in REVIEW_STATUSES:
        raise ValueError("unsupported review_status")
    if review.failure_type not in FAILURE_TYPES:
        raise ValueError("unsupported failure_type")
    if review.severity not in REVIEW_SEVERITIES:
        raise ValueError("unsupported severity")
    if review.expected_route not in {None, "rag", "chat", "out_of_scope"}:
        raise ValueError("unsupported expected_route")
    if not review.reason.strip():
        raise ValueError("review reason must not be empty")
    if review.candidate_category not in {None, *EVAL_CATEGORIES}:
        raise ValueError("unsupported candidate_category")
    if review.eval_candidate:
        if review.review_status != "user_confirmed":
            raise ValueError("Eval candidates must be user-confirmed")
        if not review.candidate_case_id or not review.candidate_category:
            raise ValueError("Eval candidates require case ID and category")
        if review.expected_route is None:
            raise ValueError("Eval candidates require expected_route")
        fixed_category_contracts = {
            "rag_answerable": ("rag", True),
            "rag_unanswerable": ("rag", False),
            "chat": ("chat", None),
            "out_of_scope": ("out_of_scope", None),
        }
        expected_contract = fixed_category_contracts.get(review.candidate_category)
        if expected_contract is not None and (
            review.expected_route,
            review.answerability,
        ) != expected_contract:
            raise ValueError(
                "Eval candidate category, expected_route, and answerability conflict"
            )
        if review.candidate_category == "routing_boundary" and (
            review.expected_route != "rag" or not isinstance(review.answerability, bool)
        ):
            raise ValueError(
                "Routing-boundary candidates require expected_route=rag "
                "and boolean answerability"
            )
        if review.candidate_category in {
            "rag_answerable",
            "rag_unanswerable",
            "routing_boundary",
        } and review.answerability is None:
            raise ValueError("RAG and routing-boundary candidates require answerability")
        if review.candidate_category == "rag_answerable":
            if review.answerability is not True:
                raise ValueError("Answerable RAG candidates require answerability=true")
            if not review.retrieval_case_id or not review.expected_answer:
                raise ValueError(
                    "Answerable RAG candidates require retrieval_case_id and expected_answer"
                )
            evidence = _json_list(review.evidence_json)
            if not evidence or not all(_valid_evidence(item) for item in evidence):
                raise ValueError("Answerable RAG candidates require strict evidence")


def _valid_evidence(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    return all(
        isinstance(value.get(field), str) and bool(str(value[field]).strip())
        for field in ("source", "evidence_text")
    )


def _source_records(value: object) -> list[tuple[object, ...]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    records = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        rank = item.get("rank")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            continue
        records.append(
            (
                rank,
                _optional_string(item.get("source")),
                _optional_string(item.get("section")),
                _optional_string(item.get("subsection")),
                _optional_number(item.get("dense_score")),
                _optional_number(item.get("rerank_score")),
            )
        )
    return records


def _numeric_items(value: Mapping[str, object] | None) -> list[tuple[str, float]]:
    if not isinstance(value, Mapping):
        return []
    return [
        (str(stage), float(duration))
        for stage, duration in value.items()
        if isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and duration >= 0
    ]


def _string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, str)]


def _json_list(value: object) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


_SCHEMA_V1 = """
CREATE TABLE assistant_requests (
    request_id TEXT PRIMARY KEY,
    endpoint TEXT NOT NULL CHECK (endpoint IN ('assistant_stream', 'ask')),
    started_at_utc TEXT NOT NULL,
    finished_at_utc TEXT,
    service_version TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    traffic_kind TEXT NOT NULL CHECK (
        traffic_kind IN ('production', 'assistant_eval', 'performance_eval')
    ),
    route TEXT CHECK (route IS NULL OR route IN ('rag', 'chat', 'out_of_scope')),
    execution_status TEXT NOT NULL CHECK (
        execution_status IN ('started', 'completed', 'failed', 'aborted', 'interrupted')
    ),
    question TEXT NOT NULL,
    answer TEXT NOT NULL DEFAULT '',
    retrieval_query TEXT,
    rewrite_query TEXT,
    query_rewrite_status TEXT,
    required_constraints_json TEXT NOT NULL DEFAULT '[]',
    missing_constraints_json TEXT NOT NULL DEFAULT '[]',
    error_code TEXT,
    error_type TEXT,
    failure_stage TEXT,
    total_duration_ms REAL CHECK (total_duration_ms IS NULL OR total_duration_ms >= 0)
);

CREATE TABLE request_timings (
    request_id TEXT NOT NULL REFERENCES assistant_requests(request_id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    duration_ms REAL NOT NULL CHECK (duration_ms >= 0),
    PRIMARY KEY (request_id, stage)
);

CREATE TABLE request_sources (
    request_id TEXT NOT NULL REFERENCES assistant_requests(request_id) ON DELETE CASCADE,
    rank INTEGER NOT NULL CHECK (rank > 0),
    source TEXT,
    section TEXT,
    subsection TEXT,
    dense_score REAL,
    rerank_score REAL,
    PRIMARY KEY (request_id, rank)
);

CREATE TABLE request_feedback (
    request_id TEXT PRIMARY KEY REFERENCES assistant_requests(request_id) ON DELETE CASCADE,
    rating TEXT NOT NULL CHECK (rating IN ('positive', 'negative')),
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL
);

CREATE TABLE request_reviews (
    request_id TEXT PRIMARY KEY REFERENCES assistant_requests(request_id) ON DELETE CASCADE,
    review_status TEXT NOT NULL CHECK (review_status IN ('pending', 'user_confirmed')),
    failure_type TEXT NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('none', 'minor', 'major')),
    expected_route TEXT CHECK (
        expected_route IS NULL OR expected_route IN ('rag', 'chat', 'out_of_scope')
    ),
    answerability INTEGER CHECK (answerability IS NULL OR answerability IN (0, 1)),
    reason TEXT NOT NULL,
    eval_candidate INTEGER NOT NULL CHECK (eval_candidate IN (0, 1)),
    candidate_case_id TEXT,
    candidate_category TEXT,
    retrieval_case_id TEXT,
    expected_answer TEXT,
    evidence_json TEXT,
    topic TEXT,
    reviewed_at_utc TEXT NOT NULL
);

CREATE INDEX idx_requests_started ON assistant_requests(started_at_utc);
CREATE INDEX idx_requests_endpoint_route ON assistant_requests(endpoint, route);
CREATE INDEX idx_requests_status ON assistant_requests(execution_status);
CREATE INDEX idx_feedback_rating ON request_feedback(rating);
CREATE INDEX idx_reviews_status_candidate ON request_reviews(review_status, eval_candidate);
"""

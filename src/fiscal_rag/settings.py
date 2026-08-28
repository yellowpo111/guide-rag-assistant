"""Validated environment settings for the internal Fiscal RAG service."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_DIRECTORY = PROJECT_ROOT / "data_private" / "corpus"
DEFAULT_INDEX_DIRECTORY = (
    PROJECT_ROOT / "data_private" / "indexes" / "fiscal_guides_chroma_v1"
)
DEFAULT_USAGE_DATABASE = (
    PROJECT_ROOT / "data_private" / "usage" / "fiscal_rag_usage.sqlite3"
)
DEFAULT_USAGE_RETENTION_DAYS = 90
DEFAULT_SERVICE_HOST = "127.0.0.1"
DEFAULT_SERVICE_PORT = 8000
DEFAULT_LOG_LEVEL = "INFO"
SUPPORTED_LOG_LEVELS = frozenset({"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"})


@dataclass(frozen=True)
class ServiceSettings:
    """Settings required to start the HTTP service safely."""

    corpus_dir: Path = DEFAULT_CORPUS_DIRECTORY
    index_dir: Path = DEFAULT_INDEX_DIRECTORY
    host: str = DEFAULT_SERVICE_HOST
    port: int = DEFAULT_SERVICE_PORT
    log_level: str = DEFAULT_LOG_LEVEL
    usage_db_path: Path = DEFAULT_USAGE_DATABASE
    usage_retention_days: int = DEFAULT_USAGE_RETENTION_DAYS

    @classmethod
    def from_environment(cls) -> ServiceSettings:
        """Load project .env values without overriding process-level settings."""
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        host = os.getenv("FISCAL_RAG_HOST", DEFAULT_SERVICE_HOST).strip()
        if not host:
            raise ValueError("FISCAL_RAG_HOST must not be empty.")

        log_level = os.getenv("FISCAL_RAG_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper().strip()
        if log_level not in SUPPORTED_LOG_LEVELS:
            raise ValueError(
                "FISCAL_RAG_LOG_LEVEL must be one of: "
                + ", ".join(sorted(SUPPORTED_LOG_LEVELS))
            )

        return cls(
            corpus_dir=_path_setting(
                "FISCAL_RAG_CORPUS_DIR", DEFAULT_CORPUS_DIRECTORY
            ),
            index_dir=_path_setting("FISCAL_RAG_INDEX_DIR", DEFAULT_INDEX_DIRECTORY),
            host=host,
            port=integer_setting(
                "FISCAL_RAG_PORT",
                DEFAULT_SERVICE_PORT,
                minimum=1,
                maximum=65535,
            ),
            log_level=log_level,
            usage_db_path=_path_setting(
                "FISCAL_RAG_USAGE_DB_PATH", DEFAULT_USAGE_DATABASE
            ),
            usage_retention_days=integer_setting(
                "FISCAL_RAG_USAGE_RETENTION_DAYS",
                DEFAULT_USAGE_RETENTION_DAYS,
                minimum=1,
            ),
        )


def positive_float_setting(name: str, default: float) -> float:
    """Read a positive floating-point setting."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive number.") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive number.")
    return parsed


def non_negative_integer_setting(name: str, default: int) -> int:
    """Read a non-negative integer setting."""
    return integer_setting(name, default, minimum=0)


def integer_setting(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    """Read an integer setting with inclusive bounds."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer.") from error
    if parsed < minimum or (maximum is not None and parsed > maximum):
        bounds = f"at least {minimum}"
        if maximum is not None:
            bounds = f"between {minimum} and {maximum}"
        raise ValueError(f"{name} must be {bounds}.")
    return parsed


def configure_service_logging(log_level: str) -> None:
    """Configure concise stdout logging without request or document content."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("fiscal_rag").setLevel(log_level)


def _path_setting(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser() if value and value.strip() else default

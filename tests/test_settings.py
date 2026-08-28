from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fiscal_rag.settings as settings_module  # noqa: E402
from fiscal_rag.settings import ServiceSettings  # noqa: E402


def test_service_settings_default_to_loopback_and_existing_private_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings_module, "load_dotenv", lambda *_args, **_kwargs: False)
    monkeypatch.delenv("FISCAL_RAG_SERVICE_TOKEN", raising=False)
    for name in (
        "FISCAL_RAG_HOST",
        "FISCAL_RAG_PORT",
        "FISCAL_RAG_LOG_LEVEL",
        "FISCAL_RAG_CORPUS_DIR",
        "FISCAL_RAG_INDEX_DIR",
        "FISCAL_RAG_USAGE_DB_PATH",
        "FISCAL_RAG_USAGE_RETENTION_DAYS",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = ServiceSettings.from_environment()

    assert settings.host == "127.0.0.1"
    assert settings.port == 8000
    assert settings.log_level == "INFO"
    assert settings.corpus_dir.name == "corpus"
    assert settings.index_dir.name == "fiscal_guides_chroma_v1"
    assert settings.usage_db_path.name == "fiscal_rag_usage.sqlite3"
    assert settings.usage_retention_days == 90


def test_service_settings_validate_port_and_log_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings_module, "load_dotenv", lambda *_args, **_kwargs: False)
    monkeypatch.setenv("FISCAL_RAG_PORT", "70000")

    with pytest.raises(ValueError, match="between 1 and 65535"):
        ServiceSettings.from_environment()

    monkeypatch.setenv("FISCAL_RAG_PORT", "8000")
    monkeypatch.setenv("FISCAL_RAG_LOG_LEVEL", "VERBOSE")
    with pytest.raises(ValueError, match="FISCAL_RAG_LOG_LEVEL"):
        ServiceSettings.from_environment()

    monkeypatch.setenv("FISCAL_RAG_LOG_LEVEL", "INFO")
    monkeypatch.setenv("FISCAL_RAG_USAGE_RETENTION_DAYS", "0")
    with pytest.raises(ValueError, match="at least 1"):
        ServiceSettings.from_environment()

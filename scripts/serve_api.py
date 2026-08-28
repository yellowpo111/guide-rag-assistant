"""Start the internal Fiscal RAG API with one Uvicorn worker."""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fiscal_rag.api import create_app  # noqa: E402
from fiscal_rag.settings import ServiceSettings  # noqa: E402


def main() -> None:
    settings = ServiceSettings.from_environment()
    uvicorn.run(
        create_app(settings=settings),
        host=settings.host,
        port=settings.port,
        workers=1,
        log_level=settings.log_level.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    main()

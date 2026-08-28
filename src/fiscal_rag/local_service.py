"""Temporary localhost service lifecycle for one-process evaluation runners."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from threading import Thread
from time import monotonic, sleep
from urllib.parse import urlsplit


@contextmanager
def local_service_if_requested(enabled: bool, base_url: str) -> Iterator[None]:
    if not enabled:
        yield
        return

    host, port = local_service_address(base_url)
    import uvicorn

    from fiscal_rag.api import create_app

    server = uvicorn.Server(
        uvicorn.Config(create_app(), host=host, port=port, log_level="warning")
    )
    thread = Thread(target=server.run, name="evaluation-uvicorn", daemon=True)
    thread.start()
    deadline = monotonic() + 60.0
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("Temporary local service exited before startup.")
        if monotonic() >= deadline:
            server.should_exit = True
            thread.join(timeout=5)
            raise TimeoutError("Temporary local service did not become ready.")
        sleep(0.05)
    try:
        yield
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        if thread.is_alive():
            server.force_exit = True
            thread.join(timeout=5)
        if thread.is_alive():
            raise RuntimeError("Temporary local service did not stop cleanly.")


def local_service_address(base_url: str) -> tuple[str, int]:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "A temporary local service requires an http://127.0.0.1 or "
            "http://localhost base URL without a path."
        )
    return "127.0.0.1", parsed.port or 80

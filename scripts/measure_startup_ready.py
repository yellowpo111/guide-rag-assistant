"""Measure cold process start until the deployed service reports ready."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter, sleep
from urllib.error import URLError
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
DEFAULT_RESULTS_DIRECTORY = PROJECT_ROOT / "data_private" / "evals" / "results"

from fiscal_rag.version import __version__  # noqa: E402


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure service cold-start readiness.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--output-file", type=Path, default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_arguments()
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    ready_url = args.base_url.rstrip("/") + "/health/ready"
    if _is_ready(ready_url):
        raise RuntimeError("A service is already ready at the probe URL; stop it first.")

    started = perf_counter()
    process = subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "serve_api.py")],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        while perf_counter() - started < args.timeout_seconds:
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout is not None else ""
                raise RuntimeError(f"Service exited before ready.\n{output}")
            if _is_ready(ready_url):
                elapsed_ms = (perf_counter() - started) * 1000
                break
            sleep(0.1)
        else:
            raise TimeoutError("Service did not become ready before the timeout.")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    created_at = datetime.now(UTC)
    output_file = args.output_file or (
        DEFAULT_RESULTS_DIRECTORY
        / f"startup_ready_v1_2_{created_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    record = {
        "schema_version": "startup-ready-result-v1",
        "release_version": f"v{__version__}",
        "created_at_utc": created_at.isoformat(),
        "startup_ready_ms": round(elapsed_ms, 3),
        "probe_path": "/health/ready",
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(record, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"Output File: {output_file}")


def _is_ready(url: str) -> bool:
    try:
        with urlopen(url, timeout=0.5) as response:  # noqa: S310
            return response.status == 200
    except (URLError, TimeoutError):
        return False


if __name__ == "__main__":
    main()

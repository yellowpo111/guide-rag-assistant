"""Run the independent qwen3-rerank smoke test from the project root."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fiscal_rag.reranker import main  # noqa: E402


if __name__ == "__main__":
    main()

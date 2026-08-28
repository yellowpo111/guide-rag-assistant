"""Run one non-sensitive DeepSeek query rewrite smoke test."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fiscal_rag.query_rewrite import DeepSeekQueryRewriter  # noqa: E402


def main() -> None:
    original_question = "已发布的采购意向需要改预算金额怎么办？"
    retrieval_query = DeepSeekQueryRewriter().rewrite(original_question)

    print(f"Original Question: {original_question}")
    print(f"Retrieval Query: {retrieval_query}")


if __name__ == "__main__":
    main()

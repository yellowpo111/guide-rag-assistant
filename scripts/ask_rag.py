"""Interactive local CLI for the current fiscal RAG pipeline."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIRECTORY = PROJECT_ROOT / "data_private" / "corpus"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fiscal_rag.pipeline import (  # noqa: E402
    RAGResult,
    build_persistent_dense_rerank_rag_pipeline,
)
from fiscal_rag.vector_store import (  # noqa: E402
    DEFAULT_PERSISTENT_INDEX_NAME,
    PersistentIndexError,
)


EXIT_COMMANDS = {"exit", "quit", "退出", "q"}
CONTENT_PREVIEW_LENGTH = 300
INDEX_DIRECTORY = PROJECT_ROOT / "data_private" / "indexes" / DEFAULT_PERSISTENT_INDEX_NAME


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask the current Rewrite + Guard + Dense-Rerank fiscal RAG pipeline."
    )
    parser.add_argument("--k", type=positive_integer, default=5, help="Final Top-k context size.")
    parser.add_argument(
        "--show-content-preview",
        action="store_true",
        help="Show a short preview of each retrieved chunk for local debugging.",
    )
    return parser.parse_args(argv)


def positive_integer(value: str) -> int:
    integer = int(value)
    if integer <= 0:
        raise argparse.ArgumentTypeError("k must be positive")
    return integer


def contains_surrogate_characters(value: str) -> bool:
    """Detect text that cannot be safely sent as UTF-8 to an API client."""
    return any(0xD800 <= ord(character) <= 0xDFFF for character in value)


def format_retrieval_debug(result: RAGResult, *, show_content_preview: bool) -> str:
    """Format source metadata without changing retrieved documents."""
    lines = ["检索到的参考资料："]
    for rank, (document, score) in enumerate(result.retrieved_results, start=1):
        metadata = document.metadata
        lines.extend(
            [
                f"{rank}. source: {metadata.get('source', 'Unknown')}",
                f"   section: {metadata.get('section', 'Unknown')}",
                f"   subsection: {metadata.get('subsection', 'Unknown')}",
                f"   score: {score:.6f}",
            ]
        )
        if show_content_preview:
            preview = document.page_content[:CONTENT_PREVIEW_LENGTH].replace("\n", " ")
            lines.append(f"   preview: {preview}")
    return "\n".join(lines)


def print_result(result: RAGResult, *, show_content_preview: bool, output: Callable[[str], None]) -> None:
    """Display answer and retrieval metadata for one completed request."""
    output("")
    output(f"实际检索问题：{result.retrieval_query}")
    if result.rewrite_query is not None:
        output(f"改写问题：{result.rewrite_query}")
        output(f"Guard 状态：{result.query_rewrite_status}")
    output("")
    output(format_retrieval_debug(result, show_content_preview=show_content_preview))
    output("")
    output("最终回答：")
    output(result.answer)


def run_interactive(
    pipeline: object,
    *,
    k: int,
    show_content_preview: bool,
    input_fn: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
) -> None:
    """Read questions until the user exits, preserving one pipeline instance."""
    output("Fiscal RAG 已启动。输入问题；输入 exit、quit、q 或 退出结束。")
    while True:
        try:
            question = input_fn("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            output("\n已退出 Fiscal RAG。")
            return

        if not question:
            output("请输入非空问题，或输入 exit 退出。")
            continue
        if contains_surrogate_characters(question):
            output("输入编码异常，请直接在交互终端重新输入问题。")
            continue
        if question.lower() in EXIT_COMMANDS:
            output("已退出 Fiscal RAG。")
            return

        try:
            result = pipeline.answer(question, k=k)
        except Exception as error:
            output(f"请求失败：{type(error).__name__}: {error}")
            continue
        print_result(result, show_content_preview=show_content_preview, output=output)


def main() -> None:
    args = parse_arguments()
    print("正在加载持久化 Chroma 索引与当前 RAG pipeline...")
    try:
        pipeline = build_persistent_dense_rerank_rag_pipeline(
            CORPUS_DIRECTORY,
            INDEX_DIRECTORY,
        )
    except PersistentIndexError as error:
        print(f"启动失败：{error}")
        print("请先运行：.\\.venv\\Scripts\\python.exe scripts\\build_vector_index.py")
        raise SystemExit(1) from error
    except Exception as error:
        print(f"启动失败：{type(error).__name__}: {error}")
        raise SystemExit(1) from error

    run_interactive(
        pipeline,
        k=args.k,
        show_content_preview=args.show_content_preview,
    )


if __name__ == "__main__":
    main()

import importlib.util
from argparse import Namespace
from pathlib import Path
import sys

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fiscal_rag.pipeline import RAGResult  # noqa: E402


def load_cli_module():
    script_path = PROJECT_ROOT / "scripts" / "ask_rag.py"
    spec = importlib.util.spec_from_file_location("ask_rag", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_result() -> RAGResult:
    document = Document(
        page_content="办理完成后点击保存。",
        metadata={"source": "guide.md", "section": "填报", "subsection": "操作步骤"},
    )
    return RAGResult(
        question="怎样保存？",
        retrieved_results=[(document, 0.9)],
        context="context",
        answer="点击保存。",
        retrieval_query="如何保存？",
        rewrite_query="如何保存？",
        query_rewrite_status="accepted",
    )


def test_cli_defaults_to_top_five() -> None:
    cli = load_cli_module()

    arguments = cli.parse_arguments([])

    assert arguments.k == 5
    assert arguments.show_content_preview is False


def test_cli_formats_source_metadata_and_optional_preview() -> None:
    cli = load_cli_module()

    without_preview = cli.format_retrieval_debug(make_result(), show_content_preview=False)
    with_preview = cli.format_retrieval_debug(make_result(), show_content_preview=True)

    assert "source: guide.md" in without_preview
    assert "section: 填报" in without_preview
    assert "subsection: 操作步骤" in without_preview
    assert "preview:" not in without_preview
    assert "preview: 办理完成后点击保存。" in with_preview


def test_cli_runs_one_question_then_exits() -> None:
    cli = load_cli_module()
    answers = []

    class FakePipeline:
        def answer(self, question: str, *, k: int) -> RAGResult:
            assert question == "怎样保存？"
            assert k == 5
            return make_result()

    inputs = iter(["怎样保存？", "exit"])
    cli.run_interactive(
        FakePipeline(),
        k=5,
        show_content_preview=False,
        input_fn=lambda _prompt: next(inputs),
        output=answers.append,
    )

    assert any("最终回答：" == line for line in answers)
    assert "点击保存。" in answers
    assert answers[-1] == "已退出 Fiscal RAG。"


def test_cli_rejects_surrogate_input_before_calling_pipeline() -> None:
    cli = load_cli_module()
    answers = []

    class FakePipeline:
        def answer(self, _question: str, *, k: int) -> RAGResult:
            raise AssertionError("Malformed text must not reach the pipeline.")

    inputs = iter(["\udcff", "exit"])
    cli.run_interactive(
        FakePipeline(),
        k=5,
        show_content_preview=False,
        input_fn=lambda _prompt: next(inputs),
        output=answers.append,
    )

    assert "输入编码异常，请直接在交互终端重新输入问题。" in answers


def test_cli_main_uses_persistent_pipeline_builder(monkeypatch) -> None:
    cli = load_cli_module()
    calls = {}

    monkeypatch.setattr(
        cli,
        "parse_arguments",
        lambda: Namespace(k=5, show_content_preview=False),
    )
    monkeypatch.setattr(
        cli,
        "build_persistent_dense_rerank_rag_pipeline",
        lambda corpus_dir, index_dir: calls.update(
            {"corpus_dir": corpus_dir, "index_dir": index_dir}
        )
        or object(),
    )
    monkeypatch.setattr(cli, "run_interactive", lambda *_args, **_kwargs: None)

    cli.main()

    assert calls["corpus_dir"] == cli.CORPUS_DIRECTORY
    assert calls["index_dir"] == cli.INDEX_DIRECTORY

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fiscal_rag.ingestion import (  # noqa: E402
    load_markdown_documents,
    split_markdown_documents,
)


def test_load_markdown_documents_reads_only_markdown_files(tmp_path: Path) -> None:
    markdown_path = tmp_path / "财报操作指南.md"
    markdown_path.write_text("# 数据填报\n\n测试内容。", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not a knowledge document", encoding="utf-8")

    documents = load_markdown_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].page_content == "# 数据填报\n\n测试内容。"
    assert documents[0].metadata == {"source": "财报操作指南.md"}


def test_loading_corpus_directory_does_not_read_sibling_eval_artifacts(
    tmp_path: Path,
) -> None:
    private_root = tmp_path / "data_private"
    corpus_dir = private_root / "corpus"
    eval_results_dir = private_root / "evals" / "results"
    corpus_dir.mkdir(parents=True)
    eval_results_dir.mkdir(parents=True)
    (corpus_dir / "操作指南.md").write_text("# 操作", encoding="utf-8")
    (eval_results_dir / "error_analysis.md").write_text("# Eval artifact", encoding="utf-8")

    documents = load_markdown_documents(corpus_dir)

    assert [document.metadata["source"] for document in documents] == ["操作指南.md"]


def test_structure_aware_split_keeps_heading_metadata_and_text(tmp_path: Path) -> None:
    guide = tmp_path / "操作指南.md"
    guide.write_text(
        "# 数据填报\n\n## 系统常见操作指引\n\n这里是测试操作内容。\n\n"
        "# 对账事项\n\n## 业务操作问答\n\n这里是测试问答。\n",
        encoding="utf-8",
    )

    chunks = split_markdown_documents(load_markdown_documents(tmp_path))

    assert len(chunks) == 2
    assert chunks[0].metadata == {
        "source": "操作指南.md",
        "section": "数据填报",
        "subsection": "系统常见操作指引",
    }
    assert chunks[1].metadata == {
        "source": "操作指南.md",
        "section": "对账事项",
        "subsection": "业务操作问答",
    }
    assert "# 数据填报" in chunks[0].page_content
    assert "## 业务操作问答" in chunks[1].page_content


def test_oversized_markdown_section_is_recursively_split_and_keeps_source(
    tmp_path: Path,
) -> None:
    guide = tmp_path / "长文档.md"
    guide.write_text(
        "# 数据填报\n\n## 系统常见操作指引\n\n" + "这是很长的测试操作内容。" * 100,
        encoding="utf-8",
    )

    chunks = split_markdown_documents(
        load_markdown_documents(tmp_path),
        chunk_size=120,
        chunk_overlap=20,
    )

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "长文档.md" for chunk in chunks)
    assert all(chunk.metadata["section"] == "数据填报" for chunk in chunks)
    assert all(chunk.metadata["subsection"] == "系统常见操作指引" for chunk in chunks)


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (100, -1), (100, 100)],
)
def test_invalid_chunk_parameters_raise_value_error(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    with pytest.raises(ValueError):
        split_markdown_documents([], chunk_size=chunk_size, chunk_overlap=chunk_overlap)

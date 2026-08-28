from pathlib import Path
import sys

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import fiscal_rag.vector_store as vector_store_module  # noqa: E402
from fiscal_rag.vector_store import (  # noqa: E402
    PersistentIndexError,
    build_persistent_chroma_index,
    open_persistent_chroma_vector_store,
    stable_chunk_id,
)


class DirectionEmbeddings(Embeddings):
    model = "fake-embedding-v1"

    def __init__(self) -> None:
        self.document_batches: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_batches.append(list(texts))
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        return [1.0, 0.0] if "alpha" in text else [0.0, 1.0]


class DifferentModelEmbeddings(DirectionEmbeddings):
    model = "fake-embedding-v2"


class ControlledFailureEmbeddings(DirectionEmbeddings):
    def __init__(self) -> None:
        super().__init__()
        self.fail_documents = False
        self.fail_query = False
        self.inconsistent_dimensions = False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.fail_documents:
            raise RuntimeError("simulated document embedding failure")
        vectors = super().embed_documents(texts)
        if self.inconsistent_dimensions and len(self.document_batches) % 2 == 0:
            return [vector + [0.0] for vector in vectors]
        return vectors

    def embed_query(self, text: str) -> list[float]:
        if self.fail_query:
            raise RuntimeError("simulated query embedding failure")
        return super().embed_query(text)


def write_corpus(corpus_dir: Path) -> None:
    corpus_dir.mkdir()
    (corpus_dir / "alpha.md").write_text(
        "# Alpha\n\n## Steps\n\nalpha document", encoding="utf-8"
    )
    (corpus_dir / "beta.md").write_text(
        "# Beta\n\n## Steps\n\nbeta document", encoding="utf-8"
    )


def test_persistent_index_survives_reopen_and_preserves_metadata(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    index_dir = tmp_path / "index"
    write_corpus(corpus_dir)
    embeddings = DirectionEmbeddings()

    created = build_persistent_chroma_index(corpus_dir, index_dir, embeddings)
    reused = build_persistent_chroma_index(corpus_dir, index_dir, embeddings)
    store = open_persistent_chroma_vector_store(corpus_dir, index_dir, embeddings)
    results = store.similarity_search_with_score("query alpha", k=2)

    assert created.created is True
    assert reused.created is False
    assert len(embeddings.document_batches) == 1
    assert created.raw_document_count == 2
    assert created.chunk_count == 2
    assert (index_dir / "manifest.json").is_file()
    assert results[0][0].metadata == {
        "source": "alpha.md",
        "section": "Alpha",
        "subsection": "Steps",
    }
    assert results[0][0].page_content.endswith("alpha document")
    assert results[0][1] > results[1][1]


def test_persistent_index_rejects_stale_corpus_or_embedding_model(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    corpus_dir = tmp_path / "corpus"
    index_dir = tmp_path / "index"
    write_corpus(corpus_dir)
    embeddings = DirectionEmbeddings()
    build_persistent_chroma_index(corpus_dir, index_dir, embeddings)

    with pytest.raises(PersistentIndexError, match="embedding model"):
        open_persistent_chroma_vector_store(
            corpus_dir, index_dir, DifferentModelEmbeddings()
        )

    (corpus_dir / "alpha.md").write_text(
        "# Alpha\n\n## Steps\n\nchanged alpha document", encoding="utf-8"
    )
    with pytest.raises(PersistentIndexError, match="corpus content"):
        open_persistent_chroma_vector_store(corpus_dir, index_dir, embeddings)
    store = open_persistent_chroma_vector_store(
        corpus_dir,
        index_dir,
        embeddings,
        allow_unpublished_corpus_changes=True,
    )
    assert "corpus_has_unpublished_changes" in caplog.text
    assert "alpha.md" not in caplog.text
    assert "alpha document" in store.similarity_search_with_score(
        "query alpha", k=1
    )[0][0].page_content
    store.close()
    with pytest.raises(PersistentIndexError, match="--rebuild"):
        build_persistent_chroma_index(corpus_dir, index_dir, embeddings)


def test_rebuild_explicitly_replaces_stale_derived_index(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    index_dir = tmp_path / "index"
    write_corpus(corpus_dir)
    embeddings = DirectionEmbeddings()
    build_persistent_chroma_index(corpus_dir, index_dir, embeddings)
    (corpus_dir / "alpha.md").write_text(
        "# Alpha\n\n## Steps\n\nchanged alpha document", encoding="utf-8"
    )

    rebuilt = build_persistent_chroma_index(
        corpus_dir, index_dir, embeddings, rebuild=True
    )
    results = open_persistent_chroma_vector_store(
        corpus_dir, index_dir, embeddings
    ).similarity_search_with_score("query alpha", k=1)

    assert rebuilt.created is True
    assert "changed alpha document" in results[0][0].page_content


def test_full_rebuild_adds_modifies_renames_and_removes_documents(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    index_dir = tmp_path / "index"
    write_corpus(corpus_dir)
    embeddings = DirectionEmbeddings()
    build_persistent_chroma_index(corpus_dir, index_dir, embeddings)

    (corpus_dir / "alpha.md").rename(corpus_dir / "renamed-alpha.md")
    (corpus_dir / "renamed-alpha.md").write_text(
        "# Alpha\n\n## Steps\n\nupdated alpha document", encoding="utf-8"
    )
    (corpus_dir / "beta.md").unlink()
    (corpus_dir / "gamma.md").write_text(
        "# Gamma\n\n## Steps\n\ngamma document", encoding="utf-8"
    )

    rebuilt = build_persistent_chroma_index(
        corpus_dir, index_dir, embeddings, rebuild=True
    )
    store = open_persistent_chroma_vector_store(corpus_dir, index_dir, embeddings)
    documents = [item[0] for item in store.similarity_search_with_score("query", k=10)]
    store.close()

    assert rebuilt.raw_document_count == 2
    assert rebuilt.chunk_count == 2
    assert {document.metadata["source"] for document in documents} == {
        "gamma.md",
        "renamed-alpha.md",
    }
    contents = "\n".join(document.page_content for document in documents)
    assert "updated alpha document" in contents
    assert "gamma document" in contents
    assert "beta document" not in contents
    assert "# Beta" not in contents


@pytest.mark.parametrize("failure_kind", ["documents", "query", "dimensions"])
def test_failed_staged_build_keeps_last_successful_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    corpus_dir = tmp_path / "corpus"
    index_dir = tmp_path / "index"
    write_corpus(corpus_dir)
    embeddings = ControlledFailureEmbeddings()
    build_persistent_chroma_index(corpus_dir, index_dir, embeddings)
    previous_manifest = (index_dir / "manifest.json").read_bytes()
    (corpus_dir / "alpha.md").write_text(
        "# Alpha\n\n## Steps\n\nchanged alpha document", encoding="utf-8"
    )

    if failure_kind == "documents":
        embeddings.fail_documents = True
    elif failure_kind == "query":
        embeddings.fail_query = True
    else:
        embeddings.document_batches.clear()
        embeddings.inconsistent_dimensions = True
        monkeypatch.setattr(vector_store_module, "INDEX_WRITE_BATCH_SIZE", 1)

    with pytest.raises(PersistentIndexError):
        build_persistent_chroma_index(
            corpus_dir, index_dir, embeddings, rebuild=True
        )

    embeddings.fail_documents = False
    embeddings.fail_query = False
    embeddings.inconsistent_dimensions = False
    assert (index_dir / "manifest.json").read_bytes() == previous_manifest
    assert not list(tmp_path.glob(".index.staging-*"))
    store = open_persistent_chroma_vector_store(
        corpus_dir,
        index_dir,
        embeddings,
        allow_unpublished_corpus_changes=True,
    )
    assert "alpha document" in store.similarity_search_with_score(
        "query alpha", k=1
    )[0][0].page_content
    store.close()


def test_corpus_change_during_build_keeps_last_successful_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_dir = tmp_path / "corpus"
    index_dir = tmp_path / "index"
    write_corpus(corpus_dir)
    embeddings = DirectionEmbeddings()
    build_persistent_chroma_index(corpus_dir, index_dir, embeddings)
    previous_manifest = (index_dir / "manifest.json").read_bytes()
    (corpus_dir / "alpha.md").write_text(
        "# Alpha\n\n## Steps\n\nchanged alpha document", encoding="utf-8"
    )
    real_manifest_builder = vector_store_module.build_corpus_source_manifest
    calls = 0

    def changing_manifest(path: Path) -> list[dict[str, object]]:
        nonlocal calls
        calls += 1
        if calls == 2:
            (corpus_dir / "beta.md").write_text(
                "# Beta\n\n## Steps\n\nchanged during build", encoding="utf-8"
            )
        return real_manifest_builder(path)

    monkeypatch.setattr(
        vector_store_module, "build_corpus_source_manifest", changing_manifest
    )

    with pytest.raises(PersistentIndexError, match="changed while"):
        build_persistent_chroma_index(
            corpus_dir, index_dir, embeddings, rebuild=True
        )

    assert (index_dir / "manifest.json").read_bytes() == previous_manifest


@pytest.mark.parametrize("failed_move", ["old_to_backup", "stage_to_active"])
def test_activation_failure_restores_previous_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_move: str,
) -> None:
    corpus_dir = tmp_path / "corpus"
    index_dir = tmp_path / "index"
    write_corpus(corpus_dir)
    embeddings = DirectionEmbeddings()
    build_persistent_chroma_index(corpus_dir, index_dir, embeddings)
    previous_manifest = (index_dir / "manifest.json").read_bytes()
    (corpus_dir / "alpha.md").write_text(
        "# Alpha\n\n## Steps\n\nchanged alpha document", encoding="utf-8"
    )
    real_replace = Path.replace

    def failing_replace(path: Path, target: Path) -> Path:
        target_path = Path(target)
        if failed_move == "old_to_backup" and path == index_dir:
            raise OSError("simulated first activation rename failure")
        if (
            failed_move == "stage_to_active"
            and path.name.startswith(".index.staging-")
            and target_path == index_dir
        ):
            raise OSError("simulated second activation rename failure")
        return real_replace(path, target_path)

    monkeypatch.setattr(Path, "replace", failing_replace)

    with pytest.raises(PersistentIndexError):
        build_persistent_chroma_index(
            corpus_dir, index_dir, embeddings, rebuild=True
        )

    assert (index_dir / "manifest.json").read_bytes() == previous_manifest
    assert not (tmp_path / ".index.backup").exists()


def test_next_maintenance_run_recovers_backup_after_rollback_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_dir = tmp_path / "corpus"
    index_dir = tmp_path / "index"
    write_corpus(corpus_dir)
    embeddings = DirectionEmbeddings()
    build_persistent_chroma_index(corpus_dir, index_dir, embeddings)
    previous_manifest = (index_dir / "manifest.json").read_bytes()
    (corpus_dir / "alpha.md").write_text(
        "# Alpha\n\n## Steps\n\nchanged alpha document", encoding="utf-8"
    )
    real_replace = Path.replace

    def failing_replace(path: Path, target: Path) -> Path:
        target_path = Path(target)
        if path.name.startswith(".index.staging-") and target_path == index_dir:
            raise OSError("simulated activation failure")
        if path == tmp_path / ".index.backup" and target_path == index_dir:
            raise OSError("simulated rollback failure")
        return real_replace(path, target_path)

    monkeypatch.setattr(Path, "replace", failing_replace)
    with pytest.raises(PersistentIndexError, match="rollback both failed"):
        build_persistent_chroma_index(
            corpus_dir, index_dir, embeddings, rebuild=True
        )
    assert not index_dir.exists()
    assert (tmp_path / ".index.backup").exists()

    monkeypatch.setattr(Path, "replace", real_replace)
    with pytest.raises(PersistentIndexError, match="--rebuild"):
        build_persistent_chroma_index(corpus_dir, index_dir, embeddings)

    assert (index_dir / "manifest.json").read_bytes() == previous_manifest
    assert not (tmp_path / ".index.backup").exists()


def test_document_without_chunks_is_rejected_without_replacing_index(
    tmp_path: Path,
) -> None:
    corpus_dir = tmp_path / "corpus"
    index_dir = tmp_path / "index"
    write_corpus(corpus_dir)
    embeddings = DirectionEmbeddings()
    build_persistent_chroma_index(corpus_dir, index_dir, embeddings)
    previous_manifest = (index_dir / "manifest.json").read_bytes()
    (corpus_dir / "empty.md").write_text("", encoding="utf-8")

    with pytest.raises(PersistentIndexError, match="at least one chunk"):
        build_persistent_chroma_index(
            corpus_dir, index_dir, embeddings, rebuild=True
        )

    assert (index_dir / "manifest.json").read_bytes() == previous_manifest


def test_stable_chunk_id_is_repeatable_and_position_sensitive() -> None:
    document = Document(
        page_content="alpha document",
        metadata={"source": "alpha.md", "section": "Alpha", "subsection": "Steps"},
    )

    assert stable_chunk_id(document, 0) == stable_chunk_id(document, 0)
    assert stable_chunk_id(document, 0) != stable_chunk_id(document, 1)

"""Vector-store helpers for frozen in-memory evals and persistent local serving."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import chromadb
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from fiscal_rag.ingestion import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    ingest_markdown_directory,
)
from fiscal_rag.timing import measure_stage


DEFAULT_PERSISTENT_INDEX_NAME = "fiscal_guides_chroma_v1"
DEFAULT_CHROMA_COLLECTION_NAME = "fiscal_guide_chunks_v1"
PERSISTENT_INDEX_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
CHROMA_DIRECTORY_NAME = "chroma"
INDEX_WRITE_BATCH_SIZE = 100
INDEX_VALIDATION_QUERY = "knowledge index validation"


LOGGER = logging.getLogger(__name__)


class PersistentIndexError(RuntimeError):
    """Raised when a local vector index is missing, stale, or inconsistent."""


@dataclass(frozen=True)
class PersistentIndexBuildResult:
    """Outcome of a create-or-validate persistent index request."""

    index_dir: Path
    manifest: Mapping[str, object]
    created: bool

    @property
    def raw_document_count(self) -> int:
        return int(self.manifest["raw_document_count"])

    @property
    def chunk_count(self) -> int:
        return int(self.manifest["chunk_count"])


class PersistentChromaVectorStore:
    """Expose a persisted Chroma collection through the existing dense-store API."""

    def __init__(self, client: Any, collection: Any, embeddings: Embeddings) -> None:
        self._client = client
        self._collection = collection
        self._embeddings = embeddings

    def close(self) -> None:
        """Release Chroma resources, including persistent SQLite file handles."""
        self._client.close()

    def similarity_search_with_score(
        self, query: str, *, k: int = 4
    ) -> list[tuple[Document, float]]:
        """Return Documents with cosine similarities, preserving stored metadata."""
        if k <= 0:
            raise ValueError("k must be positive")

        available = self._collection.count()
        if available == 0:
            return []

        with measure_stage("query_embedding"):
            query_embedding = self._embeddings.embed_query(query)
        with measure_stage("vector_search"):
            response = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, available),
                include=["documents", "metadatas", "distances"],
            )
        documents = _first_query_batch(response.get("documents"))
        metadatas = _first_query_batch(response.get("metadatas"))
        distances = _first_query_batch(response.get("distances"))

        results: list[tuple[Document, float]] = []
        for page_content, metadata, distance in zip(documents, metadatas, distances):
            if not isinstance(page_content, str):
                raise PersistentIndexError("Chroma index contains a non-text document.")
            if not isinstance(metadata, Mapping):
                raise PersistentIndexError("Chroma index contains invalid document metadata.")
            # The collection is configured with cosine distance. Preserve the existing
            # project convention that larger score means more similar.
            similarity = 1.0 - float(distance)
            results.append(
                (Document(page_content=page_content, metadata=dict(metadata)), similarity)
            )
        return results


def build_in_memory_vector_store(
    documents: list[Document],
    embeddings: Embeddings,
) -> InMemoryVectorStore:
    """Embed documents and add them to the frozen in-memory eval store."""
    vector_store = InMemoryVectorStore(embedding=embeddings)
    if documents:
        vector_store.add_documents(documents)
    return vector_store


def build_persistent_chroma_index(
    corpus_dir: str | Path,
    index_dir: str | Path,
    embeddings: Embeddings,
    *,
    rebuild: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> PersistentIndexBuildResult:
    """Create a private Chroma index, or validate and reuse a current one.

    A stale index is never silently rebuilt because rebuilding calls the document
    embedding API. Callers must explicitly pass ``rebuild=True``.
    """
    corpus_path = Path(corpus_dir)
    index_path = Path(index_dir)

    try:
        _recover_interrupted_index_swap(index_path)
        source_manifest = build_corpus_source_manifest(corpus_path)
        expected = _expected_manifest_fields(
            source_manifest,
            embeddings,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        manifest_path = index_path / MANIFEST_FILENAME

        if manifest_path.is_file():
            existing_manifest = load_index_manifest(index_path)
            differences = _manifest_differences(existing_manifest, expected)
            if not differences and not rebuild:
                _validate_persisted_collection(index_path, existing_manifest)
                return PersistentIndexBuildResult(
                    index_dir=index_path,
                    manifest=existing_manifest,
                    created=False,
                )
            if differences and not rebuild:
                raise PersistentIndexError(
                    "Persistent index is stale ("
                    + "; ".join(differences)
                    + "). Rebuild it with: python scripts/build_vector_index.py --rebuild"
                )
        elif _index_directory_has_files(index_path) and not rebuild:
            raise PersistentIndexError(
                "Persistent index directory exists but manifest.json is missing. "
                "Rebuild it with: python scripts/build_vector_index.py --rebuild"
            )

        index_path.parent.mkdir(parents=True, exist_ok=True)
        staging_path = _staging_path(index_path)
        try:
            manifest = _build_staged_index(
                corpus_path,
                staging_path,
                embeddings,
                expected,
                source_manifest,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            if list(source_manifest) != build_corpus_source_manifest(corpus_path):
                raise PersistentIndexError(
                    "Corpus changed before the staged index could be activated; "
                    "retry the rebuild."
                )
            _activate_staged_index(
                staging_path,
                index_path,
                corpus_path,
                source_manifest,
            )
        except BaseException:
            _remove_directory_if_present(staging_path, warning_only=True)
            raise

        return PersistentIndexBuildResult(
            index_dir=index_path,
            manifest=manifest,
            created=True,
        )
    except PersistentIndexError:
        raise
    except Exception as error:
        raise PersistentIndexError(f"Persistent index build failed: {error}") from error


def open_persistent_chroma_vector_store(
    corpus_dir: str | Path,
    index_dir: str | Path,
    embeddings: Embeddings,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    allow_unpublished_corpus_changes: bool = False,
) -> PersistentChromaVectorStore:
    """Load a current private index without loading chunks or document embeddings."""
    corpus_path = Path(corpus_dir)
    index_path = Path(index_dir)
    if not (index_path / MANIFEST_FILENAME).is_file():
        raise PersistentIndexError(
            "Persistent index is missing. Build it first with: "
            "python scripts/build_vector_index.py"
        )

    manifest = load_index_manifest(index_path)
    expected = _expected_manifest_fields(
        build_corpus_source_manifest(corpus_path),
        embeddings,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    difference_fields = _manifest_difference_fields(manifest, expected)
    unpublished_fields = {"corpus_sha256", "sources"}
    if difference_fields and not (
        allow_unpublished_corpus_changes
        and set(difference_fields).issubset(unpublished_fields)
    ):
        raise PersistentIndexError(
            "Persistent index is stale ("
            + "; ".join(_difference_labels(difference_fields))
            + "). Rebuild it with: python scripts/build_vector_index.py --rebuild"
        )
    if difference_fields:
        LOGGER.warning(
            "corpus_has_unpublished_changes; serving the last successfully built index"
        )

    client, collection = _open_validated_collection(index_path, manifest)
    return PersistentChromaVectorStore(client, collection, embeddings)


def load_index_manifest(index_dir: str | Path) -> dict[str, object]:
    """Read and minimally validate a persistent index manifest."""
    manifest_path = Path(index_dir) / MANIFEST_FILENAME
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PersistentIndexError(f"Missing index manifest: {manifest_path}") from error
    except json.JSONDecodeError as error:
        raise PersistentIndexError(f"Invalid index manifest JSON: {manifest_path}") from error
    if not isinstance(data, dict):
        raise PersistentIndexError("Persistent index manifest must be a JSON object.")
    if data.get("schema_version") != PERSISTENT_INDEX_SCHEMA_VERSION:
        raise PersistentIndexError(
            "Persistent index schema version is unsupported. Rebuild it with: "
            "python scripts/build_vector_index.py --rebuild"
        )
    return data


def build_corpus_source_manifest(corpus_dir: str | Path) -> list[dict[str, object]]:
    """Return a deterministic source manifest without storing source text itself."""
    root = Path(corpus_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Markdown directory does not exist: {root}")
    records = []
    for path in sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file() and candidate.suffix.lower() == ".md"),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    ):
        content = path.read_bytes()
        records.append(
            {
                "source": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(content).hexdigest(),
                "byte_count": len(content),
            }
        )
    return records


def stable_chunk_id(document: Document, position: int) -> str:
    """Build an index-only deterministic id from a chunk and its split position."""
    metadata = document.metadata
    payload = {
        "position": position,
        "source": metadata.get("source"),
        "section": metadata.get("section"),
        "subsection": metadata.get("subsection"),
        "page_content": document.page_content,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"chunk-{position:06d}-{digest}"


def _expected_manifest_fields(
    source_manifest: Sequence[Mapping[str, object]],
    embeddings: Embeddings,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> dict[str, object]:
    normalized_sources = [dict(record) for record in source_manifest]
    serialized_sources = json.dumps(
        normalized_sources,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "schema_version": PERSISTENT_INDEX_SCHEMA_VERSION,
        "collection_name": DEFAULT_CHROMA_COLLECTION_NAME,
        "embedding_model": _embedding_model_name(embeddings),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "corpus_sha256": hashlib.sha256(serialized_sources.encode("utf-8")).hexdigest(),
        "sources": normalized_sources,
    }


def _manifest_differences(
    manifest: Mapping[str, object], expected: Mapping[str, object]
) -> list[str]:
    return _difference_labels(_manifest_difference_fields(manifest, expected))


def _manifest_difference_fields(
    manifest: Mapping[str, object], expected: Mapping[str, object]
) -> list[str]:
    return [field for field in _MANIFEST_FIELD_LABELS if manifest.get(field) != expected.get(field)]


_MANIFEST_FIELD_LABELS = {
    "schema_version": "schema version",
    "collection_name": "collection name",
    "embedding_model": "embedding model",
    "chunk_size": "chunk size",
    "chunk_overlap": "chunk overlap",
    "corpus_sha256": "corpus content",
    "sources": "source manifest",
}


def _difference_labels(fields: Sequence[str]) -> list[str]:
    return [_MANIFEST_FIELD_LABELS[field] for field in fields]


def _embedding_model_name(embeddings: Embeddings) -> str:
    configured_model = getattr(embeddings, "model", None)
    if isinstance(configured_model, str) and configured_model:
        return configured_model
    return f"{type(embeddings).__module__}.{type(embeddings).__qualname__}"


def _persistent_client(index_dir: Path) -> Any:
    chroma_path = index_dir / CHROMA_DIRECTORY_NAME
    chroma_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_path))


def _validate_persisted_collection(
    index_path: Path, manifest: Mapping[str, object]
) -> None:
    """Open and close the collection after checking its internal structure."""
    client, _ = _open_validated_collection(index_path, manifest)
    client.close()


def _open_validated_collection(
    index_path: Path, manifest: Mapping[str, object]
) -> tuple[Any, Any]:
    client = _persistent_client(index_path)
    try:
        collection_name = str(manifest["collection_name"])
        collection_names = sorted(
            item if isinstance(item, str) else item.name
            for item in client.list_collections()
        )
        if collection_names != [collection_name]:
            raise PersistentIndexError(
                "Persistent index must contain exactly its manifest collection."
            )
        collection = client.get_collection(
            collection_name,
            embedding_function=None,
        )
        chunk_count = _positive_manifest_integer(manifest, "chunk_count")
        _positive_manifest_integer(manifest, "raw_document_count")
        _positive_manifest_integer(manifest, "embedding_dimension")
        if collection.count() != chunk_count:
            raise PersistentIndexError(
                "Persistent index chunk count does not match its manifest. Rebuild it with: "
                "python scripts/build_vector_index.py --rebuild"
            )
        return client, collection
    except Exception as error:
        client.close()
        if isinstance(error, PersistentIndexError):
            raise
        raise PersistentIndexError(
            "Persistent index collection is missing or invalid. Rebuild it with: "
            "python scripts/build_vector_index.py --rebuild"
        ) from error


def _positive_manifest_integer(manifest: Mapping[str, object], field: str) -> int:
    value = manifest.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise PersistentIndexError(f"Persistent index manifest has invalid {field}.")
    return value


def _build_staged_index(
    corpus_path: Path,
    staging_path: Path,
    embeddings: Embeddings,
    expected: Mapping[str, object],
    source_manifest: Sequence[Mapping[str, object]],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> dict[str, object]:
    raw_documents, chunks = ingest_markdown_directory(
        corpus_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not raw_documents or not chunks:
        raise PersistentIndexError("Cannot create a persistent index from an empty corpus.")

    raw_sources = {str(document.metadata.get("source", "")) for document in raw_documents}
    chunk_sources = {str(document.metadata.get("source", "")) for document in chunks}
    if "" in raw_sources or raw_sources != chunk_sources:
        missing_count = len(raw_sources - chunk_sources)
        raise PersistentIndexError(
            "Every Markdown document must produce at least one chunk "
            f"(documents without chunks: {missing_count})."
        )

    staging_path.mkdir(parents=False, exist_ok=False)
    embedding_dimension: int | None = None
    client = _persistent_client(staging_path)
    try:
        collection = client.create_collection(
            DEFAULT_CHROMA_COLLECTION_NAME,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )
        for start in range(0, len(chunks), INDEX_WRITE_BATCH_SIZE):
            batch = chunks[start : start + INDEX_WRITE_BATCH_SIZE]
            vectors = embeddings.embed_documents(
                [document.page_content for document in batch]
            )
            if len(vectors) != len(batch):
                raise PersistentIndexError(
                    "Embedding provider returned a different number of vectors than chunks."
                )
            batch_dimension = _validate_vector_dimensions(vectors)
            if embedding_dimension is None:
                embedding_dimension = batch_dimension
            elif embedding_dimension != batch_dimension:
                raise PersistentIndexError(
                    "Embedding provider returned inconsistent vector dimensions."
                )
            collection.add(
                ids=[
                    stable_chunk_id(document, start + offset)
                    for offset, document in enumerate(batch)
                ],
                embeddings=vectors,
                documents=[document.page_content for document in batch],
                metadatas=[
                    _serializable_metadata(document.metadata) for document in batch
                ],
            )

        if embedding_dimension is None:
            raise PersistentIndexError(
                "Persistent index did not receive any embedding vectors."
            )
        _validate_staged_collection(
            client,
            collection,
            chunks,
            embeddings,
            embedding_dimension,
        )

        final_source_manifest = build_corpus_source_manifest(corpus_path)
        if list(source_manifest) != final_source_manifest:
            raise PersistentIndexError(
                "Corpus changed while the index was being built; retry the rebuild."
            )

        manifest = {
            **expected,
            "embedding_dimension": embedding_dimension,
            "raw_document_count": len(raw_documents),
            "chunk_count": len(chunks),
            "created_at_utc": datetime.now(UTC).isoformat(),
        }
        (staging_path / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    finally:
        client.close()

    staged_manifest = load_index_manifest(staging_path)
    if _manifest_differences(staged_manifest, expected):
        raise PersistentIndexError("Staged index manifest does not match the corpus.")
    _validate_persisted_collection(staging_path, staged_manifest)
    return staged_manifest


def _validate_staged_collection(
    client: Any,
    collection: Any,
    chunks: Sequence[Document],
    embeddings: Embeddings,
    embedding_dimension: int,
) -> None:
    collection_names = sorted(
        item if isinstance(item, str) else item.name for item in client.list_collections()
    )
    if collection_names != [DEFAULT_CHROMA_COLLECTION_NAME]:
        raise PersistentIndexError("Staged index contains unexpected Chroma collections.")
    if collection.count() != len(chunks):
        raise PersistentIndexError("Staged Chroma count does not match generated chunks.")

    expected_records = {
        stable_chunk_id(document, position): (
            document.page_content,
            _serializable_metadata(document.metadata),
        )
        for position, document in enumerate(chunks)
    }
    stored = collection.get(include=["documents", "metadatas"])
    stored_ids = stored.get("ids")
    stored_documents = stored.get("documents")
    stored_metadatas = stored.get("metadatas")
    if not all(
        isinstance(values, list)
        for values in (stored_ids, stored_documents, stored_metadatas)
    ):
        raise PersistentIndexError("Staged Chroma records have an invalid shape.")
    if not (
        len(stored_ids) == len(stored_documents) == len(stored_metadatas) == len(chunks)
    ):
        raise PersistentIndexError("Staged Chroma records are incomplete.")
    actual_records = {
        str(identifier): (document, metadata)
        for identifier, document, metadata in zip(
            stored_ids, stored_documents, stored_metadatas, strict=True
        )
    }
    if actual_records != expected_records:
        raise PersistentIndexError(
            "Staged Chroma content or metadata does not match generated chunks."
        )

    query_vector = embeddings.embed_query(INDEX_VALIDATION_QUERY)
    if len(query_vector) != embedding_dimension:
        raise PersistentIndexError(
            "Query embedding dimension does not match document embeddings."
        )
    response = collection.query(
        query_embeddings=[query_vector],
        n_results=1,
        include=["documents", "metadatas", "distances"],
    )
    documents = _first_query_batch(response.get("documents"))
    metadatas = _first_query_batch(response.get("metadatas"))
    distances = _first_query_batch(response.get("distances"))
    if (
        len(documents) != 1
        or not isinstance(documents[0], str)
        or len(metadatas) != 1
        or not isinstance(metadatas[0], Mapping)
        or len(distances) != 1
    ):
        raise PersistentIndexError("Staged index retrieval smoke test returned invalid data.")
    try:
        distance = float(distances[0])
    except (TypeError, ValueError) as error:
        raise PersistentIndexError(
            "Staged index retrieval smoke test returned an invalid distance."
        ) from error
    if not math.isfinite(distance):
        raise PersistentIndexError(
            "Staged index retrieval smoke test returned a non-finite distance."
        )


def _activate_staged_index(
    staging_path: Path,
    index_path: Path,
    corpus_path: Path,
    source_manifest: Sequence[Mapping[str, object]],
) -> None:
    backup_path = _backup_path(index_path)
    if backup_path.exists():
        raise PersistentIndexError(
            f"Cannot activate while an index backup remains: {backup_path}"
        )

    had_existing_index = index_path.exists()
    if had_existing_index:
        index_path.replace(backup_path)
    try:
        staging_path.replace(index_path)
        activated_manifest = load_index_manifest(index_path)
        _validate_persisted_collection(index_path, activated_manifest)
        if list(source_manifest) != build_corpus_source_manifest(corpus_path):
            raise PersistentIndexError(
                "Corpus changed while the staged index was being activated."
            )
    except Exception as activation_error:
        rollback_error: Exception | None = None
        try:
            if index_path.exists():
                index_path.replace(staging_path)
            if had_existing_index and backup_path.exists():
                backup_path.replace(index_path)
        except Exception as error:
            rollback_error = error
        if rollback_error is not None:
            raise PersistentIndexError(
                "Index activation and rollback both failed; the previous index remains "
                f"at {backup_path}: {rollback_error}"
            ) from activation_error
        raise PersistentIndexError(
            f"Index activation failed and the previous index was restored: {activation_error}"
        ) from activation_error

    if had_existing_index:
        _remove_directory_if_present(backup_path, warning_only=True)


def _recover_interrupted_index_swap(index_path: Path) -> None:
    backup_path = _backup_path(index_path)
    if not backup_path.exists():
        return

    try:
        backup_manifest = load_index_manifest(backup_path)
        _validate_persisted_collection(backup_path, backup_manifest)
    except Exception as error:
        raise PersistentIndexError(
            f"An interrupted-swap backup is present but invalid: {backup_path}"
        ) from error

    if not index_path.exists():
        backup_path.replace(index_path)
        LOGGER.warning("restored_previous_index_after_interrupted_swap")
        return

    try:
        active_manifest = load_index_manifest(index_path)
        _validate_persisted_collection(index_path, active_manifest)
    except Exception:
        failed_path = index_path.with_name(
            f".{index_path.name}.failed-{uuid4().hex}"
        )
        index_path.replace(failed_path)
        try:
            backup_path.replace(index_path)
        except Exception as error:
            raise PersistentIndexError(
                "Could not restore the previous index; retained paths are "
                f"{backup_path} and {failed_path}."
            ) from error
        LOGGER.warning(
            "restored_previous_index; invalid interrupted index retained at %s",
            failed_path,
        )
        return

    _remove_directory_if_present(backup_path, warning_only=False)
    LOGGER.warning("completed_cleanup_after_interrupted_index_swap")


def _backup_path(index_path: Path) -> Path:
    return index_path.with_name(f".{index_path.name}.backup")


def _staging_path(index_path: Path) -> Path:
    return index_path.with_name(f".{index_path.name}.staging-{uuid4().hex}")


def _remove_directory_if_present(path: Path, *, warning_only: bool) -> None:
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
    except OSError as error:
        if not warning_only:
            raise PersistentIndexError(f"Could not remove index directory {path}.") from error
        LOGGER.warning("could_not_remove_index_directory path=%s error_type=%s", path, type(error).__name__)


def _serializable_metadata(metadata: Mapping[str, object]) -> dict[str, str | int | float | bool]:
    serializable: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            serializable[str(key)] = value
    return serializable


def _validate_vector_dimensions(vectors: Sequence[Sequence[float]]) -> int:
    if not vectors or not vectors[0]:
        raise PersistentIndexError("Embedding provider returned an empty vector.")
    dimension = len(vectors[0])
    if any(len(vector) != dimension for vector in vectors):
        raise PersistentIndexError("Embedding provider returned inconsistent vector dimensions.")
    return dimension


def _first_query_batch(value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or not value:
        return []
    first = value[0]
    return first if isinstance(first, Sequence) else []


def _index_directory_has_files(index_dir: Path) -> bool:
    return index_dir.is_dir() and any(index_dir.iterdir())

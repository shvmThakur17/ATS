"""Pinecone storage for M11 embeddings. No embedding generation and no matching logic lives here.

Layout: one namespace per resume (`<prefix>:<document_id>`), vector id = the M10 chunk id.
Upserting the same namespace + id replaces the vector, so re-ingesting a resume never duplicates.
"""

import logging
import time
from math import isfinite

from ..embeddings.models import EmbeddingRecord
from .metadata import build_metadata
from .models import (FetchedVector, FetchError, IndexDimensionMismatchError, IndexMetricMismatchError,
                     InvalidVectorError, MissingConfigurationError, NamespaceError, PineconeConnectionError,
                     QueryError, QueryMatch, UpsertError, UpsertResult, VectorStoreConfig, VectorStoreError)

logger = logging.getLogger("app.parsing.vector_store")
MAX_NAMESPACE_LENGTH = 128  # Pinecone accepted 110 characters in a live probe
DEFAULT_BATCH_SIZE = 100


def wait_until(predicate, timeout: float = 30.0, interval: float = 1.0):
    """Poll `predicate` (Pinecone reads are eventually consistent); return its first truthy result."""
    deadline = time.monotonic() + timeout
    while True:
        result = predicate()
        if result or time.monotonic() >= deadline:
            return result
        time.sleep(interval)


def validate_namespace(namespace: str) -> str:
    if not isinstance(namespace, str) or not namespace or namespace != namespace.strip() \
            or any(c.isspace() for c in namespace) or len(namespace) > MAX_NAMESPACE_LENGTH:
        raise NamespaceError(f"invalid namespace {namespace!r}: must be a non-empty string without whitespace, "
                             f"at most {MAX_NAMESPACE_LENGTH} characters")
    return namespace


class PineconeStore:
    def __init__(self, config: VectorStoreConfig, client=None):
        self.config = config
        try:
            if client is None:
                from pinecone import Pinecone
                client = Pinecone(api_key=config.api_key)
        except Exception as error:
            raise PineconeConnectionError(f"could not create Pinecone client: {type(error).__name__}") from error
        self.client = client
        self._index = None
        self.dimension: int | None = None  # set by ensure_index()

    # --- naming ----------------------------------------------------------------------------
    def namespace_for(self, document_id: str) -> str:
        if not document_id:
            raise NamespaceError("document_id is empty")
        return validate_namespace(f"{self.config.namespace_prefix}:{document_id}")

    # --- index management ------------------------------------------------------------------
    def index_exists(self) -> bool:
        try:
            return bool(self.client.has_index(self.config.index_name))
        except Exception as error:
            raise PineconeConnectionError(f"could not check index {self.config.index_name!r}: "
                                          f"{type(error).__name__}: {error}") from error

    def describe_index(self):
        try:
            return self.client.describe_index(self.config.index_name)
        except Exception as error:
            raise PineconeConnectionError(f"could not describe index {self.config.index_name!r}: "
                                          f"{type(error).__name__}: {error}") from error

    def ensure_index(self, dimension: int, timeout: float = 300.0) -> None:
        """Create the index if missing, else verify it. A mismatching index is an error, never recreated."""
        if not isinstance(dimension, int) or dimension <= 0:
            raise InvalidVectorError(f"invalid dimension {dimension!r}")
        if self.index_exists():
            description = self.describe_index()
            if description.dimension != dimension:
                raise IndexDimensionMismatchError(
                    f"index {self.config.index_name!r} has dimension {description.dimension}, expected {dimension}")
            if description.metric != self.config.metric:
                raise IndexMetricMismatchError(
                    f"index {self.config.index_name!r} uses metric {description.metric!r}, expected {self.config.metric!r}")
        else:
            if not (self.config.cloud and self.config.region):
                raise MissingConfigurationError(
                    f"index {self.config.index_name!r} does not exist; set PINECONE_CLOUD and PINECONE_REGION to create it")
            try:
                from pinecone import ServerlessSpec
                self.client.create_index(name=self.config.index_name, dimension=dimension, metric=self.config.metric,
                                         spec=ServerlessSpec(cloud=self.config.cloud, region=self.config.region))
            except Exception as error:
                raise PineconeConnectionError(f"could not create index {self.config.index_name!r}: "
                                              f"{type(error).__name__}: {error}") from error
            logger.info("created index %s (dimension=%d, metric=%s)", self.config.index_name, dimension,
                        self.config.metric)
        if not wait_until(self._is_ready, timeout, 2.0):
            raise PineconeConnectionError(f"index {self.config.index_name!r} was not ready after {timeout}s")
        self.dimension = dimension

    def _is_ready(self) -> bool:
        status = self.describe_index().status
        return bool(status["ready"] if isinstance(status, dict) else status.ready)

    @property
    def index(self):
        if self._index is None:
            try:
                self._index = self.client.Index(self.config.index_name)
            except Exception as error:
                raise PineconeConnectionError(f"could not open index {self.config.index_name!r}: "
                                              f"{type(error).__name__}: {error}") from error
        return self._index

    def _require_dimension(self) -> int:
        if self.dimension is None:
            self.dimension = self.describe_index().dimension  # reuse the live index's size
        return self.dimension

    # --- validation ------------------------------------------------------------------------
    def check_vector(self, vector_id: str, values) -> None:
        if not isinstance(vector_id, str) or not vector_id:
            raise InvalidVectorError("vector id is missing")
        if not values:
            raise InvalidVectorError(f"{vector_id}: vector is empty")
        expected = self._require_dimension()
        if len(values) != expected:
            raise InvalidVectorError(f"{vector_id}: vector has {len(values)} values, index dimension is {expected}")
        if any(isinstance(v, (bool, str, bytes)) or not isinstance(v, (int, float)) or not isfinite(v) for v in values):
            raise InvalidVectorError(f"{vector_id}: vector contains non-numeric or non-finite values")

    # --- write -----------------------------------------------------------------------------
    def upsert_embeddings(self, records: list[EmbeddingRecord], batch_size: int = DEFAULT_BATCH_SIZE) -> UpsertResult:
        """Validate everything first (nothing is skipped), then upsert in batches into one resume namespace."""
        if not records:
            raise InvalidVectorError("no embeddings to upsert")
        document_ids = {r.metadata.get("document_id") for r in records}
        if len(document_ids) != 1:
            raise InvalidVectorError(f"records span {len(document_ids)} documents; upsert one resume at a time")
        namespace = self.namespace_for(document_ids.pop())
        ids = [r.chunk_id for r in records]
        if len(ids) != len(set(ids)):
            raise InvalidVectorError("duplicate chunk ids in one upsert")
        vectors = []
        for record in records:
            self.check_vector(record.chunk_id, record.embedding)
            vectors.append({"id": record.chunk_id, "values": list(record.embedding), "metadata": build_metadata(record)})
        upserted = 0
        for start in range(0, len(vectors), batch_size):
            batch = vectors[start:start + batch_size]
            try:
                self.index.upsert(vectors=batch, namespace=namespace)
            except Exception as error:
                raise UpsertError(f"upsert failed for namespace {namespace!r} after {upserted} of {len(vectors)} "
                                  f"vectors (batch starting {batch[0]['id']}): {type(error).__name__}: {error}") from error
            upserted += len(batch)
        logger.info("upserted index=%s namespace=%s chunks=%d upserted=%d failed=0",
                    self.config.index_name, namespace, len(records), upserted)
        return UpsertResult(namespace=namespace, upserted=upserted, failed=0, vector_ids=tuple(ids))

    # --- read ------------------------------------------------------------------------------
    def fetch(self, vector_ids: list[str], namespace: str) -> dict[str, FetchedVector]:
        validate_namespace(namespace)
        try:
            response = self.index.fetch(ids=list(vector_ids), namespace=namespace)
        except Exception as error:
            raise FetchError(f"fetch failed in namespace {namespace!r}: {type(error).__name__}: {error}") from error
        return {vid: FetchedVector(id=vid, values=list(v.values), metadata=dict(v.metadata or {}))
                for vid, v in response.vectors.items()}

    def query(self, vector: list[float], top_k: int, namespace: str, filter: dict | None = None) -> list[QueryMatch]:
        validate_namespace(namespace)
        self.check_vector("query", vector)
        if top_k < 1:
            raise QueryError(f"top_k must be >= 1, got {top_k}")
        try:
            arguments = dict(vector=list(vector), top_k=top_k, namespace=namespace, include_metadata=True)
            if filter:
                arguments["filter"] = filter
            response = self.index.query(**arguments)
        except Exception as error:
            raise QueryError(f"query failed in namespace {namespace!r}: {type(error).__name__}: {error}") from error
        matches = []
        for match in response.matches:
            metadata = dict(match.metadata or {})
            matches.append(QueryMatch(id=match.id, score=float(match.score), metadata=metadata,
                                      chunk_type=metadata.get("chunk_type"),
                                      canonical_path=metadata.get("canonical_path"),
                                      document_id=metadata.get("document_id")))
        return matches

    def namespace_vector_count(self, namespace: str) -> int:
        validate_namespace(namespace)
        try:
            stats = self.index.describe_index_stats()
        except Exception as error:
            raise NamespaceError(f"could not read index stats: {type(error).__name__}: {error}") from error
        entry = stats.namespaces.get(namespace)
        return entry.vector_count if entry else 0

    # --- delete ----------------------------------------------------------------------------
    def delete(self, vector_ids: list[str], namespace: str) -> None:
        validate_namespace(namespace)
        if not vector_ids:
            raise VectorStoreError("delete needs explicit vector ids")
        try:
            self.index.delete(ids=list(vector_ids), namespace=namespace)
        except Exception as error:
            raise NamespaceError(f"delete failed in namespace {namespace!r}: {type(error).__name__}: {error}") from error

    def delete_namespace(self, namespace: str) -> bool:
        """Remove every vector in ONE namespace. The default namespace and the index itself are never touched.

        Returns False if the namespace did not exist (Pinecone answers 404); any other failure raises."""
        validate_namespace(namespace)
        try:
            self.index.delete(delete_all=True, namespace=namespace)
        except Exception as error:
            if getattr(error, "status", None) == 404 or type(error).__name__ == "NotFoundError":
                logger.info("namespace=%s did not exist; nothing to delete", namespace)
                return False
            raise NamespaceError(f"could not delete namespace {namespace!r}: {type(error).__name__}: {error}") from error
        logger.info("deleted namespace=%s index=%s", namespace, self.config.index_name)
        return True

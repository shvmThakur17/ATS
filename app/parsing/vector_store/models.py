"""M12 vector-store configuration, results and errors."""

from dataclasses import dataclass, field
import os

from pydantic import BaseModel, ConfigDict

DEFAULT_NAMESPACE_PREFIX = "resume"


class VectorStoreError(Exception):
    """Base class for every M12 failure. Nothing is skipped or retried silently."""


class MissingConfigurationError(VectorStoreError):
    """PINECONE_API_KEY, PINECONE_INDEX_NAME (or cloud/region when creating an index) is missing."""


class IndexDimensionMismatchError(VectorStoreError):
    """The existing index has a different dimension than the embeddings."""


class IndexMetricMismatchError(VectorStoreError):
    """The existing index uses a different metric than required."""


class InvalidVectorError(VectorStoreError):
    """A vector or its metadata is invalid (id, dimension, non-finite value, bad metadata)."""


class PineconeConnectionError(VectorStoreError):
    """Pinecone could not be reached or the index could not be described/created."""


class UpsertError(VectorStoreError):
    pass


class FetchError(VectorStoreError):
    pass


class QueryError(VectorStoreError):
    pass


class NamespaceError(VectorStoreError):
    """Invalid namespace, or a refused delete (the default namespace is never deleted)."""


@dataclass(frozen=True)
class VectorStoreConfig:
    api_key: str = field(repr=False)           # never printed or logged
    index_name: str
    namespace_prefix: str = DEFAULT_NAMESPACE_PREFIX
    metric: str = "cosine"
    cloud: str | None = None                   # only needed if the index has to be created
    region: str | None = None

    @classmethod
    def from_env(cls) -> "VectorStoreConfig":
        api_key = os.environ.get("PINECONE_API_KEY")
        index_name = os.environ.get("PINECONE_INDEX_NAME")
        if not api_key:
            raise MissingConfigurationError("PINECONE_API_KEY is not set")
        if not index_name:
            raise MissingConfigurationError("PINECONE_INDEX_NAME is not set")
        return cls(api_key=api_key, index_name=index_name,
                   namespace_prefix=os.environ.get("PINECONE_NAMESPACE") or DEFAULT_NAMESPACE_PREFIX,
                   cloud=os.environ.get("PINECONE_CLOUD") or None,
                   region=os.environ.get("PINECONE_REGION") or None)


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class UpsertResult(FrozenModel):
    namespace: str
    upserted: int
    failed: int
    vector_ids: tuple[str, ...]


class FetchedVector(FrozenModel):
    id: str
    values: list[float]
    metadata: dict


class QueryMatch(FrozenModel):
    id: str
    score: float
    metadata: dict
    chunk_type: str | None = None
    canonical_path: str | None = None
    document_id: str | None = None

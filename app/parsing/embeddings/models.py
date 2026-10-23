"""M11 models, configuration and errors."""

from dataclasses import dataclass
import os

from pydantic import BaseModel, ConfigDict

from ..chunks.models import ChunkProvenance, MetadataValue


class EmbeddingError(Exception):
    """Base class. `chunk_id` names the chunk involved when there is one."""

    def __init__(self, message: str, chunk_id: str | None = None):
        super().__init__(f"[{chunk_id}] {message}" if chunk_id else message)
        self.chunk_id = chunk_id


class ModelUnavailableError(EmbeddingError):
    """The model could not be found or loaded (not cached locally, offline, no access)."""


class IncompatibleModelError(EmbeddingError):
    """The model loaded or ran but is not compatible with the installed libraries."""


class EmptyTextError(EmbeddingError):
    """A chunk/text to embed was empty."""


class InvalidEmbeddingError(EmbeddingError):
    """A vector is not a list of finite numbers."""


class EmptyEmbeddingError(InvalidEmbeddingError):
    """A vector has no values."""


class InconsistentDimensionError(InvalidEmbeddingError):
    """Vectors in a batch (or the model's documented size) disagree on dimension."""


class EmbeddingBatchError(EmbeddingError):
    """The backend failed on a batch."""


@dataclass(frozen=True)
class EmbeddingConfig:
    """All embedding settings in one place. Environment variables override the defaults."""
    model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    device: str = "cpu"                  # "cpu", "cuda" or "auto" (cuda if available, else cpu)
    batch_size: int = 8
    local_files_only: bool = True        # embedding never downloads; see NomicEmbedder docstring
    trust_remote_code: bool = False      # native nomic_bert in transformers>=5.5; the old remote code breaks on 5.x
    # Nomic v1.5 expects a task prefix. It is added inside the embedder, never to the chunk text.
    document_prefix: str = "search_document: "
    query_prefix: str = "search_query: "

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        default = cls()
        return cls(model_name=os.environ.get("NOMIC_EMBEDDING_MODEL", default.model_name),
                   device=os.environ.get("NOMIC_EMBEDDING_DEVICE", default.device),
                   batch_size=int(os.environ.get("NOMIC_EMBEDDING_BATCH_SIZE", default.batch_size)),
                   local_files_only=os.environ.get("NOMIC_EMBEDDING_LOCAL_ONLY", "1") != "0")


class EmbeddingRecord(BaseModel):
    """One vector, always traceable to its exact M10 chunk."""
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    chunk_type: str
    canonical_path: str
    source_record_id: str
    text: str                              # exactly the M10 chunk text
    embedding: list[float]
    dimension: int
    model_name: str
    metadata: dict[str, MetadataValue]
    provenance: ChunkProvenance

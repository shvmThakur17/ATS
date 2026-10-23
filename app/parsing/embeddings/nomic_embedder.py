"""Nomic embedding adapter: M10 chunks -> EmbeddingRecords.

Setup vs execution: this module never downloads a model. With the default
`local_files_only=True` the model must already be in the standard Hugging Face cache; to fetch it
once, run a one-off `SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")`
with network access (or set NOMIC_EMBEDDING_LOCAL_ONLY=0 for a single run). trust_remote_code stays off: transformers>=5.5 ships a native nomic_bert.
"""

from math import isfinite
from typing import Protocol, Sequence

from ..chunks.models import SemanticChunk
from .models import (EmbeddingBatchError, EmbeddingConfig, EmbeddingRecord, EmptyEmbeddingError,
                     EmptyTextError, InconsistentDimensionError, IncompatibleModelError,
                     InvalidEmbeddingError, ModelUnavailableError)


class EmbeddingBackend(Protocol):
    """The only thing the adapter needs from a model library."""
    dimension: int | None  # the model's documented output size, if it reports one

    def encode(self, texts: list[str]) -> Sequence[Sequence[float]]: ...


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def library_versions() -> str:
    versions = []
    for name in ("sentence_transformers", "transformers", "torch"):
        try:
            versions.append(f"{name}=={__import__(name).__version__}")
        except Exception:  # version info is only for the error message
            versions.append(f"{name}==?")
    return ", ".join(versions)


class SentenceTransformerBackend:
    """Loads the model once. Exceptions are classified, never swallowed."""

    def __init__(self, config: EmbeddingConfig, device: str):
        self.batch_size = config.batch_size
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(config.model_name, device=device,
                                             trust_remote_code=config.trust_remote_code,
                                             local_files_only=config.local_files_only)
        except OSError as error:
            raise ModelUnavailableError(
                f"model {config.model_name!r} is not available (local_files_only={config.local_files_only}): {error}") from error
        except Exception as error:
            raise IncompatibleModelError(
                f"model {config.model_name!r} failed to load with {library_versions()}: "
                f"{type(error).__name__}: {error}") from error
        self.dimension = self.model.get_embedding_dimension()

    def encode(self, texts: list[str]):
        return self.model.encode(texts, batch_size=len(texts), convert_to_numpy=True,
                                 show_progress_bar=False, normalize_embeddings=False).tolist()


class NomicEmbedder:
    """Reusable embedder: the model is loaded once (in the backend) and shared by every call."""

    def __init__(self, config: EmbeddingConfig | None = None, backend: EmbeddingBackend | None = None):
        self.config = config or EmbeddingConfig.from_env()
        self.device = resolve_device(self.config.device)
        self.backend = backend or SentenceTransformerBackend(self.config, self.device)
        self.model_name = self.config.model_name
        self.documented_dimension = self.backend.dimension
        self.dimension: int | None = None  # set from the first real vector

    # --- public API -------------------------------------------------------------------------
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(list(texts), [f"texts[{i}]" for i in range(len(texts))], self.config.document_prefix)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], ["query"], self.config.query_prefix)[0]

    def embed_chunks(self, chunks: list[SemanticChunk]) -> list[EmbeddingRecord]:
        """One record per chunk, in chunk order. chunk.text goes to the model unchanged."""
        vectors = self._embed([c.text for c in chunks], [c.chunk_id for c in chunks], self.config.document_prefix)
        return [EmbeddingRecord(chunk_id=c.chunk_id, chunk_type=c.chunk_type.value, canonical_path=c.canonical_path,
                                source_record_id=c.source_record_id, text=c.text, embedding=vector,
                                dimension=len(vector), model_name=self.model_name, metadata=c.metadata,
                                provenance=c.provenance)
                for c, vector in zip(chunks, vectors)]

    # --- internals --------------------------------------------------------------------------
    def _embed(self, texts: list[str], labels: list[str], prefix: str) -> list[list[float]]:
        for text, label in zip(texts, labels):
            if not text or not text.strip():
                raise EmptyTextError("text to embed is empty", label)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[start:start + self.config.batch_size]
            batch_labels = labels[start:start + self.config.batch_size]
            try:
                output = self.backend.encode([prefix + text for text in batch_texts])
            except (AttributeError, TypeError, ImportError) as error:
                raise IncompatibleModelError(
                    f"model {self.model_name!r} is incompatible with the installed libraries "
                    f"({library_versions()}): {type(error).__name__}: {error}", batch_labels[0]) from error
            except Exception as error:
                raise EmbeddingBatchError(
                    f"backend failed on batch {batch_labels}: {type(error).__name__}: {error}") from error
            if len(output) != len(batch_texts):
                raise EmbeddingBatchError(f"backend returned {len(output)} vectors for {len(batch_texts)} texts "
                                          f"in batch {batch_labels}")
            vectors.extend(self._checked(vector, label) for vector, label in zip(output, batch_labels))
        return vectors

    def _checked(self, vector, label: str) -> list[float]:
        if not hasattr(vector, "__len__") or isinstance(vector, (str, bytes)):
            raise InvalidEmbeddingError("embedding is not a list of numbers", label)
        if len(vector) == 0:
            raise EmptyEmbeddingError("embedding is empty", label)
        if any(isinstance(v, (str, bytes, bool)) or not hasattr(v, "__float__") for v in vector):
            raise InvalidEmbeddingError("embedding contains non-numeric values", label)
        values = [float(v) for v in vector]
        if not all(isfinite(v) for v in values):
            raise InvalidEmbeddingError("embedding contains non-finite values", label)
        expected = self.dimension or self.documented_dimension
        if expected is not None and len(values) != expected:
            raise InconsistentDimensionError(
                f"vector has {len(values)} values, expected {expected}", label)
        self.dimension = len(values)
        return values

"""EmbeddingRecord -> Pinecone metadata (flat str / number / bool / list[str] only)."""

import json
from math import isfinite

from ..canonical.models import SCHEMA_VERSION
from ..embeddings.models import EmbeddingRecord
from .models import InvalidVectorError

# Contact data is never written to the vector store.
PII_KEYS = frozenset({"email", "phone"})
MAX_METADATA_BYTES = 40_000  # Pinecone's per-vector metadata limit is 40 KB


def build_metadata(record: EmbeddingRecord) -> dict:
    """Metadata for one vector. M10 metadata is carried over (minus PII); `text` is the exact chunk text."""
    source = record.metadata
    document_id = source.get("document_id")
    if not isinstance(document_id, str) or not document_id:
        raise InvalidVectorError(f"{record.chunk_id}: chunk metadata has no document_id")
    provenance = record.provenance
    metadata = {k: v for k, v in source.items() if k not in PII_KEYS}
    metadata.update({
        "document_id": document_id,
        "source_file": source.get("source_file", ""),
        "chunk_id": record.chunk_id,
        "chunk_type": record.chunk_type,
        "canonical_path": record.canonical_path,
        "source_record_id": record.source_record_id,
        "text": record.text,
        "page_numbers": [str(n) for n in provenance.page_numbers],        # Pinecone lists must be strings
        "logical_block_ids": list(provenance.logical_block_ids),
        "source_block_ids": list(provenance.source_block_ids),
        "entity_ids": [e.entity_id for p in provenance.fields.values() for e in p.entities],
        "model_name": record.model_name,
        "embedding_dimension": record.dimension,
        "schema_version": SCHEMA_VERSION,
    })
    metadata = {k: [str(x) for x in v] if isinstance(v, list) else v for k, v in metadata.items()}
    validate_metadata(record.chunk_id, metadata)
    return metadata


def validate_metadata(vector_id: str, metadata: dict) -> None:
    for key, value in metadata.items():
        scalar = isinstance(value, (str, int, float, bool)) and not (isinstance(value, float) and not isfinite(value))
        string_list = isinstance(value, list) and all(isinstance(x, str) for x in value)
        if not (scalar or string_list):
            raise InvalidVectorError(f"{vector_id}: metadata field {key!r} has unsupported value {type(value).__name__}")
    size = len(json.dumps(metadata, ensure_ascii=False).encode("utf-8"))
    if size > MAX_METADATA_BYTES:
        raise InvalidVectorError(f"{vector_id}: metadata is {size} bytes, over the {MAX_METADATA_BYTES} byte limit")

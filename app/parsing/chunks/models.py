"""M10 semantic chunks. Text is for embedding; provenance and metadata stay separate from it."""

from enum import Enum

from pydantic import BaseModel, ConfigDict

from ..canonical.models import Provenance


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class ChunkType(str, Enum):
    SUMMARY = "SUMMARY"
    WORK_EXPERIENCE = "WORK_EXPERIENCE"
    PROJECT = "PROJECT"
    EDUCATION = "EDUCATION"
    SKILLS = "SKILLS"
    CERTIFICATION = "CERTIFICATION"
    AWARD = "AWARD"
    LANGUAGE = "LANGUAGE"
    INTEREST = "INTEREST"
    CONTACT = "CONTACT"


class ChunkProvenance(FrozenModel):
    """Everything here is carried over from M9; nothing is rebuilt from the PDF."""
    logical_block_ids: tuple[str, ...]
    page_numbers: tuple[int, ...]
    source_block_ids: tuple[str, ...]
    raw_values: dict[str, str]          # M9 provenance key -> raw (pre-normalization) value
    fields: dict[str, Provenance]       # M9 provenance key -> full M9 provenance


# Flat values only (no nulls, no nested objects). NOTE for M12: Pinecone lists must be strings.
MetadataValue = str | int | bool | list[int] | list[str]


class SemanticChunk(FrozenModel):
    chunk_id: str
    chunk_type: ChunkType
    text: str
    canonical_path: str
    source_record_id: str
    source_fields: tuple[str, ...]      # M9 field names that contributed to `text`
    provenance: ChunkProvenance
    metadata: dict[str, MetadataValue]

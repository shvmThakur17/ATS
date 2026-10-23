"""Rebuild the minimal M9 `CanonicalResume` that M13 needs from metadata already stored in Pinecone.

Why: M13 reads structured skills from `resume.skills` and `resume.projects[].technologies` (explicit skill
evidence) and from the resume's own skill terms (to avoid matching "Spring" inside "Spring Boot"). Pinecone holds
exactly that in the `skills` (SKILLS chunk) and `technologies` (PROJECT chunk) metadata written by M10/M12.

Nothing is inferred: skills and technologies are copied verbatim, in stored order. Descriptions, roles, dates,
education and summaries are NOT reconstructed (M13 still sees their text through the retrieved chunks), so the
experience/education output of M13 stays informational and partial. Nothing is written anywhere.
"""

import re
from math import isfinite

from ..parsing.canonical.models import (CanonicalCandidate, CanonicalMetadata, CanonicalProject, CanonicalResume,
                                        CanonicalSkill, CanonicalValidation, Provenance)
from ..parsing.normalization.models import ValidationStatus
from ..parsing.vector_store import PineconeStore, VectorStoreError
from .models import MatchingError

RECONSTRUCTED = "RECONSTRUCTED_FROM_VECTOR_METADATA"
METADATA_CHUNK_TYPES = ("SKILLS", "PROJECT")
MAX_PROJECT_INDEX = 10_000          # guards the gap filling below against a corrupt record index
PROJECT_PATH = re.compile(r"^projects\[(\d+)\]$")


class CanonicalReconstructionError(MatchingError):
    """The canonical resume could not be rebuilt safely. Nothing is guessed or fabricated."""


class NamespaceMismatchError(CanonicalReconstructionError):
    """The namespace does not belong to the supplied document id (M12 naming convention)."""


class MalformedSkillMetadataError(CanonicalReconstructionError):
    """Stored SKILLS/PROJECT metadata is missing, mistyped or inconsistent."""


class MetadataRetrievalError(CanonicalReconstructionError):
    """Pinecone could not return the metadata."""


def clean_list(value, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise MalformedSkillMetadataError(f"{label} must be a non-empty list of strings")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise MalformedSkillMetadataError(f"{label} must contain only non-empty strings")
    return list(value)


def as_index(value, label: str) -> int:
    """Pinecone may hand integers back as floats (3.0); accept only whole non-negative numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) \
            or value != int(value) or not 0 <= value <= MAX_PROJECT_INDEX:
        raise MalformedSkillMetadataError(f"{label} must be a whole number between 0 and {MAX_PROJECT_INDEX}")
    return int(value)


def skill(value: str, raw: str) -> CanonicalSkill:
    return CanonicalSkill(value=value, raw_value=raw,
                          provenance=Provenance(raw_value=raw, normalization_status=RECONSTRUCTED))


class PineconeCanonicalAdapter:
    """Read-only. One metadata query per resume, scoped to that resume's own namespace."""

    def __init__(self, store: PineconeStore):
        self.store = store

    def rebuild(self, document_id: str, namespace: str, vector_count: int) -> CanonicalResume:
        """`vector_count` is the number of vectors in `namespace`; it is the depth of the metadata query, so
        every SKILLS/PROJECT chunk is returned. A count of 0 gives an empty resume without querying."""
        self.check_namespace(document_id, namespace)
        if isinstance(vector_count, bool) or not isinstance(vector_count, int) or vector_count < 0:
            raise CanonicalReconstructionError(f"vector_count must be a non-negative integer, got {vector_count!r}")
        skills: list[CanonicalSkill] = []
        projects: list[CanonicalProject] = []
        source_file = ""
        if vector_count:
            matches = self.fetch_metadata(namespace, vector_count)
            skills, projects, source_file = self.parse(document_id, matches, vector_count)
        return CanonicalResume(
            candidate=CanonicalCandidate(), skills=skills, projects=projects,
            metadata=CanonicalMetadata(document_id=document_id, source_file_name=source_file, page_count=0,
                                       logical_block_count=0, extracted_entity_count=0,
                                       validation_status=ValidationStatus.VALID_WITH_WARNINGS),
            validation=CanonicalValidation(status=ValidationStatus.VALID_WITH_WARNINGS))

    # --- namespace --------------------------------------------------------------------------
    def check_namespace(self, document_id: str, namespace: str) -> None:
        if not isinstance(document_id, str) or not document_id:
            raise NamespaceMismatchError("document_id is empty")
        try:
            expected = self.store.namespace_for(document_id)
        except VectorStoreError as error:
            raise NamespaceMismatchError(f"document_id {document_id!r} has no valid namespace: {error}") from error
        if namespace != expected:
            raise NamespaceMismatchError(f"namespace {namespace!r} does not belong to document {document_id!r} "
                                         f"(expected {expected!r})")

    # --- read -------------------------------------------------------------------------------
    def fetch_metadata(self, namespace: str, depth: int):
        dimension = self.store.dimension
        if dimension is None:
            try:
                dimension = self.store.describe_index().dimension
            except VectorStoreError as error:
                raise MetadataRetrievalError(f"index dimension unavailable: {error}") from error
        probe = [1.0] + [0.0] * (dimension - 1)   # similarity is irrelevant: the filter selects, depth returns all
        try:
            return self.store.query(probe, depth, namespace, filter={"chunk_type": {"$in": list(METADATA_CHUNK_TYPES)}})
        except VectorStoreError as error:
            raise MetadataRetrievalError(f"could not read skill metadata from {namespace!r}: {error}") from error

    # --- parse ------------------------------------------------------------------------------
    def parse(self, document_id: str, matches, depth: int):
        if len(matches) > depth:
            raise MalformedSkillMetadataError(f"{len(matches)} records returned for a depth of {depth}")
        skills: list[CanonicalSkill] = []
        by_index: dict[int, CanonicalProject] = {}
        source_files, seen_skills = set(), 0
        for match in sorted(matches, key=lambda m: m.id):         # deterministic regardless of similarity order
            metadata = match.metadata or {}
            where = f"chunk {match.id!r}"
            if metadata.get("document_id") != document_id:
                raise NamespaceMismatchError(f"{where} belongs to document {metadata.get('document_id')!r}, "
                                             f"not {document_id!r}")
            if isinstance(metadata.get("source_file"), str):
                source_files.add(metadata["source_file"])
            kind, path = metadata.get("chunk_type"), metadata.get("canonical_path")
            if kind == "SKILLS":
                seen_skills += 1
                if seen_skills > 1:
                    raise MalformedSkillMetadataError(f"{where}: more than one SKILLS chunk in the namespace")
                skills = self.parse_skills(metadata, where)
            elif kind == "PROJECT":
                index, project = self.parse_project(metadata, path, where)
                if index in by_index:
                    raise MalformedSkillMetadataError(f"{where}: duplicate project index {index}")
                by_index[index] = project
            else:
                raise MalformedSkillMetadataError(f"{where}: unexpected chunk_type {kind!r}")
        gap = CanonicalProject(raw_text="", logical_block_ids=())   # keeps projects[j] aligned with the original paths
        projects = [by_index.get(i, gap) for i in range(max(by_index, default=-1) + 1)]
        source_file = next(iter(source_files)) if len(source_files) == 1 else ""
        return skills, projects, source_file

    def parse_skills(self, metadata: dict, where: str) -> list[CanonicalSkill]:
        if metadata.get("canonical_path") != "skills":
            raise MalformedSkillMetadataError(f"{where}: SKILLS chunk has canonical_path "
                                              f"{metadata.get('canonical_path')!r}")
        values = clean_list(metadata.get("skills"), f"{where}: skills")
        raw = metadata.get("raw_skills")
        raw = values if raw is None else clean_list(raw, f"{where}: raw_skills")
        if len(raw) != len(values):
            raise MalformedSkillMetadataError(f"{where}: skills ({len(values)}) and raw_skills ({len(raw)}) differ")
        text = metadata.get("text")
        if isinstance(text, str) and text != ", ".join(values):
            raise MalformedSkillMetadataError(f"{where}: skills metadata does not match the stored chunk text")
        return [skill(v, r) for v, r in zip(values, raw)]

    def parse_project(self, metadata: dict, path, where: str):
        match = PROJECT_PATH.match(path) if isinstance(path, str) else None
        if not match:
            raise MalformedSkillMetadataError(f"{where}: PROJECT chunk has canonical_path {path!r}")
        index = int(match.group(1))
        if index > MAX_PROJECT_INDEX:
            raise MalformedSkillMetadataError(f"{where}: project index {index} is out of range")
        if "record_index" in metadata and as_index(metadata["record_index"], f"{where}: record_index") != index:
            raise MalformedSkillMetadataError(f"{where}: record_index disagrees with canonical_path {path!r}")
        technologies = []
        if "technologies" in metadata:                      # absent = the project lists none (M10 omits empty lists)
            values = clean_list(metadata["technologies"], f"{where}: technologies")
            text = metadata.get("text")
            if isinstance(text, str) and "Technologies: " + ", ".join(values) not in text:
                raise MalformedSkillMetadataError(f"{where}: technologies metadata does not match the stored chunk text")
            technologies = [skill(v, v) for v in values]
        return index, CanonicalProject(technologies=technologies, raw_text="", logical_block_ids=())

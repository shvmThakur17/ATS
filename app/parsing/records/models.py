"""M7 structured records. Values are raw M6 text; nothing is normalized or inferred."""

from pydantic import BaseModel, ConfigDict

from ..entities.models import Entity, EntityType
from ..models import Evidence, SourceRef
from ..sections.models import SectionType


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class EntityRef(FrozenModel):
    """Lossless pointer back to the M6 entity a record field came from."""
    entity_id: str
    entity_type: EntityType
    raw_value: str
    logical_block_id: str
    section_type: SectionType
    page_number: int
    source_block_ids: tuple[str, ...]
    text_start: int
    text_end: int
    sources: tuple[SourceRef, ...]
    evidence: Evidence

    @classmethod
    def from_entity(cls, entity: Entity) -> "EntityRef":
        return cls(entity_id=entity.entity_id, entity_type=entity.entity_type,
                   raw_value=entity.raw_value, logical_block_id=entity.logical_block_id,
                   section_type=entity.section_type, page_number=entity.page_number,
                   source_block_ids=entity.source_block_ids, text_start=entity.text_start,
                   text_end=entity.text_end, sources=entity.sources, evidence=entity.evidence)


class Record(FrozenModel):
    """Common provenance. `entities` lists every member M6 entity in document order."""
    raw_text: str
    logical_block_ids: tuple[str, ...]
    entities: tuple[EntityRef, ...]


class ContactRecord(Record):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    linkedin: str | None = None
    github: str | None = None
    website: str | None = None


class WorkExperienceRecord(Record):
    company: str | None = None
    role: str | None = None
    date_range: str | None = None
    description: str | None = None


class EducationRecord(Record):
    institution: str | None = None
    degree: str | None = None
    field_of_study: str | None = None
    date_range: str | None = None


class ProjectRecord(Record):
    name: str | None = None
    description: str | None = None
    technology: tuple[str, ...] = ()


class CertificationRecord(Record):
    certification: str | None = None
    issuing_organization: str | None = None
    credential_id: str | None = None
    credential_url: str | None = None


class AwardRecord(Record):
    text: str


class SummaryRecord(Record):
    text: str


class SkillsRecord(Record):
    skills: tuple[str, ...]


class LanguageRecord(Record):
    languages: tuple[str, ...]


class InterestRecord(Record):
    interests: tuple[str, ...]


class ResumeRecords(FrozenModel):
    document_id: str
    contact: ContactRecord
    summary: SummaryRecord | None = None
    work_experience: tuple[WorkExperienceRecord, ...] = ()
    education: tuple[EducationRecord, ...] = ()
    projects: tuple[ProjectRecord, ...] = ()
    certifications: tuple[CertificationRecord, ...] = ()
    awards: tuple[AwardRecord, ...] = ()
    skills: SkillsRecord | None = None
    languages: LanguageRecord | None = None
    interests: InterestRecord | None = None
    # M6 entities no record claimed (UNKNOWN, DATE, surplus contact values).
    unassigned: tuple[EntityRef, ...] = ()

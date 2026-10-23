"""M9 canonical resume: the stable data contract downstream stages consume.

Conventions: a missing singular value is None, a missing collection is []. Every value keeps its
raw text and a Provenance back to the M6 entities it came from.
"""

from pydantic import BaseModel, ConfigDict

from ..normalization.models import ValidationIssue, ValidationStatus
from ..records.models import EntityRef

SCHEMA_VERSION = "1.0"


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class Provenance(FrozenModel):
    """canonical value -> M8 value -> M7 record -> M6 entities (id, block, page, PDF source refs)."""
    raw_value: str
    normalization_status: str
    normalization_rules: tuple[str, ...] = ()
    entities: tuple[EntityRef, ...] = ()


# field name -> provenance, only for fields that have an M8 value
FieldProvenance = dict[str, Provenance]


class CanonicalCandidate(FrozenModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    linkedin: str | None = None
    github: str | None = None
    website: str | None = None
    provenance: FieldProvenance = {}


class CanonicalSummary(FrozenModel):
    value: str | None
    raw_value: str
    provenance: Provenance


class CanonicalSkill(FrozenModel):
    value: str
    raw_value: str
    provenance: Provenance


class CanonicalLanguage(CanonicalSkill):
    pass


class CanonicalInterest(CanonicalSkill):
    pass


class CanonicalWorkExperience(FrozenModel):
    company: str | None = None
    role: str | None = None
    start_date: str | None = None  # "YYYY" or "YYYY-MM"
    end_date: str | None = None    # None while current
    current: bool = False
    description: str | None = None
    raw_text: str
    logical_block_ids: tuple[str, ...]
    provenance: FieldProvenance = {}


class CanonicalEducation(FrozenModel):
    institution: str | None = None
    degree: str | None = None
    field_of_study: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    raw_text: str
    logical_block_ids: tuple[str, ...]
    provenance: FieldProvenance = {}


class CanonicalProject(FrozenModel):
    name: str | None = None
    description: str | None = None
    technologies: list[CanonicalSkill] = []
    raw_text: str
    logical_block_ids: tuple[str, ...]
    provenance: FieldProvenance = {}


class CanonicalCertification(FrozenModel):
    name: str | None = None
    issuing_organization: str | None = None
    credential_id: str | None = None
    credential_url: str | None = None
    raw_text: str
    logical_block_ids: tuple[str, ...]
    provenance: FieldProvenance = {}


class CanonicalAward(FrozenModel):
    value: str | None
    raw_text: str
    logical_block_ids: tuple[str, ...]
    provenance: Provenance


class CanonicalMetadata(FrozenModel):
    document_id: str
    source_file_name: str
    page_count: int
    logical_block_count: int
    extracted_entity_count: int
    validation_status: ValidationStatus


class CanonicalValidation(FrozenModel):
    """M8's verdict carried forward unchanged, plus the entities M7 left unresolved."""
    status: ValidationStatus
    issues: list[ValidationIssue] = []
    unresolved: list[EntityRef] = []


class CanonicalResume(FrozenModel):
    schema_version: str = SCHEMA_VERSION
    candidate: CanonicalCandidate
    summary: CanonicalSummary | None = None
    skills: list[CanonicalSkill] = []
    work_experience: list[CanonicalWorkExperience] = []
    education: list[CanonicalEducation] = []
    projects: list[CanonicalProject] = []
    certifications: list[CanonicalCertification] = []
    awards: list[CanonicalAward] = []
    languages: list[CanonicalLanguage] = []
    interests: list[CanonicalInterest] = []
    metadata: CanonicalMetadata
    validation: CanonicalValidation

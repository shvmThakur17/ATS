"""M8 normalized models. They wrap M7 values; the raw value is never dropped."""

from enum import Enum

from pydantic import BaseModel, ConfigDict

from ..records.models import EntityRef


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class NormalizationStatus(str, Enum):
    NORMALIZED = "NORMALIZED"  # normalized_value differs from raw_value
    UNCHANGED = "UNCHANGED"    # normalized_value equals raw_value
    UNKNOWN = "UNKNOWN"        # could not be normalized safely; normalized_value is None
    INVALID = "INVALID"        # recognised but impossible/contradictory; normalized_value is None


class NormalizedValue(FrozenModel):
    """`warnings` are "code: message" strings. `sources` are the M7/M6 entity references."""
    raw_value: str
    normalized_value: str | None
    status: NormalizationStatus
    rules: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    sources: tuple[EntityRef, ...] = ()


class NormalizedDateRange(NormalizedValue):
    start_year: int | None = None
    start_month: int | None = None
    end_year: int | None = None
    end_month: int | None = None
    is_current: bool = False


class NormalizedRecord(FrozenModel):
    raw_text: str
    logical_block_ids: tuple[str, ...]


class NormalizedContact(NormalizedRecord):
    name: NormalizedValue | None = None
    email: NormalizedValue | None = None
    phone: NormalizedValue | None = None
    location: NormalizedValue | None = None
    linkedin: NormalizedValue | None = None
    github: NormalizedValue | None = None
    website: NormalizedValue | None = None


class NormalizedSummary(NormalizedRecord):
    text: NormalizedValue


class NormalizedWorkExperience(NormalizedRecord):
    company: NormalizedValue | None = None
    role: NormalizedValue | None = None
    date_range: NormalizedDateRange | None = None
    description: NormalizedValue | None = None


class NormalizedEducation(NormalizedRecord):
    institution: NormalizedValue | None = None
    degree: NormalizedValue | None = None
    field_of_study: NormalizedValue | None = None
    date_range: NormalizedDateRange | None = None


class NormalizedProject(NormalizedRecord):
    name: NormalizedValue | None = None
    description: NormalizedValue | None = None
    technology: tuple[NormalizedValue, ...] = ()


class NormalizedCertification(NormalizedRecord):
    certification: NormalizedValue | None = None
    issuing_organization: NormalizedValue | None = None
    credential_id: NormalizedValue | None = None
    credential_url: NormalizedValue | None = None


class NormalizedAward(NormalizedRecord):
    text: NormalizedValue


class NormalizedSkills(NormalizedRecord):
    skills: tuple[NormalizedValue, ...]


class NormalizedLanguages(NormalizedRecord):
    languages: tuple[NormalizedValue, ...]


class NormalizedInterests(NormalizedRecord):
    interests: tuple[NormalizedValue, ...]


class NormalizedResume(FrozenModel):
    document_id: str
    contact: NormalizedContact
    summary: NormalizedSummary | None = None
    work_experience: tuple[NormalizedWorkExperience, ...] = ()
    education: tuple[NormalizedEducation, ...] = ()
    projects: tuple[NormalizedProject, ...] = ()
    certifications: tuple[NormalizedCertification, ...] = ()
    awards: tuple[NormalizedAward, ...] = ()
    skills: NormalizedSkills | None = None
    languages: NormalizedLanguages | None = None
    interests: NormalizedInterests | None = None
    unassigned: tuple[EntityRef, ...] = ()  # M7 leftovers, passed through untouched


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class ValidationStatus(str, Enum):
    VALID = "VALID"
    VALID_WITH_WARNINGS = "VALID_WITH_WARNINGS"
    INVALID = "INVALID"


class ValidationIssue(FrozenModel):
    severity: Severity
    code: str
    message: str
    field: str | None = None
    record_type: str | None = None
    record_index: int | None = None
    source_entity_ids: tuple[str, ...] = ()


class NormalizationResult(FrozenModel):
    stage_version: str
    resume: NormalizedResume
    issues: tuple[ValidationIssue, ...]
    status: ValidationStatus

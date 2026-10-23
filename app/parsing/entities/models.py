"""M6 atomic entities. Offsets refer to the resolved M4 logical block text."""

from dataclasses import asdict, dataclass
from enum import Enum
import json

from ..models import Evidence, SourceRef
from ..sections.models import SectionType


class EntityType(str, Enum):
    NAME = "NAME"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    LOCATION = "LOCATION"
    LINKEDIN = "LINKEDIN"
    GITHUB = "GITHUB"
    WEBSITE = "WEBSITE"
    SKILL = "SKILL"
    ROLE = "ROLE"
    COMPANY = "COMPANY"
    DATE_RANGE = "DATE_RANGE"
    DATE = "DATE"
    DESCRIPTION = "DESCRIPTION"
    INSTITUTION = "INSTITUTION"
    DEGREE = "DEGREE"
    FIELD_OF_STUDY = "FIELD_OF_STUDY"
    PROJECT_NAME = "PROJECT_NAME"
    PROJECT_DESCRIPTION = "PROJECT_DESCRIPTION"
    TECHNOLOGY = "TECHNOLOGY"
    CERTIFICATION = "CERTIFICATION"
    ISSUING_ORGANIZATION = "ISSUING_ORGANIZATION"
    CREDENTIAL_ID = "CREDENTIAL_ID"
    CREDENTIAL_URL = "CREDENTIAL_URL"
    AWARD = "AWARD"
    LANGUAGE = "LANGUAGE"
    INTEREST = "INTEREST"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Entity:
    entity_id: str
    entity_type: EntityType
    raw_value: str
    logical_block_id: str
    page_number: int
    section_type: SectionType
    source_block_ids: tuple[str, ...]
    sources: tuple[SourceRef, ...]
    text_start: int
    text_end: int
    evidence: Evidence


@dataclass(frozen=True)
class EntityDocumentIR:
    document_id: str
    schema_version: str
    stage_version: str
    source_stage_version: str
    source_config_fingerprint: str
    entities: tuple[Entity, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False)

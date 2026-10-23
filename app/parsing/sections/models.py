"""M5 classifications reference M4 blocks without copying their content."""

from dataclasses import asdict, dataclass
from enum import Enum
import json

from ..models import Evidence


class SectionType(str, Enum):
    SUMMARY = "SUMMARY"
    SKILLS = "SKILLS"
    EXPERIENCE = "EXPERIENCE"
    PROJECTS = "PROJECTS"
    EDUCATION = "EDUCATION"
    CERTIFICATIONS = "CERTIFICATIONS"
    AWARDS = "AWARDS"
    LANGUAGES = "LANGUAGES"
    INTERESTS = "INTERESTS"
    CONTACT = "CONTACT"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class SectionBlock:
    logical_block_id: str
    page_number: int
    section_type: SectionType
    heading_block_id: str | None
    is_heading: bool
    evidence: Evidence


@dataclass(frozen=True)
class SectionDocumentIR:
    document_id: str
    schema_version: str
    source_stage_version: str
    source_config_fingerprint: str
    stage_version: str
    blocks: tuple[SectionBlock, ...]

    def to_json(self):
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False)

"""M4 immutable structural grouping contracts; deliberately no section taxonomy."""

from dataclasses import asdict, dataclass
from enum import Enum
import json

from ..models import Box, Evidence, Issue, SourceRef

LOGICAL_BLOCKS_VERSION = "logical-blocks/1"


class StructuralType(str, Enum):
    PARAGRAPH = "paragraph"
    HEADING_LIKE = "heading_like"
    BULLET_LIST = "bullet_list"
    NUMBERED_LIST = "numbered_list"
    LABEL_VALUE = "label_value"
    TEXT_GROUP = "text_group"
    UNKNOWN = "unknown"


class RelationshipKind(str, Enum):
    CONTINUATION = "CONTINUATION"
    PARAGRAPH_CONTINUATION = "PARAGRAPH_CONTINUATION"
    BULLET_ITEM = "BULLET_ITEM"
    LIST_CONTINUATION = "LIST_CONTINUATION"
    LABEL_VALUE = "LABEL_VALUE"
    DATE_RELATIONSHIP = "DATE_RELATIONSHIP"
    SAME_REGION = "SAME_REGION"
    SAME_STYLE = "SAME_STYLE"


@dataclass(frozen=True)
class LogicalBlock:
    logical_block_id: str
    page_number: int
    source_block_ids: tuple[str, ...]
    reading_order: tuple[int, ...]
    bounding_box: Box
    structural_type: StructuralType
    heading_like: bool
    sources: tuple[SourceRef, ...]
    evidence: Evidence


@dataclass(frozen=True)
class LogicalRelationship:
    first_block_id: str
    second_block_id: str
    relationship_kind: RelationshipKind
    evidence: Evidence


@dataclass(frozen=True)
class LogicalPage:
    page_number: int
    logical_blocks: tuple[LogicalBlock, ...]
    relationships: tuple[LogicalRelationship, ...]
    issues: tuple[Issue, ...]


@dataclass(frozen=True)
class LogicalDocumentIR:
    document_id: str
    schema_version: str
    stage_version: str
    source_layout_version: str
    source_reading_order_version: str
    config_fingerprint: str
    pages: tuple[LogicalPage, ...]
    source_issues: tuple[Issue, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False)

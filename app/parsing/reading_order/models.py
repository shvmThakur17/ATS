"""M3 immutable, semantic-neutral reading-order contracts."""

from dataclasses import asdict, dataclass
import json

from ..models import Evidence, Issue, SourceRef


READING_ORDER_VERSION = "reading-order/1"


@dataclass(frozen=True)
class OrderedBlock:
    """A reference to one M2 text block; text is deliberately not copied."""

    block_id: str
    region_ids: tuple[str, ...]
    sources: tuple[SourceRef, ...]
    evidence: Evidence


@dataclass(frozen=True)
class OrderTransition:
    before_block_id: str
    after_block_id: str
    evidence: Evidence


@dataclass(frozen=True)
class ReadingOrderPage:
    page_number: int
    ordered_block_ids: tuple[str, ...]
    blocks: tuple[OrderedBlock, ...]
    transitions: tuple[OrderTransition, ...]
    issues: tuple[Issue, ...]


@dataclass(frozen=True)
class ReadingOrderIR:
    document_id: str
    schema_version: str
    stage_version: str
    source_layout_version: str
    source_config_fingerprint: str
    config_fingerprint: str
    pages: tuple[ReadingOrderPage, ...]
    source_issues: tuple[Issue, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False)

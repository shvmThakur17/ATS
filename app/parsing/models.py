"""Immutable contracts. Geometry order is source order, never reading order."""

from dataclasses import dataclass
from enum import Enum
from math import isfinite


SCHEMA_VERSION = "resume-ir/1"


@dataclass(frozen=True)
class Box:
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self):
        if not all(isfinite(v) for v in (self.x0, self.y0, self.x1, self.y1)):
            raise ValueError("Non-finite geometry")
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("Inverted geometry")


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def __post_init__(self):
        if not isfinite(self.x) or not isfinite(self.y):
            raise ValueError("Non-finite point")


@dataclass(frozen=True)
class SourceRef:
    document_id: str
    page: int  # zero-based
    object_id: str
    box: Box
    char_start: int | None = None
    char_end: int | None = None  # exclusive, within a span
    method: str = "pdf-native"

    def __post_init__(self):
        if not self.document_id or not self.object_id or self.page < 0:
            raise ValueError("Invalid source identity")
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("Character offsets must be paired")
        if self.char_start is not None and not (0 <= self.char_start < self.char_end):
            raise ValueError("Invalid character interval")


@dataclass(frozen=True)
class Evidence:
    score: float
    rules: tuple[str, ...]
    reasons: tuple[str, ...]

    def __post_init__(self):
        if not isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("Confidence must be within [0, 1]")
        if not self.rules or not self.reasons:
            raise ValueError("Confidence needs rule IDs and reasons")


class ValueStatus(str, Enum):
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class FieldValue:
    raw: str
    normalized: str | None
    status: ValueStatus
    sources: tuple[SourceRef, ...]
    evidence: Evidence
    taxonomy_id: str | None = None
    taxonomy_version: str | None = None

    def __post_init__(self):
        if self.status == ValueStatus.KNOWN and (not self.raw or not self.sources):
            raise ValueError("Known fields require raw text and provenance")
        if self.raw and not self.sources:
            raise ValueError("Observed raw text requires provenance")
        if self.status == ValueStatus.UNKNOWN and self.normalized is not None:
            raise ValueError("Unknown fields cannot claim normalization")
        if (self.taxonomy_id is None) != (self.taxonomy_version is None):
            raise ValueError("Taxonomy identifier and version must be paired")


@dataclass(frozen=True)
class Issue:
    code: str
    severity: str
    message: str  # never include resume content in diagnostics
    sources: tuple[SourceRef, ...] = ()


@dataclass(frozen=True)
class Character:
    text: str
    box: Box
    origin: Point


@dataclass(frozen=True)
class TextStyle:
    font: str
    size: float
    flags: int  # PDF engine style flags, including bold/italic hints
    color: int
    alpha: int | None
    ascender: float | None
    descender: float | None


@dataclass(frozen=True)
class TextSpan:
    source: SourceRef
    source_block: int
    source_line: int
    source_span: int
    characters: tuple[Character, ...]
    quad: tuple[Point, Point, Point, Point]  # ul, ur, ll, lr
    style: TextStyle
    direction: Point
    writing_mode: int

    @property
    def text(self) -> str:
        """Exact span text only; never a document-level flattening."""
        return "".join(char.text for char in self.characters)


@dataclass(frozen=True)
class TextLine:
    source: SourceRef
    span_ids: tuple[str, ...]
    direction: Point
    writing_mode: int


@dataclass(frozen=True)
class TextBlock:
    source: SourceRef
    native_number: int
    lines: tuple[TextLine, ...]


@dataclass(frozen=True)
class ImageObject:
    source: SourceRef
    pixel_width: int
    pixel_height: int
    transform: tuple[float, ...]


@dataclass(frozen=True)
class DrawingObject:
    source: SourceRef
    path_type: str
    stroke_width: float
    segments: tuple[tuple[Point, Point], ...]


@dataclass(frozen=True)
class PageIR:
    number: int
    bounds: Box
    media_box: Box
    crop_box: Box
    rotation: int
    transformation: tuple[float, ...]
    rotation_matrix: tuple[float, ...]
    blocks: tuple[TextBlock, ...]
    spans: tuple[TextSpan, ...]
    images: tuple[ImageObject, ...]
    drawings: tuple[DrawingObject, ...]
    text_status: str  # NATIVE_TEXT, NEEDS_OCR, EMPTY_OR_OUTLINED


@dataclass(frozen=True)
class DocumentIR:
    document_id: str  # sha256 of original bytes
    schema_version: str
    stage_version: str
    engine_version: str
    config_fingerprint: str
    pages: tuple[PageIR, ...]
    issues: tuple[Issue, ...]


# Downstream stage contracts; algorithms are implemented in later increments.
@dataclass(frozen=True)
class Region:
    id: str
    kind: str  # main, sidebar, column, spanner, header, footer, table, UNKNOWN
    sources: tuple[SourceRef, ...]
    parent_id: str | None
    evidence: Evidence


@dataclass(frozen=True)
class OrderEdge:
    before: str
    after: str
    evidence: Evidence

    def __post_init__(self):
        if self.before == self.after:
            raise ValueError("Self-edge is not reading order")


@dataclass(frozen=True)
class OrderGraph:
    nodes: tuple[str, ...]
    edges: tuple[OrderEdge, ...]
    rejected_edges: tuple[OrderEdge, ...]
    issues: tuple[Issue, ...]
    stage_version: str


@dataclass(frozen=True)
class LogicalBlock:
    id: str
    kind: str
    region_id: str
    member_ids: tuple[str, ...]  # ordered only after graph validation
    sources: tuple[SourceRef, ...]
    evidence: Evidence
    table_row: int | None = None
    table_column: int | None = None
    row_span: int = 1
    column_span: int = 1


@dataclass(frozen=True)
class Section:
    id: str
    kind: str  # contact, summary, experience, education, skills, projects, certifications, UNKNOWN
    block_ids: tuple[str, ...]
    heading: FieldValue
    evidence: Evidence


@dataclass(frozen=True)
class Record:
    id: str
    kind: str
    fields: tuple[tuple[str, FieldValue], ...]
    items: tuple[tuple[str, tuple[FieldValue, ...]], ...]
    evidence: Evidence  # association confidence, distinct from field confidence


@dataclass(frozen=True)
class CanonicalResume:
    document_id: str
    schema_version: str
    stage_versions: tuple[tuple[str, str], ...]
    taxonomy_versions: tuple[tuple[str, str], ...]
    contact: tuple[tuple[str, tuple[FieldValue, ...]], ...]
    summary: tuple[FieldValue, ...]
    experience: tuple[Record, ...]
    education: tuple[Record, ...]
    skills: tuple[FieldValue, ...]
    projects: tuple[Record, ...]
    certifications: tuple[Record, ...]
    unassigned: tuple[FieldValue, ...]


@dataclass(frozen=True)
class ValidationReport:
    document_id: str
    stage_version: str
    accepted_record_ids: tuple[str, ...]
    issues: tuple[Issue, ...]


@dataclass(frozen=True)
class SemanticChunk:
    id: str
    document_id: str
    record_id: str
    text: str
    field_paths: tuple[str, ...]
    sources: tuple[SourceRef, ...]
    evidence: Evidence
    compiler_version: str

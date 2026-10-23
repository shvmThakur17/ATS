"""M2 contracts. Tuples are stored in ID order, not document reading order."""

from dataclasses import asdict, dataclass, fields
from enum import Enum
import json
from math import isfinite

from ..models import Box, Evidence, Issue, SourceRef

LAYOUT_VERSION = "layout/1"


class RegionKind(str, Enum):
    UNKNOWN = "UNKNOWN"
    MAIN = "MAIN"
    SIDEBAR = "SIDEBAR"
    HEADER = "HEADER"
    FOOTER = "FOOTER"
    FULL_WIDTH = "FULL_WIDTH"


class RelationKind(str, Enum):
    ABOVE = "ABOVE"
    BELOW = "BELOW"
    LEFT_OF = "LEFT_OF"
    RIGHT_OF = "RIGHT_OF"
    NEAR = "NEAR"
    OVERLAPS = "OVERLAPS"
    CONTAINS = "CONTAINS"
    WITHIN = "WITHIN"
    HORIZONTAL_OVERLAP = "HORIZONTAL_OVERLAP"
    VERTICAL_OVERLAP = "VERTICAL_OVERLAP"
    ALIGNED_LEFT = "ALIGNED_LEFT"
    ALIGNED_RIGHT = "ALIGNED_RIGHT"
    ALIGNED_CENTER = "ALIGNED_CENTER"
    ALIGNED_TOP = "ALIGNED_TOP"
    SIMILAR_WIDTH = "SIMILAR_WIDTH"


@dataclass(frozen=True)
class LayoutConfig:
    # All distances scale with page or median native line dimensions.
    gutter_page_ratio: float = 0.025
    gutter_line_factor: float = 1.0
    min_gutter_rows: int = 2
    row_tolerance_line_factor: float = 0.15
    alignment_page_ratio: float = 0.008
    proximity_page_ratio: float = 0.025
    proximity_line_factor: float = 2.5
    similar_width_ratio: float = 0.15
    margin_page_ratio: float = 0.08
    margin_isolation_line_factor: float = 1.0
    full_width_content_ratio: float = 0.75
    sidebar_content_ratio: float = 0.30
    sidebar_peer_ratio: float = 0.55
    max_direction_deviation: float = 0.10
    grid_page_ratio: float = 0.10
    min_table_rows: int = 3
    min_table_columns: int = 2
    min_grid_boundaries: int = 2
    score_unknown: float = 0.20
    score_region: float = 0.70
    score_gutter: float = 0.85
    score_margin: float = 0.60
    score_table: float = 0.55
    score_relationship: float = 1.0
    max_candidate_pairs: int = 250_000
    max_grid_entries: int = 500_000
    max_spatial_checks: int = 2_000_000
    max_segmentation_work: int = 2_000_000
    max_recursion_depth: int = 32

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isfinite(value) or value <= 0:
                raise ValueError(f"Invalid layout configuration: {field.name}")
            if isinstance(field.default, int) and (not isinstance(value, int)):
                raise ValueError(f"Integer required: {field.name}")
            if ("ratio" in field.name or field.name.startswith("score_")
                    or field.name == "max_direction_deviation") and value > 1:
                raise ValueError(f"Ratio outside (0, 1]: {field.name}")
        if self.margin_page_ratio >= 0.5:
            raise ValueError("Page margin bands must not overlap")


class LayoutLimitError(ValueError):
    """Explicit resource failure; partial layout must not look complete."""


@dataclass(frozen=True)
class BlockFeatures:
    source: SourceRef
    x0: float
    y0: float
    x1: float
    y1: float
    width: float
    height: float
    center_x: float
    center_y: float
    area: float
    page_width_ratio: float
    page_height_ratio: float

    @property
    def block_id(self):
        return self.source.object_id


@dataclass(frozen=True)
class BlockRelationship:
    first_id: str
    second_id: str
    # Every directional signal describes first relative to second.
    signals: tuple[RelationKind, ...]
    horizontal_overlap: float
    vertical_overlap: float
    gap_x: float
    gap_y: float
    center_distance: float
    evidence: Evidence


@dataclass(frozen=True)
class LayoutRegion:
    region_id: str
    page_number: int
    bounding_box: Box
    block_ids: tuple[str, ...]
    line_ids: tuple[str, ...]
    region_kind: RegionKind
    evidence: Evidence
    sources: tuple[SourceRef, ...]

    @property
    def confidence(self):
        return self.evidence.score


@dataclass(frozen=True)
class BlockAssignment:
    block_id: str
    region_ids: tuple[str, ...]
    spans_regions: bool


@dataclass(frozen=True)
class TableCandidate:
    candidate_id: str
    bounding_box: Box
    block_ids: tuple[str, ...]
    drawing_ids: tuple[str, ...]
    evidence: Evidence
    possible_table: bool = True


@dataclass(frozen=True)
class LayoutPage:
    page_number: int
    bounds: Box
    block_features: tuple[BlockFeatures, ...]
    regions: tuple[LayoutRegion, ...]
    assignments: tuple[BlockAssignment, ...]
    relationships: tuple[BlockRelationship, ...]
    table_candidates: tuple[TableCandidate, ...]
    image_ids: tuple[str, ...]
    drawing_ids: tuple[str, ...]
    issues: tuple[Issue, ...]


@dataclass(frozen=True)
class LayoutIR:
    document_id: str
    schema_version: str
    stage_version: str
    source_schema_version: str
    source_stage_version: str
    source_engine_version: str
    source_config_fingerprint: str
    config_fingerprint: str
    pages: tuple[LayoutPage, ...]
    source_issues: tuple[Issue, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False)

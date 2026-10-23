"""M1 DocumentIR -> M2 LayoutIR. No I/O, text flattening or semantic parsing."""

from dataclasses import asdict
from hashlib import sha256
import json

from ..models import DocumentIR, Issue, SCHEMA_VERSION
from ..pdf_objects import ExtractionConfig
from .block_features import block_features, line_scale
from .geometry import contains
from .models import (LAYOUT_VERSION, BlockAssignment, LayoutConfig, LayoutIR,
                     LayoutLimitError, LayoutPage)
from .region_detector import WorkBudget, detect_regions
from .relationship_detector import detect_relationships
from .table_detector import detect_tables


def _validate_source(document, limits):
    if document.schema_version != SCHEMA_VERSION:
        raise ValueError("Unsupported M1 schema version")
    if len(document.pages) > limits.max_pages:
        raise LayoutLimitError("M1 page limit exceeded")
    seen, page_numbers = set(), set()
    objects = characters = 0
    for page in document.pages:
        if page.number in page_numbers or page.number < 0:
            raise ValueError("Duplicate or invalid M1 page number")
        page_numbers.add(page.number)
        sources = [b.source for b in page.blocks]
        sources += [line.source for block in page.blocks for line in block.lines]
        sources += [s.source for s in page.spans]
        sources += [i.source for i in page.images] + [d.source for d in page.drawings]
        objects += len(sources)
        characters += sum(len(span.characters) for span in page.spans)
        if objects > limits.max_objects or characters > limits.max_characters:
            raise LayoutLimitError("M1 object or character limit exceeded")
        for source in sources:
            if (source.document_id != document.document_id or source.page != page.number
                    or source.object_id in seen):
                raise ValueError("Invalid or duplicate M1 source reference")
            seen.add(source.object_id)
        spans = {s.source.object_id for s in page.spans}
        for block in page.blocks:
            for line in block.lines:
                if any(span_id not in spans for span_id in line.span_ids):
                    raise ValueError("Dangling M1 span reference")


def analyze_layout(document: DocumentIR, config: LayoutConfig | None = None,
                   source_limits: ExtractionConfig | None = None) -> LayoutIR:
    """Analyze each page independently; source limits default to M1 limits.

    Tuple sorting is canonical storage order, not a reading-order prediction.
    Retain the input DocumentIR to resolve referenced spans/images/drawings.
    """
    config = config or LayoutConfig()
    limits = source_limits or ExtractionConfig()
    _validate_source(document, limits)
    fingerprint = sha256(json.dumps({"layout": asdict(config), "source_limits": asdict(limits)},
                                   sort_keys=True).encode()).hexdigest()
    pages = []
    budget = WorkBudget(config.max_segmentation_work)
    for page in sorted(document.pages, key=lambda p: p.number):
        features = block_features(page)
        scale = line_scale(page, config)
        regions, issues, atoms = detect_regions(page, config, scale, budget)
        memberships = {feature.block_id: [] for feature in features}
        for region in regions:
            for block_id in region.block_ids:
                memberships[block_id].append(region.region_id)
        assignments = tuple(BlockAssignment(key, tuple(sorted(value)), len(value) > 1)
                            for key, value in sorted(memberships.items()))
        if any(not assignment.region_ids for assignment in assignments):
            raise ValueError("M2 invariant violated: unassigned block")
        extra = []
        for block in page.blocks:
            if any(not contains(block.source.box, line.source.box) for line in block.lines):
                extra.append(Issue("M1_BLOCK_LINE_BOUNDS_MISMATCH", "warning",
                                   "Native line extends beyond its block; both geometries retained", (block.source,)))
        if not page.blocks:
            extra.append(Issue("NO_TEXT_LAYOUT", "info",
                               "No text blocks; image/drawing references retained for later processing"))
        pages.append(LayoutPage(
            page.number, page.bounds, features, regions, assignments,
            detect_relationships(features, page.bounds, scale, config),
            detect_tables(page, atoms, config, scale, budget),
            tuple(sorted(image.source.object_id for image in page.images)),
            tuple(sorted(drawing.source.object_id for drawing in page.drawings)),
            tuple(sorted(issues + tuple(extra), key=lambda i: (i.code, tuple(s.object_id for s in i.sources)))),
        ))
    return LayoutIR(document.document_id, "layout-ir/1", LAYOUT_VERSION,
                    document.schema_version, document.stage_version, document.engine_version,
                    document.config_fingerprint, fingerprint, tuple(pages), document.issues)


def format_diagnostics(layout: LayoutIR) -> str:
    """Only IDs, counts, geometry and issue codes; never native resume text."""
    output = []
    for page in layout.pages:
        output.extend((f"PAGE {page.page_number + 1}",
                       f"Blocks: {len(page.block_features)} Regions: {len(page.regions)}",
                       f"Relationships: {len(page.relationships)} Possible tables: {len(page.table_candidates)}"))
        for region in page.regions:
            output.extend((f"Region {region.region_id} {region.region_kind.value}",
                           f"  bounds: {region.bounding_box}",
                           f"  blocks: {', '.join(region.block_ids)}",
                           f"  confidence: {region.confidence}"))
        output.extend(f"Issue: {issue.code}" for issue in page.issues)
    return "\n".join(output)

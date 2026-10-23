"""M3: deterministic traversal over M2 physical regions; no semantic inference."""

from dataclasses import asdict, dataclass
from bisect import bisect_right
from hashlib import sha256
import json

from ..models import Evidence, Issue
from ..layout.models import LayoutIR, RegionKind
from .models import (READING_ORDER_VERSION, OrderTransition, OrderedBlock,
                     ReadingOrderIR, ReadingOrderPage)


@dataclass(frozen=True)
class ReadingOrderConfig:
    """Page-relative tolerances; M3 intentionally has no semantic options."""

    vertical_band_page_ratio: float = 0.006

    def __post_init__(self):
        if not 0 < self.vertical_band_page_ratio <= 1:
            raise ValueError("vertical_band_page_ratio must be in (0, 1]")


def _region_bands(regions, tolerance):
    """Return structural bands: full-width regions are explicit boundaries."""
    bands = []
    for region in sorted(regions, key=lambda r: (r.bounding_box.y0, r.bounding_box.y1,
                                                  r.bounding_box.x0, r.region_id)):
        full = region.region_kind in {RegionKind.HEADER, RegionKind.FOOTER, RegionKind.FULL_WIDTH}
        if full:
            bands.append([region])
            continue
        if bands and not any(r.region_kind in {RegionKind.HEADER, RegionKind.FOOTER, RegionKind.FULL_WIDTH}
                             for r in bands[-1]):
            end = max(r.bounding_box.y1 for r in bands[-1])
            if region.bounding_box.y0 <= end + tolerance:
                bands[-1].append(region)
                continue
        bands.append([region])
    continued_bands = []
    for band in bands:
        unknown_only = all(region.region_kind == RegionKind.UNKNOWN for region in band)
        supported_columns = continued_bands and any(
            "m2/persistent-gutter/1" in region.evidence.rules for region in continued_bands[-1])
        if unknown_only and supported_columns:
            continued_bands[-1].extend(band)
        else:
            continued_bands.append(list(band))
    return tuple(tuple(sorted(band, key=lambda r: (r.bounding_box.x0, r.bounding_box.y0, r.region_id)))
                 for band in continued_bands)


def _block_key(feature):
    return (feature.y0, feature.x0, feature.y1, feature.x1, feature.block_id)


def _band_block_ids(band, features):
    """Use established gutters even when UNKNOWN regions bridge their columns."""
    intervals = []
    for region in band:
        if "m2/persistent-gutter/1" in region.evidence.rules:
            box = region.bounding_box
            intervals.append((box.x0, box.x1))
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    cuts = [(first[1] + second[0]) / 2 for first, second in zip(merged, merged[1:])]
    if not cuts:
        return [block_id for region in band
                for block_id in sorted(region.block_ids, key=lambda key: _block_key(features[key]))]
    ids = {block_id for region in band for block_id in region.block_ids}
    line_positions = {}
    for region in band:
        for source in region.sources:
            if "/l" in source.object_id and "/s" not in source.object_id:
                block_id = source.object_id.split("/l", 1)[0]
                line_positions.setdefault(block_id, {})[source.object_id] = source.box.x0

    def lane_key(block_id):
        feature = features[block_id]
        # M2 retains native line references for blocks spanning regions. Use their
        # majority column as an anchor; the ambiguity remains explicit below.
        positions = sorted(line_positions.get(block_id, {}).values())
        anchor = positions[len(positions) // 2] if positions else feature.x0
        return (bisect_right(cuts, anchor), *_block_key(feature))

    return sorted(ids, key=lane_key)


def analyze_reading_order(layout: LayoutIR,
                          config: ReadingOrderConfig | None = None) -> ReadingOrderIR:
    """Derive per-page block order from M2 regions, never PDF object order.

    M2 only supports ordinary left-to-right horizontal text as known geometry.
    Thus regions sharing a vertical structural band traverse left-to-right; blocks
    within each region traverse top-to-bottom.  IDs are a last-resort stable tie
    breaker and cannot override either layout signal.
    """
    if layout.schema_version != "layout-ir/1" or layout.stage_version != "layout/1":
        raise ValueError("Unsupported M2 LayoutIR")
    config = config or ReadingOrderConfig()
    fingerprint = sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()
    pages = []
    for page in sorted(layout.pages, key=lambda value: value.page_number):
        features = {feature.block_id: feature for feature in page.block_features}
        assignments = {assignment.block_id: assignment for assignment in page.assignments}
        regions = {region.region_id: region for region in page.regions}
        if set(features) != set(assignments):
            raise ValueError("M2 assignments must cover every block feature")
        bands = _region_bands(page.regions, (page.bounds.y1 - page.bounds.y0) * config.vertical_band_page_ratio)
        placed, items, issues = set(), [], []
        for band in bands:
            for block_id in _band_block_ids(band, features):
                if block_id not in placed:
                    assignment = assignments[block_id]
                    region = max((regions[key] for key in assignment.region_ids),
                                 key=lambda value: (value.confidence, value.region_id))
                    source = features[block_id].source
                    evidence = Evidence(region.evidence.score, ("m3/region-vertical-order/1",),
                                        ("Region traversal follows structural vertical band and horizontal region position",))
                    items.append(OrderedBlock(block_id, assignment.region_ids, (source,), evidence))
                    placed.add(block_id)
                    if assignment.spans_regions:
                        issues.append(Issue("READING_ORDER_SPANNING_BLOCK", "warning",
                                            "Native block spans regions; one anchored position cannot express all line flows",
                                            (source,)))
        # Defensive coverage for malformed/unknown M2 references. Geometry still wins over IDs.
        for block_id in sorted(set(features) - placed, key=lambda value: _block_key(features[value])):
            assignment = assignments[block_id]
            items.append(OrderedBlock(block_id, assignment.region_ids, (features[block_id].source,),
                                      Evidence(0.20, ("m3/unassigned-region-fallback/1",),
                                               ("M2 region reference unavailable; ordered by native geometry",))))
            issues.append(Issue("READING_ORDER_REGION_REFERENCE_MISSING", "warning",
                                "Block retained using geometry fallback", (features[block_id].source,)))
        for previous, current in zip(items, items[1:]):
            a, b = features[previous.block_id], features[current.block_id]
            overlap = min(a.y1, b.y1) - max(a.y0, b.y0) > 0 and min(a.x1, b.x1) - max(a.x0, b.x0) > 0
            if overlap:
                issues.append(Issue("READING_ORDER_AMBIGUOUS_OVERLAP", "warning",
                                    "Overlapping blocks use deterministic geometry and ID tie-break evidence",
                                    (a.source, b.source)))
        transitions = tuple(OrderTransition(first.block_id, second.block_id,
                            Evidence(min(first.evidence.score, second.evidence.score),
                                     ("m3/structural-transition/1",),
                                     ("Same region vertical progression" if set(first.region_ids) & set(second.region_ids)
                                      else "Structural region transition",)))
                            for first, second in zip(items, items[1:]))
        pages.append(ReadingOrderPage(page.page_number, tuple(item.block_id for item in items),
                                      tuple(items), transitions,
                                      tuple(sorted(issues, key=lambda i: (i.code, tuple(s.object_id for s in i.sources))))) )
    return ReadingOrderIR(layout.document_id, "reading-order-ir/1", READING_ORDER_VERSION,
                          layout.stage_version, layout.config_fingerprint, fingerprint,
                          tuple(pages), layout.source_issues)


def format_diagnostics(order: ReadingOrderIR) -> str:
    """IDs and evidence only; never native resume text."""
    output = []
    for page in order.pages:
        output.extend((f"PAGE {page.page_number + 1}", "Reading Order:"))
        output.extend(f"{index:02d}  {block_id}" for index, block_id in enumerate(page.ordered_block_ids, 1))
        output.extend(f"Issue: {issue.code}" for issue in page.issues)
    return "\n".join(output)

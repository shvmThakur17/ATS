"""Local whitespace segmentation. No text interpretation or reading order."""

from bisect import bisect_right
from dataclasses import dataclass
from hashlib import sha256
import json

from ..models import Evidence, Issue, SourceRef
from .geometry import contains, height, union, vertical_overlap, width
from .models import LayoutLimitError, LayoutRegion, RegionKind


@dataclass(frozen=True)
class Atom:
    # Native lines allow an M1 block to participate in more than one region.
    source: SourceRef
    block_id: str
    line_id: str | None
    supported: bool

    @property
    def box(self):
        return self.source.box


class WorkBudget:
    def __init__(self, limit):
        self.remaining = limit

    def spend(self, count):
        self.remaining -= count
        if self.remaining < 0:
            raise LayoutLimitError("Layout segmentation work limit exceeded")


def atoms_for_page(page, config):
    atoms = []
    for block in sorted(page.blocks, key=lambda b: b.source.object_id):
        if not block.lines:
            atoms.append(Atom(block.source, block.source.object_id, None, True))
        for line in sorted(block.lines, key=lambda line: line.source.object_id):
            supported = (abs(line.direction.y) <= config.max_direction_deviation
                         and line.direction.x > 0 and line.writing_mode == 0)
            atoms.append(Atom(line.source, block.source.object_id, line.source.object_id, supported))
    return tuple(atoms)


def row_bands(atoms, tolerance):
    """Connected y intervals, used as spatial samples, never ordered output."""
    groups, end = [], None
    for atom in sorted(atoms, key=lambda a: (a.box.y0, a.box.y1, a.source.object_id)):
        if end is None or atom.box.y0 > end + tolerance:
            groups.append([])
            end = atom.box.y1
        else:
            end = max(end, atom.box.y1)
        groups[-1].append(atom)
    return groups


def x_intervals(atoms, min_gap):
    intervals = []
    for atom in sorted(atoms, key=lambda a: (a.box.x0, a.box.x1, a.source.object_id)):
        if not intervals or atom.box.x0 - intervals[-1][1] >= min_gap:
            intervals.append([atom.box.x0, atom.box.x1])
        else:
            intervals[-1][1] = max(intervals[-1][1], atom.box.x1)
    return intervals


def find_gutters(atoms, page_width, scale, config, budget):
    gap = max(page_width * config.gutter_page_ratio, scale * config.gutter_line_factor)
    events = {}
    rows = row_bands(atoms, scale * config.row_tolerance_line_factor)
    budget.spend(len(atoms))
    for row in rows:
        intervals = x_intervals(row, gap)
        for first, second in zip(intervals, intervals[1:]):
            events[first[1]] = events.get(first[1], 0) + 1
            events[second[0]] = events.get(second[0], 0) - 1
    # Persistent row whitespace: sweep support rather than compare every pair.
    runs, start, count = [], None, 0
    for x in sorted(events):
        count += events[x]
        if count >= config.min_gutter_rows and start is None:
            start = x
        if count < config.min_gutter_rows and start is not None:
            if x - start >= gap:
                runs.append((start, x))
            start = None
    # Staggered baselines can lack common rows. A globally empty gutter with
    # repeated objects on each side and overlapping vertical extent is evidence.
    intervals = x_intervals(atoms, gap)
    for left, right in zip(intervals, intervals[1:]):
        budget.spend(len(atoms))
        first = [a for a in atoms if a.box.x1 <= left[1]]
        second = [a for a in atoms if a.box.x0 >= right[0]]
        if min(len(first), len(second)) >= config.min_gutter_rows:
            if vertical_overlap(union(a.box for a in first), union(a.box for a in second)) >= scale:
                if not any(lo <= (left[1] + right[0]) / 2 <= hi for lo, hi in runs):
                    runs.append((left[1], right[0]))
    return tuple(sorted(runs))


def stable_id(prefix, payload):
    digest = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return f"{prefix}/{digest}"


def detect_regions(page, config, scale, budget):
    atoms = atoms_for_page(page, config)
    regions, issues = [], []
    block_sources = {b.source.object_id: b.source for b in page.blocks}

    def emit(members, kind, rule, score):
        if not members:
            return
        block_ids = tuple(sorted({a.block_id for a in members}))
        line_ids = tuple(sorted(a.line_id for a in members if a.line_id is not None))
        sources = {a.source.object_id: a.source for a in members}
        sources.update((key, block_sources[key]) for key in block_ids)
        box = union(a.box for a in members)
        region_id = stable_id(f"p{page.number}/region", [page.number, kind.value,
                              [a.source.object_id for a in sorted(members, key=lambda a: a.source.object_id)]])
        regions.append(LayoutRegion(region_id, page.number, box, block_ids, line_ids, kind,
                                    Evidence(score, (f"m2/{rule}/1",),
                                             ("Physical geometry candidate; no semantic classification",)),
                                    tuple(sources[key] for key in sorted(sources))))

    valid = []
    for atom in atoms:
        if (not atom.supported or width(atom.box) <= 0 or height(atom.box) <= 0
                or not contains(page.bounds, atom.box)):
            emit([atom], RegionKind.UNKNOWN, "unsupported-geometry", config.score_unknown)
            issues.append(Issue("LAYOUT_UNSUPPORTED_GEOMETRY", "warning",
                                "Object retained without axis-aligned layout inference", (atom.source,)))
        else:
            valid.append(atom)

    # Margin candidates require both a page-relative margin and body separation.
    for kind in (RegionKind.HEADER, RegionKind.FOOTER):
        if kind == RegionKind.HEADER:
            candidates = [a for a in valid if a.box.y1 <= page.bounds.y0 + height(page.bounds) * config.margin_page_ratio]
        else:
            candidates = [a for a in valid if a.box.y0 >= page.bounds.y1 - height(page.bounds) * config.margin_page_ratio]
        candidate_ids = {a.source.object_id for a in candidates}
        rest = [a for a in valid if a.source.object_id not in candidate_ids]
        if candidates and rest:
            margin, body = union(a.box for a in candidates), union(a.box for a in rest)
            separation = body.y0 - margin.y1 if kind == RegionKind.HEADER else margin.y0 - body.y1
            if separation >= scale * config.margin_isolation_line_factor:
                emit(candidates, kind, "isolated-margin", config.score_margin)
                valid = rest

    def segment(members, depth=0):
        if not members:
            return
        if depth > config.max_recursion_depth:
            raise LayoutLimitError("Layout segmentation depth limit exceeded")
        budget.spend(len(members))
        # Find repeated local row profiles before combining gutter evidence.
        # Otherwise a two-column top and three-column bottom can contribute
        # incompatible gaps that incorrectly merge into one page-wide gutter.
        min_gap = max(width(page.bounds) * config.gutter_page_ratio, scale * config.gutter_line_factor)
        runs = []
        previous = None
        for row in row_bands(members, scale * config.row_tolerance_line_factor):
            profile = x_intervals(row, min_gap)
            compatible = (previous is not None and len(profile) == len(previous)
                          and all(min(a[1], b[1]) > max(a[0], b[0]) for a, b in zip(profile, previous)))
            if not compatible:
                runs.append([])
            runs[-1].append(row)
            previous = profile
        stable_runs = [run for run in runs if len(run) >= config.min_gutter_rows]
        if len(runs) > 1 and len(stable_runs) >= 2:
            extent = union(a.box for a in members)
            for run in runs:
                group = [a for row in run for a in row]
                if (len(run) < config.min_gutter_rows
                        and all(width(a.box) / width(extent) >= config.full_width_content_ratio for a in group)):
                    emit(group, RegionKind.FULL_WIDTH, "local-spanning-band", config.score_gutter)
                else:
                    segment(group, depth + 1)
            return
        gutters = find_gutters(members, width(page.bounds), scale, config, budget)
        if not gutters:
            # No supported separator is evidence of one candidate region, not
            # proof of a known layout on sparse/disjoint objects.
            gap = max(width(page.bounds) * config.gutter_page_ratio, scale * config.gutter_line_factor)
            kind = RegionKind.MAIN if len(x_intervals(members, gap)) == 1 else RegionKind.UNKNOWN
            emit(members, kind, "unsplit-region", config.score_region if kind == RegionKind.MAIN else config.score_unknown)
            return
        cuts = [(lo + hi) / 2 for lo, hi in gutters]
        budget.spend(len(members) * len(cuts))
        crossing = [a for a in members if any(a.box.x0 < cut < a.box.x1 for cut in cuts)]
        if crossing:
            # Full-width interruptions define independent vertical zones, so a
            # page does not have to share one global set of columns.
            bands = row_bands(crossing, scale * config.row_tolerance_line_factor)
            band_boxes = [union(a.box for a in band) for band in bands]
            grouped = [[] for _ in bands]
            zones = [[] for _ in range(len(bands) + 1)]
            extent = union(a.box for a in members)
            for atom in members:
                budget.spend(len(bands))
                touched = [i for i, box in enumerate(band_boxes) if vertical_overlap(atom.box, box) > 0]
                if touched:
                    # Preserve the object once; ambiguity is explicit below.
                    grouped[touched[0]].append(atom)
                else:
                    index = bisect_right([box.y1 for box in band_boxes], atom.box.y0)
                    zones[index].append(atom)
            for group in grouped:
                if not group:
                    continue
                broad = all(width(a.box) / width(extent) >= config.full_width_content_ratio for a in group)
                emit(group, RegionKind.FULL_WIDTH if broad else RegionKind.UNKNOWN,
                     "gutter-spanning-band", config.score_gutter if broad else config.score_unknown)
                if not broad:
                    issues.append(Issue("LAYOUT_AMBIGUOUS_SPANNER", "warning",
                                        "Gutter crossing or overlapping band retained as UNKNOWN",
                                        tuple(a.source for a in group)))
            for zone in zones:
                segment(zone, depth + 1)
            return
        groups = [[] for _ in range(len(cuts) + 1)]
        for atom in members:
            groups[bisect_right(cuts, (atom.box.x0 + atom.box.x1) / 2)].append(atom)
        groups = [group for group in groups if group]
        widths = [width(union(a.box for a in group)) for group in groups]
        extent_width = width(union(a.box for a in members))
        for group, group_width in zip(groups, widths):
            sidebar = (group_width / extent_width <= config.sidebar_content_ratio
                       and group_width / max(widths) <= config.sidebar_peer_ratio)
            emit(group, RegionKind.SIDEBAR if sidebar else RegionKind.MAIN,
                 "persistent-gutter", config.score_gutter)

    segment(valid)
    return tuple(sorted(regions, key=lambda r: r.region_id)), tuple(issues), atoms

"""Table-like signals only. Does not infer cells, fields or table semantics."""

from ..models import Evidence
from .geometry import intersection, union, width
from .models import TableCandidate
from .region_detector import row_bands, stable_id, x_intervals


def detect_tables(page, atoms, config, scale, budget):
    result = []
    tolerance = width(page.bounds) * config.alignment_page_ratio

    # Connected drawing rectangles are hints only when both horizontal and
    # vertical boundaries exist. Group geometrically touching drawing paths.
    paths = sorted(page.drawings, key=lambda d: (d.source.box.x0, d.source.object_id))
    components = []
    for drawing in paths:
        touching = []
        for i, group in enumerate(components):
            budget.spend(len(group))
            box = union(d.source.box for d in group)
            other = drawing.source.box
            if (other.x0 <= box.x1 + tolerance and other.x1 >= box.x0 - tolerance
                    and other.y0 <= box.y1 + tolerance and other.y1 >= box.y0 - tolerance):
                touching.append(i)
        group = [drawing]
        for i in reversed(touching):
            group.extend(components.pop(i))
        components.append(group)
    for group in components:
        horizontal, vertical = set(), set()
        for drawing in group:
            for a, b in drawing.segments:
                budget.spend(1)
                if abs(a.y - b.y) <= tolerance and abs(a.x - b.x) > tolerance:
                    horizontal.add(round(((a.y + b.y) / 2) / tolerance))
                if abs(a.x - b.x) <= tolerance and abs(a.y - b.y) > tolerance:
                    vertical.add(round(((a.x + b.x) / 2) / tolerance))
        if min(len(horizontal), len(vertical)) < config.min_grid_boundaries:
            continue
        box = union(d.source.box for d in group)
        budget.spend(len(page.blocks))
        blocks = tuple(sorted(b.source.object_id for b in page.blocks if intersection(box, b.source.box)))
        drawing_ids = tuple(sorted(d.source.object_id for d in group))
        result.append(TableCandidate(stable_id(f"p{page.number}/table", [blocks, drawing_ids]),
                                     box, blocks, drawing_ids,
                                     Evidence(config.score_table, ("m2/orthogonal-boundaries/1",),
                                              ("Possible grid or bordered panel; not confirmed table",))))

    # Repeated aligned rows are a deliberately weak signal: text columns can
    # also meet it. Never reclassify a region solely on this signal.
    rows = row_bands([a for a in atoms if a.supported], scale * config.row_tolerance_line_factor)
    runs = []
    for row in rows:
        intervals = x_intervals(row, max(tolerance, scale * config.gutter_line_factor))
        budget.spend(len(row))
        if len(intervals) < config.min_table_columns:
            runs.append([])
            continue
        signature = tuple(interval[0] for interval in intervals)
        if (not runs or not runs[-1] or len(signature) != len(runs[-1][-1][0])
                or any(abs(a - b) > tolerance for a, b in zip(signature, runs[-1][-1][0]))):
            runs.append([])
        runs[-1].append((signature, row))
    for run in runs:
        if len(run) < config.min_table_rows:
            continue
        members = [atom for _, row in run for atom in row]
        ids = tuple(sorted({a.block_id for a in members}))
        result.append(TableCandidate(stable_id(f"p{page.number}/table",
                                              [ids, sorted(a.source.object_id for a in members), "aligned-rows"]),
                                     union(a.box for a in members), ids, (),
                                     Evidence(config.score_table, ("m2/repeated-row-alignment/1",),
                                              ("Repeated alignment may be columns or a table",))))
    return tuple(sorted(result, key=lambda c: c.candidate_id))

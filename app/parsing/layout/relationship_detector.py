"""Bounded spatial grid; only local relationships, no reading-order edges."""

from collections import defaultdict
from math import floor

from ..models import Evidence
from .geometry import (alignment, center_distance, contains, gaps, height,
                       horizontal_overlap, intersection, rectangle_distance,
                       relative_position, vertical_overlap, width)
from .models import BlockRelationship, LayoutLimitError, RelationKind


def candidate_pairs(features, bounds, radius, config):
    cell = max(width(bounds), height(bounds)) * config.grid_page_ratio
    grid = defaultdict(list)
    pairs = set()
    entries = 0
    checks = 0
    for index, feature in enumerate(features):
        box = feature.source.box
        x0, x1 = floor((box.x0 - radius) / cell), floor((box.x1 + radius) / cell)
        y0, y1 = floor((box.y0 - radius) / cell), floor((box.y1 + radius) / cell)
        entries += (x1 - x0 + 1) * (y1 - y0 + 1)
        if entries > config.max_grid_entries:
            raise LayoutLimitError("Layout spatial-grid entry limit exceeded")
        xs, ys = range(x0, x1 + 1), range(y0, y1 + 1)
        for x in xs:
            for y in ys:
                for other in grid[x, y]:
                    checks += 1
                    if checks > config.max_spatial_checks:
                        raise LayoutLimitError("Layout spatial comparison limit exceeded")
                    pairs.add((other, index))
                    if len(pairs) > config.max_candidate_pairs:
                        raise LayoutLimitError("Layout candidate-pair limit exceeded")
                grid[x, y].append(index)
    return sorted(pairs)


def detect_relationships(features, bounds, scale, config):
    radius = max(max(width(bounds), height(bounds)) * config.proximity_page_ratio,
                 scale * config.proximity_line_factor)
    tolerance = width(bounds) * config.alignment_page_ratio
    result = []
    for i, j in candidate_pairs(features, bounds, radius, config):
        first, second = features[i], features[j]
        a, b = first.source.box, second.source.box
        if rectangle_distance(a, b) > radius:
            continue
        signals = set(relative_position(a, b)) | set(alignment(a, b, tolerance))
        signals.add(RelationKind.NEAR)
        if intersection(a, b):
            signals.add(RelationKind.OVERLAPS)
        if contains(a, b):
            signals.add(RelationKind.CONTAINS)
        if contains(b, a):
            signals.add(RelationKind.WITHIN)
        if horizontal_overlap(a, b) > 0:
            signals.add(RelationKind.HORIZONTAL_OVERLAP)
        if vertical_overlap(a, b) > 0:
            signals.add(RelationKind.VERTICAL_OVERLAP)
        largest_width = max(width(a), width(b))
        if largest_width and abs(width(a) - width(b)) / largest_width <= config.similar_width_ratio:
            signals.add(RelationKind.SIMILAR_WIDTH)
        dx, dy = gaps(a, b)
        result.append(BlockRelationship(
            first.block_id, second.block_id, tuple(sorted(signals, key=lambda s: s.value)),
            horizontal_overlap(a, b), vertical_overlap(a, b), dx, dy, center_distance(a, b),
            Evidence(config.score_relationship, ("m2/local-geometry/1",),
                     ("Measured physical relations only; not semantic precedence",)),
        ))
    return tuple(result)

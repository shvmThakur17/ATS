"""Geometry only. Relative positions are not precedence or reading order."""

from math import hypot

from ..models import Box
from .models import RelationKind


def width(box: Box) -> float:
    return box.x1 - box.x0


def height(box: Box) -> float:
    return box.y1 - box.y0


def horizontal_overlap(a: Box, b: Box) -> float:
    return max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))


def vertical_overlap(a: Box, b: Box) -> float:
    return max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))


def intersection(a: Box, b: Box) -> Box | None:
    if horizontal_overlap(a, b) <= 0 or vertical_overlap(a, b) <= 0:
        return None
    return Box(max(a.x0, b.x0), max(a.y0, b.y0), min(a.x1, b.x1), min(a.y1, b.y1))


def contains(a: Box, b: Box, tolerance: float = 0.0) -> bool:
    return (a.x0 - tolerance <= b.x0 and a.y0 - tolerance <= b.y0
            and a.x1 + tolerance >= b.x1 and a.y1 + tolerance >= b.y1)


def gaps(a: Box, b: Box) -> tuple[float, float]:
    return max(0.0, a.x0 - b.x1, b.x0 - a.x1), max(0.0, a.y0 - b.y1, b.y0 - a.y1)


def rectangle_distance(a: Box, b: Box) -> float:
    return hypot(*gaps(a, b))


def center_distance(a: Box, b: Box) -> float:
    return hypot((a.x0 + a.x1 - b.x0 - b.x1) / 2, (a.y0 + a.y1 - b.y0 - b.y1) / 2)


def alignment(a: Box, b: Box, tolerance: float) -> tuple[RelationKind, ...]:
    tests = (
        (RelationKind.ALIGNED_LEFT, abs(a.x0 - b.x0)),
        (RelationKind.ALIGNED_RIGHT, abs(a.x1 - b.x1)),
        (RelationKind.ALIGNED_CENTER, abs((a.x0 + a.x1 - b.x0 - b.x1) / 2)),
        (RelationKind.ALIGNED_TOP, abs(a.y0 - b.y0)),
    )
    return tuple(kind for kind, delta in tests if delta <= tolerance)


def relative_position(a: Box, b: Box) -> tuple[RelationKind, ...]:
    result = []
    if a.y1 <= b.y0:
        result.append(RelationKind.ABOVE)
    elif a.y0 >= b.y1:
        result.append(RelationKind.BELOW)
    if a.x1 <= b.x0:
        result.append(RelationKind.LEFT_OF)
    elif a.x0 >= b.x1:
        result.append(RelationKind.RIGHT_OF)
    return tuple(result)


def union(boxes) -> Box:
    boxes = tuple(boxes)
    if not boxes:
        raise ValueError("Cannot bound empty geometry")
    return Box(min(b.x0 for b in boxes), min(b.y0 for b in boxes),
               max(b.x1 for b in boxes), max(b.y1 for b in boxes))

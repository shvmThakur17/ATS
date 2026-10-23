import unittest

from app.parsing.models import Box
from app.parsing.layout import RelationKind
from app.parsing.layout.geometry import (alignment, center_distance, contains, gaps,
                                        horizontal_overlap, intersection, rectangle_distance,
                                        relative_position, vertical_overlap)


class GeometryTests(unittest.TestCase):
    def test_overlap_containment_and_touching_edges(self):
        a, b = Box(0, 0, 20, 20), Box(10, 10, 30, 40)
        self.assertEqual(horizontal_overlap(a, b), 10)
        self.assertEqual(vertical_overlap(a, b), 10)
        self.assertEqual(intersection(a, b), Box(10, 10, 20, 20))
        self.assertTrue(contains(a, Box(2, 2, 3, 3)))
        self.assertFalse(contains(a, b))
        self.assertIsNone(intersection(a, Box(20, 0, 40, 20)))

    def test_distance_alignment_and_relative_position(self):
        a, b = Box(0, 0, 10, 10), Box(0, 20, 10, 30)
        self.assertEqual(gaps(a, b), (0, 10))
        self.assertEqual(rectangle_distance(a, b), 10)
        self.assertEqual(center_distance(a, b), 20)
        self.assertIn(RelationKind.ALIGNED_LEFT, alignment(a, b, 0.1))
        self.assertEqual(relative_position(a, b), (RelationKind.ABOVE,))

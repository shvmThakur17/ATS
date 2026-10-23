from dataclasses import asdict, replace
import unittest

from app.parsing.models import Box, DrawingObject, Point, SourceRef, TextBlock
from app.parsing.pdf_objects import ExtractionConfig
from app.parsing.layout import (LayoutConfig, LayoutLimitError, RegionKind, RelationKind,
                                analyze_layout, format_diagnostics)
from .fixtures import columns, document, memberships


class LayoutTests(unittest.TestCase):
    def analyze(self, boxes):
        return analyze_layout(document(boxes)).pages[0]

    def test_single_column(self):
        page = self.analyze(columns(((50, 550),)))
        self.assertEqual(len(page.regions), 1)
        self.assertEqual(page.regions[0].region_kind, RegionKind.MAIN)
        self.assertEqual(set(page.regions[0].block_ids), {"p0/b0", "p0/b1", "p0/b2"})

    def test_two_columns(self):
        page = self.analyze(columns())
        self.assertEqual(memberships(page), {frozenset(f"p0/b{i}" for i in range(3)),
                                            frozenset(f"p0/b{i}" for i in range(3, 6))})

    def test_unequal_columns(self):
        page = self.analyze(columns(((50, 220), (290, 550))))
        self.assertEqual(len(page.regions), 2)
        self.assertEqual(sorted(r.bounding_box.x1 - r.bounding_box.x0 for r in page.regions), [170, 260])

    def test_sidebar_can_be_on_either_side(self):
        for ranges, sidebar_x in ((((50, 150), (200, 550)), 50), (((50, 400), (450, 550)), 450)):
            with self.subTest(ranges=ranges):
                page = self.analyze(columns(ranges))
                sidebars = [r for r in page.regions if r.region_kind == RegionKind.SIDEBAR]
                self.assertEqual(len(sidebars), 1)
                self.assertEqual(sidebars[0].bounding_box.x0, sidebar_x)

    def test_full_width_header_columns_footer(self):
        page = self.analyze([(40, 20, 560, 40)] + columns() + [(40, 760, 560, 780)])
        self.assertEqual(len(page.regions), 4)
        self.assertEqual({r.region_kind for r in page.regions},
                         {RegionKind.HEADER, RegionKind.MAIN, RegionKind.FOOTER})

    def test_full_width_interruption(self):
        boxes = columns(ys=(140, 200)) + [(50, 300, 550, 320)] + columns(ys=(420, 480))
        page = self.analyze(boxes)
        self.assertEqual(len(page.regions), 5)
        self.assertEqual(sum(r.region_kind == RegionKind.FULL_WIDTH for r in page.regions), 1)

    def test_mixed_column_counts_on_same_page(self):
        boxes = (columns(ys=(140, 200)) + [(50, 300, 550, 320)]
                 + columns(((50, 180), (250, 350), (430, 550)), ys=(420, 480)))
        page = self.analyze(boxes)
        self.assertEqual(len(page.regions), 6)
        self.assertEqual(sum(r.region_kind == RegionKind.FULL_WIDTH for r in page.regions), 1)
        self.assertFalse(any(r.region_kind == RegionKind.UNKNOWN for r in page.regions))

    def test_four_columns_not_fixed_to_two(self):
        page = self.analyze(columns(((20, 130), (160, 270), (300, 410), (440, 560))))
        self.assertEqual(len(page.regions), 4)

    def test_column_count_changes_without_full_width_separator(self):
        boxes = (columns(ys=(140, 200))
                 + columns(((50, 180), (250, 350), (430, 550)), ys=(420, 480)))
        self.assertEqual(len(self.analyze(boxes).regions), 5)

    def test_single_column_then_two_columns(self):
        boxes = columns(((50, 550),), ys=(100, 160)) + columns(ys=(320, 380))
        self.assertEqual(len(self.analyze(boxes).regions), 3)

    def test_staggered_baselines_preserve_gutter(self):
        boxes = columns(((50, 250),), ys=(140, 220, 300)) + columns(((350, 550),), ys=(180, 260, 340))
        self.assertEqual(len(self.analyze(boxes).regions), 2)

    def test_overlaps_are_retained(self):
        page = self.analyze([(50, 100, 300, 150), (100, 120, 350, 170)])
        self.assertEqual(len(page.assignments), 2)
        self.assertTrue(all(a.region_ids for a in page.assignments))
        self.assertIn(RelationKind.OVERLAPS, page.relationships[0].signals)

    def test_directional_relationships_and_containment(self):
        page = self.analyze([(50, 100, 300, 120), (50, 135, 300, 155), (70, 103, 100, 115)])
        relations = {(r.first_id, r.second_id): r for r in page.relationships}
        self.assertIn(RelationKind.ABOVE, relations["p0/b0", "p0/b1"].signals)
        self.assertIn(RelationKind.ALIGNED_LEFT, relations["p0/b0", "p0/b1"].signals)
        self.assertIn(RelationKind.CONTAINS, relations["p0/b0", "p0/b2"].signals)

    def test_page_relative_scaling_is_not_a_fixed_x_cutoff(self):
        original = document(columns())
        scaled_boxes = [tuple(coord * 2 for coord in box) for box in columns()]
        scaled = document(scaled_boxes, width=1200, height=1600)
        self.assertEqual(memberships(analyze_layout(original).pages[0]), memberships(analyze_layout(scaled).pages[0]))

    def test_different_sizes_have_correct_features(self):
        page = self.analyze([(50, 100, 550, 120), (80, 150, 300, 210)])
        f = page.block_features[1]
        self.assertEqual((f.width, f.height, f.center_x, f.center_y, f.area), (220, 60, 190, 180, 13200))
        self.assertAlmostEqual(f.page_width_ratio, 220 / 600)
        self.assertAlmostEqual(f.page_height_ratio, 60 / 800)

    def test_each_page_has_independent_layout(self):
        result = analyze_layout(document(columns(((50, 550),)), columns()))
        self.assertEqual([len(p.regions) for p in result.pages], [1, 2])
        self.assertTrue(all(r.page_number == 1 for r in result.pages[1].regions))

    def test_deterministic_even_when_source_collections_are_permuted(self):
        source = document(columns())
        shuffled = replace(source, pages=(replace(source.pages[0], blocks=tuple(reversed(source.pages[0].blocks))),))
        self.assertEqual(analyze_layout(source).to_json(), analyze_layout(shuffled).to_json())
        self.assertEqual(analyze_layout(source).to_json(), analyze_layout(source).to_json())

    def test_provenance_and_input_are_preserved(self):
        source = document(columns())
        before = asdict(source)
        result = analyze_layout(source)
        self.assertEqual(asdict(source), before)
        for feature, block in zip(result.pages[0].block_features, source.pages[0].blocks):
            self.assertIs(feature.source, block.source)
        references = {s.object_id: s for r in result.pages[0].regions for s in r.sources}
        for block in source.pages[0].blocks:
            self.assertIs(references[block.source.object_id], block.source)
            self.assertIs(references[block.lines[0].source.object_id], block.lines[0].source)
        self.assertEqual(result.source_config_fingerprint, source.config_fingerprint)

    def test_no_semantics_or_reading_order_fields(self):
        serialized = analyze_layout(document(columns())).to_json().lower()
        for forbidden in ("skills", "experience", "projects", "education", "certifications", "reading_order"):
            self.assertNotIn(forbidden, serialized)
        self.assertTrue(all(r.region_kind in RegionKind for r in self.analyze(columns()).regions))

    def test_native_block_can_span_multiple_regions(self):
        source = document(columns())
        page = source.pages[0]
        combined = TextBlock(SourceRef("synthetic", 0, "p0/combined", Box(50, 140, 550, 300)), 0,
                             tuple(b.lines[0] for b in page.blocks))
        result = analyze_layout(replace(source, pages=(replace(page, blocks=(combined,)),))).pages[0]
        self.assertEqual(len(result.regions), 2)
        self.assertTrue(result.assignments[0].spans_regions)
        self.assertEqual(len(result.assignments[0].region_ids), 2)

    def test_table_signals_are_not_field_extraction(self):
        source = document(columns())
        corners = (Point(40, 100), Point(560, 100), Point(560, 340), Point(40, 340))
        drawing = DrawingObject(SourceRef("synthetic", 0, "p0/grid", Box(40, 100, 560, 340)), "s", 1,
                                tuple((corners[i], corners[(i + 1) % 4]) for i in range(4)))
        source = replace(source, pages=(replace(source.pages[0], drawings=(drawing,)),))
        page = analyze_layout(source).pages[0]
        self.assertTrue(page.table_candidates)
        self.assertTrue(all(candidate.possible_table for candidate in page.table_candidates))
        self.assertTrue(any("p0/grid" in candidate.drawing_ids for candidate in page.table_candidates))
        self.assertEqual(len(page.regions), 2)

    def test_blank_page_and_source_issues_preserved(self):
        source = document([])
        result = analyze_layout(source)
        self.assertEqual(result.pages[0].regions, ())
        self.assertEqual(result.pages[0].issues[0].code, "NO_TEXT_LAYOUT")
        self.assertEqual(result.source_issues, source.issues)

    def test_invalid_geometry_direction_is_unknown(self):
        source = document([(50, 100, 100, 200)])
        page = source.pages[0]
        block = page.blocks[0]
        block = replace(block, lines=(replace(block.lines[0], direction=Point(0, 1)),))
        source = replace(source, pages=(replace(page, blocks=(block,)),))
        self.assertEqual(analyze_layout(source).pages[0].regions[0].region_kind, RegionKind.UNKNOWN)

    def test_limits_and_configuration_are_validated(self):
        with self.assertRaises(ValueError):
            LayoutConfig(gutter_page_ratio=float("nan"))
        with self.assertRaises(ValueError):
            LayoutConfig(min_gutter_rows=1.5)
        with self.assertRaises(LayoutLimitError):
            analyze_layout(document(columns()), source_limits=ExtractionConfig(max_objects=1))
        with self.assertRaises(LayoutLimitError):
            analyze_layout(document(columns()), LayoutConfig(max_segmentation_work=1))
        with self.assertRaises(LayoutLimitError):
            analyze_layout(document([(50, 100, 300, 200)] * 4), LayoutConfig(max_candidate_pairs=1))

    def test_duplicate_source_ids_rejected(self):
        source = document(columns())
        page = source.pages[0]
        source = replace(source, pages=(replace(page, blocks=(page.blocks[0], page.blocks[0])),))
        with self.assertRaisesRegex(ValueError, "source reference"):
            analyze_layout(source)

    def test_diagnostics_only_geometry_and_identifiers(self):
        result = analyze_layout(document(columns()))
        output = format_diagnostics(result)
        self.assertIn("PAGE 1", output)
        self.assertIn("Regions: 2", output)
        self.assertIn("bounds:", output)
        self.assertNotIn("text:", output)


if __name__ == "__main__":
    unittest.main()

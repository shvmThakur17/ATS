from dataclasses import asdict, replace
import unittest

import fitz

from app.parsing.layout import analyze_layout
from app.parsing.pdf_objects import extract_pdf
from app.parsing.reading_order import analyze_reading_order, format_diagnostics
from tests.parsing.layout.fixtures import columns, document


class ReadingOrderTests(unittest.TestCase):
    def order(self, *pages):
        return analyze_reading_order(analyze_layout(document(*pages)))

    def ids(self, boxes):
        return self.order(boxes).pages[0].ordered_block_ids

    def test_single_column(self):
        self.assertEqual(self.ids(columns(((50, 550),))), ("p0/b0", "p0/b1", "p0/b2"))

    def test_two_unequal_and_sidebar_columns_are_physical(self):
        for ranges in (((50, 270), (330, 550)), ((50, 180), (250, 550)), ((50, 150), (200, 550))):
            with self.subTest(ranges=ranges):
                self.assertEqual(self.ids(columns(ranges)),
                                 ("p0/b0", "p0/b1", "p0/b2", "p0/b3", "p0/b4", "p0/b5"))

    def test_header_columns_footer_and_full_width_interruption(self):
        boxes = [(40, 20, 560, 40)] + columns(ys=(140, 200)) + [(50, 300, 550, 320)] + columns(ys=(420, 480)) + [(40, 760, 560, 780)]
        self.assertEqual(self.ids(boxes), tuple(f"p0/b{i}" for i in range(len(boxes))))

    def test_four_columns_and_staggered_starts(self):
        four = columns(((20, 130), (160, 270), (300, 410), (440, 560)), ys=(140, 210))
        self.assertEqual(self.ids(four), tuple(f"p0/b{i}" for i in range(8)))
        staggered = columns(((50, 250),), ys=(140, 220, 300)) + columns(((350, 550),), ys=(180, 260, 340))
        self.assertEqual(self.ids(staggered), tuple(f"p0/b{i}" for i in range(6)))

    def test_region_spanning_block_is_once_and_every_block_is_covered(self):
        source = document(columns())
        page = source.pages[0]
        from app.parsing.models import Box, SourceRef, TextBlock
        combined = TextBlock(SourceRef("synthetic", 0, "p0/combined", Box(50, 140, 550, 300)), 0,
                             tuple(block.lines[0] for block in page.blocks))
        result = analyze_reading_order(analyze_layout(replace(source, pages=(replace(page, blocks=(combined,)),))))
        ordered = result.pages[0].ordered_block_ids
        self.assertEqual(ordered, ("p0/combined",))
        self.assertEqual(len(ordered), len(set(ordered)))

    def test_overlaps_are_explicit_ambiguity_not_semantics(self):
        result = self.order([(50, 100, 300, 150), (100, 120, 350, 170)])
        self.assertEqual(result.pages[0].ordered_block_ids, ("p0/b0", "p0/b1"))
        self.assertIn("READING_ORDER_AMBIGUOUS_OVERLAP", [issue.code for issue in result.pages[0].issues])
        serialized = result.to_json().lower()
        for forbidden in ("skills", "experience", "projects", "education", "certifications"):
            self.assertNotIn(forbidden, serialized)

    def test_page_independence_provenance_and_determinism(self):
        source = document(columns(((50, 550),)), columns())
        before = asdict(source)
        layout = analyze_layout(source)
        result = analyze_reading_order(layout)
        self.assertEqual(asdict(source), before)
        self.assertEqual([page.page_number for page in result.pages], [0, 1])
        self.assertEqual(result.to_json(), analyze_reading_order(layout).to_json())
        self.assertEqual({item.block_id for item in result.pages[0].blocks},
                         {feature.block_id for feature in layout.pages[0].block_features})
        for item in result.pages[0].blocks:
            self.assertIs(item.sources[0], next(f.source for f in layout.pages[0].block_features if f.block_id == item.block_id))

    def test_real_pdf_uses_layout_not_pdf_insertion_order(self):
        with fitz.open() as pdf:
            page = pdf.new_page(width=600, height=800)
            # Right column is inserted first, deliberately reversing desired traversal.
            for x in (340, 50):
                for y in (140, 210, 280):
                    page.insert_text((x, y), "Synthetic", fontsize=10)
            data = pdf.tobytes()
        source = extract_pdf(data)
        layout = analyze_layout(source)
        result = analyze_reading_order(layout)
        native_x = {block.source.object_id: block.source.box.x0 for block in source.pages[0].blocks}
        ordered_x = [native_x[block_id] for block_id in result.pages[0].ordered_block_ids]
        self.assertTrue(all(x > 300 for x in ordered_x[3:]))
        self.assertTrue(all(x < 100 for x in ordered_x[:3]))
        self.assertEqual(result.to_json(), analyze_reading_order(analyze_layout(extract_pdf(data))).to_json())
        self.assertNotIn("Synthetic", format_diagnostics(result))


if __name__ == "__main__":
    unittest.main()

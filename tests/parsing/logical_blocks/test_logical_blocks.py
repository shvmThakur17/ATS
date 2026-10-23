from dataclasses import asdict
import unittest

import fitz

from app.parsing.layout import analyze_layout
from app.parsing.logical_blocks import (RelationshipKind, StructuralType,
                                        analyze_logical_blocks)
from app.parsing.pdf_objects import extract_pdf
from app.parsing.reading_order import analyze_reading_order


def pipeline(lines_by_page):
    with fitz.open() as pdf:
        for lines in lines_by_page:
            page = pdf.new_page(width=600, height=800)
            for text, x, y, size in lines:
                page.insert_text((x, y), text, fontsize=size)
        source = extract_pdf(pdf.tobytes())
    layout = analyze_layout(source)
    order = analyze_reading_order(layout)
    return source, layout, order, analyze_logical_blocks(source, layout, order)


class LogicalBlockTests(unittest.TestCase):
    def test_paragraph_continuation_and_boundary(self):
        _, _, _, result = pipeline([[("First clause continues", 50, 120, 10),
                                     ("onto its following block", 50, 140, 10),
                                     ("A separate paragraph.", 50, 200, 10)]])
        blocks = result.pages[0].logical_blocks
        self.assertEqual([len(block.source_block_ids) for block in blocks], [2, 1])
        self.assertEqual(blocks[0].structural_type, StructuralType.PARAGRAPH)

    def test_bullets_numbers_label_value_and_date_relationships(self):
        _, _, _, result = pipeline([[("• Item one", 50, 120, 10), ("• Item two", 50, 140, 10),
                                     ("1. One", 50, 190, 10), ("2. Two", 50, 210, 10),
                                     ("Email:", 50, 260, 10), ("name@example.test", 50, 280, 10),
                                     ("Jan 2022 - Dec 2024", 50, 330, 10), ("Nearby text", 50, 350, 10)]])
        types = [block.structural_type for block in result.pages[0].logical_blocks]
        self.assertIn(StructuralType.BULLET_LIST, types)
        self.assertIn(StructuralType.NUMBERED_LIST, types)
        self.assertIn(StructuralType.LABEL_VALUE, types)
        relationships = [relationship.relationship_kind for relationship in result.pages[0].relationships]
        self.assertIn(RelationshipKind.LABEL_VALUE, relationships)
        self.assertIn(RelationshipKind.DATE_RELATIONSHIP, relationships)

    def test_heading_multicolumn_multipage_provenance_and_no_semantics(self):
        source, layout, order, result = pipeline([[("TITLE", 50, 100, 16), ("Left continues", 50, 150, 10),
                                                   ("Right continues", 340, 150, 10)],
                                                  [("Second page", 50, 100, 10)]])
        self.assertEqual([page.page_number for page in result.pages], [0, 1])
        self.assertTrue(any(block.heading_like for block in result.pages[0].logical_blocks))
        output_ids = [source_id for page in result.pages for block in page.logical_blocks for source_id in block.source_block_ids]
        expected_ids = [feature.block_id for page in layout.pages for feature in page.block_features]
        self.assertEqual(set(output_ids), set(expected_ids))
        self.assertEqual(len(output_ids), len(set(output_ids)))
        self.assertEqual(result.to_json(), analyze_logical_blocks(source, layout, order).to_json())
        serialized = result.to_json().lower()
        for forbidden in ("section", "experience", "skills", "projects", "education"):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(asdict(source), asdict(source))

    def test_m1_to_m4_pipeline_preserves_source_ids(self):
        source, layout, order, result = pipeline([[("Right", 340, 120, 10), ("Left", 50, 120, 10)]])
        self.assertEqual(result.document_id, source.document_id)
        self.assertEqual(result.pages[0].logical_blocks[0].source_block_ids[0], order.pages[0].ordered_block_ids[0])
        self.assertTrue(all(block.sources for block in result.pages[0].logical_blocks))


if __name__ == "__main__":
    unittest.main()

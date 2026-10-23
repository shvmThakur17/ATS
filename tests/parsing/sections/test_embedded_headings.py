"""M5-only recovery of whole-line section aliases attached to prior content."""

from dataclasses import asdict
import unittest

import fitz

from app.parsing.sections import detect_sections, resolve_logical_text, SectionType
from app.parsing.pdf_objects import extract_pdf
from app.parsing.layout import analyze_layout
from app.parsing.reading_order import analyze_reading_order
from app.parsing.logical_blocks import analyze_logical_blocks
from tests.parsing.sections.test_sections import fixture


class EmbeddedHeadingTests(unittest.TestCase):
    def test_trailing_education_preserves_projects_and_continues(self):
        logical, text = fixture([
            ("PROJECTS", True, 50, 40),
            ("Built a service.", False, 50, 80),
            ("Maintained the service.\nEDUCATION", False, 50, 120),
            ("Diploma", True, 50, 160),
            ("Example Institute", False, 50, 200),
            ("Another qualification", False, 50, 240),
            ("AWARDS", True, 50, 280),
            ("Received an award.", False, 50, 320),
        ])
        before = asdict(logical)
        original_text = dict(text)
        result = detect_sections(logical, text)
        self.assertEqual([item.section_type for item in result.blocks],
                         [SectionType.PROJECTS] * 3 + [SectionType.EDUCATION] * 3
                         + [SectionType.AWARDS] * 2)
        self.assertFalse(result.blocks[2].is_heading)
        self.assertEqual(result.blocks[3].heading_block_id, logical.pages[0].logical_blocks[2].logical_block_id)
        self.assertIn("m5/trailing-heading-and-flow/1", result.blocks[2].evidence.rules)
        self.assertEqual(before, asdict(logical))
        self.assertEqual(text, original_text)
        self.assertEqual(result.to_json(), detect_sections(logical, dict(reversed(list(text.items())))).to_json())

    def test_normal_headings_and_small_award_alias_set(self):
        for heading, section in (("EDUCATION", SectionType.EDUCATION),
                                 ("ACHIVEMENT", SectionType.AWARDS),
                                 ("ACHIEVEMENT", SectionType.AWARDS),
                                 ("ACHIEVEMENTS", SectionType.AWARDS),
                                 ("AWARD", SectionType.AWARDS),
                                 ("AWARDS", SectionType.AWARDS)):
            with self.subTest(heading=heading):
                logical, text = fixture([(heading, True, 50, 100),
                                         ("Supporting content", False, 50, 140)])
                result = detect_sections(logical, text)
                self.assertEqual([item.section_type for item in result.blocks], [section, section])
                self.assertTrue(result.blocks[0].is_heading)

    def test_unknown_and_other_column_are_not_overridden(self):
        logical, text = fixture([
            ("Previous content\nEDUCATION", False, 50, 100),
            ("Content", False, 50, 140),
            ("Unrecognized Heading", True, 50, 180),
            ("Another Unrecognized Heading", True, 50, 220),
            ("Other column", False, 350, 140),
        ])
        result = detect_sections(logical, text)
        self.assertEqual(result.blocks[1].section_type, SectionType.EDUCATION)
        self.assertEqual([item.section_type for item in result.blocks[2:]], [SectionType.UNKNOWN] * 3)

    def test_sentence_and_unconfirmed_alias_remain_unknown(self):
        for value in ("Mentioned education in a sentence.",
                      "Previous text\nEDUCATION", "ACHIVEMENTS"):
            logical, text = fixture([(value, False, 50, 100)])
            self.assertEqual(detect_sections(logical, text).blocks[0].section_type, SectionType.UNKNOWN)

    def test_native_line_geometry_recovers_earlier_column_without_splitting(self):
        # PDF engines can combine same-baseline lines across columns into one block.
        # The education entries are visited before the mixed right-column block.
        with fitz.open() as pdf:
            page = pdf.new_page(width=600, height=800)
            page.insert_text((300, 100), "PROJECTS", fontsize=16)
            page.insert_text((300, 150), "Maintained a service.", fontsize=10)
            page.insert_text((300, 162), "Provided technical support.", fontsize=10)
            page.insert_text((30, 150), "EDUCATION", fontsize=16)
            page.insert_text((30, 210), "Diploma", fontsize=10, fontname="hebo")
            page.insert_text((30, 250), "Example Institute", fontsize=10)
            source = extract_pdf(pdf.tobytes())
        layout = analyze_layout(source)
        order = analyze_reading_order(layout)
        logical = analyze_logical_blocks(source, layout, order)
        text = resolve_logical_text(logical, source)
        mixed = [key for key, value in text.items() if value.endswith("EDUCATION") and "Maintained" in value]
        self.assertEqual(len(mixed), 1)
        before = asdict(logical)
        result = detect_sections(logical, text)
        by_text = {text[item.logical_block_id]: item for item in result.blocks}
        for value in ("Diploma", "Example Institute"):
            self.assertEqual(by_text[value].section_type, SectionType.EDUCATION)
            self.assertEqual(by_text[value].heading_block_id, mixed[0])
        self.assertEqual(next(item for item in result.blocks if item.logical_block_id == mixed[0]).section_type,
                         SectionType.PROJECTS)
        self.assertEqual(before, asdict(logical))
        self.assertEqual(len(result.blocks), sum(len(page.logical_blocks) for page in logical.pages))
        self.assertEqual(result.to_json(), detect_sections(logical, resolve_logical_text(logical, source)).to_json())


if __name__ == "__main__":
    unittest.main()

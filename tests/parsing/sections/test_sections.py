from dataclasses import asdict, replace
import unittest

import fitz

from app.parsing.models import Box, Evidence, SourceRef
from app.parsing.logical_blocks.models import LogicalBlock, LogicalPage, LogicalDocumentIR, StructuralType
from app.parsing.logical_blocks import analyze_logical_blocks
from app.parsing.layout import analyze_layout
from app.parsing.pdf_objects import extract_pdf
from app.parsing.reading_order import analyze_reading_order
from app.parsing.sections import SectionType, detect_sections, resolve_logical_text
from app.parsing.sections.section_rules import ALIASES


def fixture(*pages):
    """Entries are (text, heading_like, x, y), already in M4 reading order."""
    logical_pages, texts = [], {}
    for number, entries in enumerate(pages):
        blocks = []
        for index, (text, heading, x, y) in enumerate(entries):
            key = f"p{number}/lb{index}"
            box = Box(x, y, x + 180, y + 12)
            source = SourceRef("fixture", number, f"p{number}/b{index}", box)
            blocks.append(LogicalBlock(key, number, (source.object_id,), (index,), box,
                                       StructuralType.HEADING_LIKE if heading else StructuralType.UNKNOWN,
                                       heading, (source,), Evidence(.8, ("fixture",), ("Fixture",))))
            texts[key] = text
        logical_pages.append(LogicalPage(number, tuple(blocks), (), ()))
    logical = LogicalDocumentIR("fixture", "logical-document-ir/1", "logical-blocks/1",
                                "layout/1", "reading-order/1", "fixture", tuple(logical_pages), ())
    return logical, texts


class SectionTests(unittest.TestCase):
    def classify(self, *pages):
        logical, text = fixture(*pages)
        return detect_sections(logical, text)

    def test_all_section_types(self):
        for section, aliases in ALIASES.items():
            with self.subTest(section=section):
                result = self.classify([(aliases[0], True, 50, 100), ("Content.", False, 50, 140)])
                self.assertEqual([b.section_type for b in result.blocks], [section, section])

    def test_common_aliases(self):
        for section, aliases in ALIASES.items():
            for alias in aliases:
                with self.subTest(alias=alias):
                    result = self.classify([(alias, True, 50, 100)])
                    self.assertEqual(result.blocks[0].section_type, section)

    def test_heading_punctuation(self):
        result = self.classify([("01. TECHNICAL SKILLS:", True, 50, 100)])
        self.assertEqual(result.blocks[0].section_type, SectionType.SKILLS)

    def test_heading_with_following_content_without_typography(self):
        result = self.classify([("Skills", False, 50, 100), ("C# Angular SQL Server Azure", False, 50, 140)])
        self.assertEqual(result.blocks[0].section_type, SectionType.SKILLS)
        self.assertGreater(len(result.blocks[0].evidence.reasons), 1)

    def test_alias_alone_without_structure_remains_unknown(self):
        self.assertEqual(self.classify([("Skills", False, 50, 100)]).blocks[0].section_type, SectionType.UNKNOWN)

    def test_normal_sentences_do_not_become_headings(self):
        for text in ("Experienced developer skilled in C# and Angular.", "Projects included API development."):
            for heading in (True, False):
                with self.subTest(text=text, heading=heading):
                    result = self.classify([(text, heading, 50, 100)])
                    self.assertEqual(result.blocks[0].section_type, SectionType.UNKNOWN)

    def test_unknown_text(self):
        self.assertEqual(self.classify([("Random text", False, 50, 100)]).blocks[0].section_type, SectionType.UNKNOWN)

    def test_boundaries_and_inherited_heading_reference(self):
        result = self.classify([("Skills", True, 50, 100), ("Python", False, 50, 140),
                                ("Projects", True, 50, 180), ("Application", False, 50, 220)])
        self.assertEqual([b.section_type for b in result.blocks],
                         [SectionType.SKILLS, SectionType.SKILLS, SectionType.PROJECTS, SectionType.PROJECTS])
        self.assertEqual(result.blocks[3].heading_block_id, "p0/lb2")

    def test_unrecognized_heading_ends_previous_section(self):
        result = self.classify([("Skills", True, 50, 100), ("Unrecognized Heading", True, 50, 140),
                                ("Content", False, 50, 180)])
        self.assertEqual(result.blocks[-1].section_type, SectionType.UNKNOWN)

    def test_multicolumn_sections_in_either_column(self):
        for left, right in (("Skills", "Experience"), ("Experience", "Skills")):
            result = self.classify([(left, True, 50, 100), ("Content", False, 50, 140),
                                    (right, True, 350, 100), ("Content", False, 350, 140)])
            self.assertEqual(result.blocks[1].section_type.value, left.upper())
            self.assertEqual(result.blocks[3].section_type.value, right.upper())

    def test_column_transition_does_not_leak_context(self):
        result = self.classify([("Skills", True, 50, 100), ("Python", False, 50, 140),
                                ("Unlabeled content", False, 350, 100)])
        self.assertEqual(result.blocks[-1].section_type, SectionType.UNKNOWN)

    def test_aligned_multipage_continuation(self):
        result = self.classify([("Experience", True, 50, 100), ("Record", False, 50, 700)],
                               [("Continued record", False, 50, 50)])
        self.assertEqual(result.blocks[-1].section_type, SectionType.EXPERIENCE)
        self.assertEqual(result.blocks[-1].heading_block_id, "p0/lb0")
        self.assertLess(result.blocks[-1].evidence.score, result.blocks[1].evidence.score)

    def test_unaligned_multipage_content_is_unknown(self):
        result = self.classify([("Experience", True, 50, 100)], [("New text", False, 350, 50)])
        self.assertEqual(result.blocks[-1].section_type, SectionType.UNKNOWN)

    def test_empty_page_resets_context(self):
        result = self.classify([("Experience", True, 50, 100)], [], [("New text", False, 50, 50)])
        self.assertEqual(result.blocks[-1].section_type, SectionType.UNKNOWN)

    def test_provenance_coverage_determinism_and_no_mutation(self):
        logical, text = fixture([("Skills", True, 50, 100), ("Python", False, 50, 140)])
        before = asdict(logical)
        result = detect_sections(logical, text)
        self.assertEqual(asdict(logical), before)
        self.assertEqual(result.to_json(), detect_sections(logical, dict(reversed(list(text.items())))).to_json())
        self.assertEqual([b.logical_block_id for b in result.blocks], list(text))
        self.assertEqual(len(result.blocks), len(set(b.logical_block_id for b in result.blocks)))
        self.assertEqual(result.document_id, logical.document_id)
        self.assertNotIn("Python", result.to_json())

    def test_missing_text_is_rejected(self):
        logical, text = fixture([("Skills", True, 50, 100)])
        with self.assertRaises(ValueError):
            detect_sections(logical, {})

    def test_duplicate_blocks_are_rejected(self):
        logical, text = fixture([("Skills", True, 50, 100)])
        page = logical.pages[0]
        logical = replace(logical, pages=(replace(page, logical_blocks=page.logical_blocks * 2),))
        with self.assertRaises(ValueError):
            detect_sections(logical, text)

    def test_pdf_to_sections_integration(self):
        with fitz.open() as pdf:
            page = pdf.new_page(width=600, height=800)
            for text, y, size in (("Technical Skills", 100, 16), ("Python, SQL, Angular", 150, 10),
                                  ("Selected Projects", 250, 16), ("Built an application.", 300, 10),
                                  ("Additional project details.", 340, 10)):
                page.insert_text((50, y), text, fontsize=size)
            source = extract_pdf(pdf.tobytes())
        layout = analyze_layout(source)
        order = analyze_reading_order(layout)
        logical = analyze_logical_blocks(source, layout, order)
        result = detect_sections(logical, resolve_logical_text(logical, source))
        self.assertEqual([b.section_type for b in result.blocks],
                         [SectionType.SKILLS, SectionType.SKILLS, SectionType.PROJECTS,
                          SectionType.PROJECTS, SectionType.PROJECTS])
        by_id = {b.logical_block_id: b for p in logical.pages for b in p.logical_blocks}
        references = [sid for b in result.blocks for sid in by_id[b.logical_block_id].source_block_ids]
        self.assertEqual(references, list(order.pages[0].ordered_block_ids))
        native = {b.source.object_id: b.source for b in source.pages[0].blocks}
        for assignment in result.blocks:
            for ref in by_id[assignment.logical_block_id].sources:
                self.assertIs(ref, native[ref.object_id])


if __name__ == "__main__":
    unittest.main()

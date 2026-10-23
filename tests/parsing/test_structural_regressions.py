"""Regression patterns found during real-resume inspection."""

from dataclasses import asdict, replace
import unittest

from app.parsing.layout import analyze_layout
from app.parsing.layout.models import LayoutRegion, RegionKind
from app.parsing.models import Box, Evidence
from app.parsing.reading_order import analyze_reading_order
from app.parsing.sections import detect_sections, SectionType
from tests.parsing.layout.fixtures import columns, document
from tests.parsing.logical_blocks.test_logical_blocks import pipeline
from tests.parsing.sections.test_sections import fixture
from app.parsing.entities import extract_entities, EntityType
from tests.parsing.entities.test_entities import inputs


class StructuralRegressionTests(unittest.TestCase):
    def test_unknown_bridge_does_not_interleave_known_columns(self):
        boxes = columns(ys=(140, 210)) + [(50, 600, 270, 620), (330, 600, 550, 620)]
        layout = analyze_layout(document(boxes))
        page = layout.pages[0]
        # An ambiguous band's extent overlaps the supported columns above it.
        supported = analyze_layout(document(columns(ys=(140, 210)))).pages[0]
        regions = supported.regions
        bridge = LayoutRegion("bridge", 0, Box(50, 200, 550, 620), ("p0/b4", "p0/b5"), (),
                              RegionKind.UNKNOWN, Evidence(.2, ("fixture",), ("Ambiguous region",)),
                              tuple(f.source for f in page.block_features if f.block_id in ("p0/b4", "p0/b5")))
        assignments = supported.assignments + tuple(replace(a, region_ids=("bridge",))
                            for a in page.assignments if a.block_id in bridge.block_ids)
        layout = replace(layout, pages=(replace(page, regions=regions + (bridge,), assignments=assignments),))
        before = asdict(layout)
        result = analyze_reading_order(layout)
        self.assertEqual(result.pages[0].ordered_block_ids,
                         ("p0/b0", "p0/b1", "p0/b4", "p0/b2", "p0/b3", "p0/b5"))
        self.assertEqual(asdict(layout), before)
        self.assertEqual(result.to_json(), analyze_reading_order(layout).to_json())

    def test_same_region_different_columns_are_not_grouped(self):
        _, _, _, logical = pipeline([[("Left item", 50, 120, 10), ("Right item", 350, 145, 10)]])
        self.assertEqual(len(logical.pages[0].logical_blocks), 2)
        self.assertTrue(all(len(b.source_block_ids) == 1 for b in logical.pages[0].logical_blocks))

    def test_heading_boundary_is_not_swallowed_by_paragraph(self):
        _, _, _, logical = pipeline([[("Earlier fragment", 50, 120, 10),
                                      ("NEW HEADING", 50, 145, 16), ("Following content.", 50, 175, 10)]])
        self.assertEqual(len(logical.pages[0].logical_blocks), 3)
        self.assertTrue(logical.pages[0].logical_blocks[1].heading_like)

    def test_project_title_retains_section_and_new_section_replaces_it(self):
        logical, text = fixture([("Projects", True, 50, 100), ("Inventory Application", True, 50, 140),
                                 ("Built an application.", False, 50, 180),
                                 ("Education", True, 50, 220), ("Example University", False, 50, 260)])
        result = detect_sections(logical, text)
        self.assertEqual([b.section_type for b in result.blocks],
                         [SectionType.PROJECTS] * 3 + [SectionType.EDUCATION] * 2)
        self.assertFalse(result.blocks[1].is_heading)
        self.assertEqual(result.blocks[1].heading_block_id, "p0/lb0")

    def test_role_and_date_share_one_line_without_losing_raw_offsets(self):
        text = "Software Engineer        July 2022 – Jan 2025"
        sections, logical, resolved = inputs(SectionType.EXPERIENCE, text)
        result = extract_entities(sections, logical, resolved)
        self.assertEqual([(e.entity_type, e.raw_value) for e in result.entities],
                         [(EntityType.ROLE, "Software Engineer"), (EntityType.DATE_RANGE, "July 2022 – Jan 2025")])
        for entity in result.entities:
            self.assertEqual(text[entity.text_start:entity.text_end], entity.raw_value)

    def test_company_with_abbreviated_suffix(self):
        result = extract_entities(*inputs(SectionType.EXPERIENCE, "Example Solutions Pvt. Ltd."))
        self.assertEqual([(e.entity_type, e.raw_value) for e in result.entities],
                         [(EntityType.COMPANY, "Example Solutions Pvt. Ltd.")])

    def test_bold_job_and_company_keep_experience_context(self):
        logical, text = fixture([("Work Experience", True, 50, 100),
                                 ("Software Engineer    2020 - 2022", True, 50, 150),
                                 ("Example Solutions Pvt. Ltd.", True, 50, 190)])
        result = detect_sections(logical, text)
        self.assertTrue(all(b.section_type == SectionType.EXPERIENCE for b in result.blocks))

    def test_project_name_uses_m4_title_context(self):
        logical, text = fixture([("Projects", True, 50, 100),
                                 ("Inventory CRM (Web Forms)", True, 50, 140),
                                 ("Maintained features and supported users.", False, 50, 180)])
        sections = detect_sections(logical, text)
        result = extract_entities(sections, logical, text)
        self.assertEqual(result.entities[0].entity_type, EntityType.PROJECT_NAME)
        self.assertEqual(result.entities[0].raw_value, "Inventory CRM (Web Forms)")


if __name__ == "__main__":
    unittest.main()

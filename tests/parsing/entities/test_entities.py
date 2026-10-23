from dataclasses import asdict, replace
import unittest

from app.parsing.entities import EntityType as E, extract_entities
from app.parsing.models import Evidence
from app.parsing.sections import SectionType as S, SectionBlock, SectionDocumentIR
from tests.parsing.sections.test_sections import fixture


def inputs(section, text):
    logical, resolved = fixture([(text, False, 50, 100)])
    block = logical.pages[0].logical_blocks[0]
    assignments = (SectionBlock(block.logical_block_id, 0, section, None, False,
                                Evidence(.8, ("fixture",), ("Assigned section",))),)
    sections = SectionDocumentIR(logical.document_id, "sections-ir/1", logical.stage_version,
                                  logical.config_fingerprint, "sections/1", assignments)
    return sections, logical, resolved


class EntityTests(unittest.TestCase):
    def entities(self, section, text):
        return extract_entities(*inputs(section, text)).entities

    def values(self, section, text, kind):
        return [entity.raw_value for entity in self.entities(section, text) if entity.entity_type == kind]

    def test_contact_name_requires_label(self):
        self.assertEqual(self.values(S.CONTACT, "Name: Shivam Thakur", E.NAME), ["Shivam Thakur"])
        self.assertEqual(self.values(S.CONTACT, "Shivam Thakur", E.NAME), [])

    def test_email(self):
        self.assertEqual(self.values(S.CONTACT, "Email: shivam@example.com", E.EMAIL), ["shivam@example.com"])
        self.assertEqual(self.values(S.CONTACT, "shivam@example.com", E.EMAIL), ["shivam@example.com"])
        self.assertEqual(self.values(S.CONTACT, "Email: not-an-email", E.EMAIL), [])

    def test_phone(self):
        self.assertEqual(self.values(S.CONTACT, "Phone: +91 9876543210", E.PHONE), ["+91 9876543210"])
        self.assertEqual(self.values(S.CONTACT, "Phone: 123", E.PHONE), [])

    def test_linkedin(self):
        self.assertEqual(self.values(S.CONTACT, "linkedin.com/in/example", E.LINKEDIN), ["linkedin.com/in/example"])

    def test_github(self):
        self.assertEqual(self.values(S.CONTACT, "GitHub: github.com/example", E.GITHUB), ["github.com/example"])

    def test_website(self):
        self.assertEqual(self.values(S.CONTACT, "Website: https://example.test/portfolio", E.WEBSITE), ["https://example.test/portfolio"])
        self.assertEqual(self.values(S.CONTACT, "https://github.com.evil.test/example", E.GITHUB), [])

    def test_contact_location(self):
        self.assertEqual(self.values(S.CONTACT, "Location: Ludhiana", E.LOCATION), ["Ludhiana"])

    def test_skills_preserve_raw_values(self):
        self.assertEqual(self.values(S.SKILLS, "C#, .NET Core, Angular, SQL Server, Azure, TypeScript", E.SKILL),
                         ["C#", ".NET Core", "Angular", "SQL Server", "Azure", "TypeScript"])

    def test_skill_separators(self):
        self.assertEqual(self.values(S.SKILLS, "C#|Angular;SQL Server\n• Azure\n.NET Core", E.SKILL),
                         ["C#", "Angular", "SQL Server", "Azure", ".NET Core"])

    def test_skill_conjunction(self):
        self.assertEqual(self.values(S.SKILLS, "Python and Java", E.SKILL), ["Python", "Java"])

    def test_skill_sentence_is_not_split(self):
        self.assertEqual(self.values(S.SKILLS, "Strong communication and leadership skills", E.SKILL), [])

    def test_experience_role(self):
        self.assertEqual(self.values(S.EXPERIENCE, "Senior Software Engineer", E.ROLE), ["Senior Software Engineer"])

    def test_company(self):
        self.assertEqual(self.values(S.EXPERIENCE, "ABC Technologies", E.COMPANY), ["ABC Technologies"])

    def test_date_ranges(self):
        for date in ("Jan 2022 - Present", "January 2020 - December 2022", "2020 - 2022", "06/2020 - 08/2023"):
            with self.subTest(date=date):
                self.assertEqual(self.values(S.EXPERIENCE, date, E.DATE_RANGE), [date])

    def test_single_dates(self):
        for date in ("2022", "Present"):
            self.assertEqual(self.values(S.EDUCATION, date, E.DATE), [date])

    def test_experience_location(self):
        self.assertEqual(self.values(S.EXPERIENCE, "Ludhiana, India", E.LOCATION), ["Ludhiana, India"])

    def test_experience_description_is_atomic(self):
        text = "Worked with a highly motivated team to build scalable APIs."
        self.assertEqual(self.values(S.EXPERIENCE, text, E.DESCRIPTION), [text])
        self.assertEqual(self.values(S.EXPERIENCE, text, E.COMPANY), [])

    def test_institution(self):
        self.assertEqual(self.values(S.EDUCATION, "XYZ University", E.INSTITUTION), ["XYZ University"])

    def test_degree_and_field(self):
        text = "Bachelor of Technology in Computer Science"
        self.assertEqual(self.values(S.EDUCATION, text, E.DEGREE), ["Bachelor of Technology"])
        self.assertEqual(self.values(S.EDUCATION, text, E.FIELD_OF_STUDY), ["Computer Science"])

    def test_education_dates(self):
        self.assertEqual(self.values(S.EDUCATION, "2018 - 2022", E.DATE_RANGE), ["2018 - 2022"])

    def test_project_name_needs_context(self):
        text = "Employee Management System\nBuilt a web application for managing employees."
        self.assertEqual(self.values(S.PROJECTS, text, E.PROJECT_NAME), ["Employee Management System"])
        self.assertEqual(self.values(S.PROJECTS, "Employee Management System", E.PROJECT_NAME), [])

    def test_project_description(self):
        text = "Built a web application for managing employees."
        self.assertEqual(self.values(S.PROJECTS, text, E.PROJECT_DESCRIPTION), [text])

    def test_project_technologies(self):
        self.assertEqual(self.values(S.PROJECTS, "Technologies: Angular, .NET Core, SQL Server", E.TECHNOLOGY),
                         ["Angular", ".NET Core", "SQL Server"])

    def test_certification(self):
        self.assertEqual(self.values(S.CERTIFICATIONS, "AWS Certified Developer", E.CERTIFICATION), ["AWS Certified Developer"])
        self.assertEqual(self.values(S.CERTIFICATIONS, "Issuer: Amazon Web Services", E.ISSUING_ORGANIZATION), ["Amazon Web Services"])

    def test_credential(self):
        self.assertEqual(self.values(S.CERTIFICATIONS, "Credential ID: ABC-123", E.CREDENTIAL_ID), ["ABC-123"])
        self.assertEqual(self.values(S.CERTIFICATIONS, "Credential URL: https://example.test/verify/123", E.CREDENTIAL_URL), ["https://example.test/verify/123"])

    def test_award(self):
        self.assertEqual(self.values(S.AWARDS, "Innovation Award", E.AWARD), ["Innovation Award"])
        self.assertEqual(self.values(S.AWARDS, "Issuer: Example Foundation", E.ISSUING_ORGANIZATION), ["Example Foundation"])

    def test_languages(self):
        self.assertEqual(self.values(S.LANGUAGES, "English and Hindi\nPunjabi", E.LANGUAGE), ["English", "Hindi", "Punjabi"])
        self.assertEqual(self.values(S.LANGUAGES, "Python, Java, C#", E.LANGUAGE), [])

    def test_interests(self):
        self.assertEqual(self.values(S.INTERESTS, "Photography; Hiking", E.INTEREST), ["Photography", "Hiking"])
        self.assertEqual(self.values(S.INTERESTS, "I enjoy photography and hiking.", E.INTEREST), [])

    def test_uncertain_values(self):
        text = "Technology enthusiast and problem solver"
        entities = self.entities(S.UNKNOWN, text)
        self.assertEqual([(e.entity_type, e.raw_value) for e in entities], [(E.UNKNOWN, text)])

    def test_negative_prose_and_section_scope(self):
        text = "Experienced developer skilled in C# and Angular."
        for section in (S.UNKNOWN, S.SKILLS, S.EXPERIENCE):
            entities = self.entities(section, text)
            self.assertFalse(any(e.entity_type in (E.ROLE, E.SKILL) for e in entities))
        self.assertEqual(self.values(S.PROJECTS, "Projects included API development.", E.PROJECT_NAME), [])
        self.assertEqual(self.values(S.UNKNOWN, "ABC Technologies", E.COMPANY), [])
        self.assertEqual(self.values(S.UNKNOWN, "Bachelor of Technology", E.DEGREE), [])

    def test_missing_values(self):
        self.assertEqual(self.entities(S.CONTACT, ""), ())
        self.assertEqual(self.values(S.CONTACT, "Email:", E.EMAIL), [])

    def test_label_value_on_separate_lines(self):
        self.assertEqual(self.values(S.CONTACT, "Name:\nShivam Thakur\nEmail:\nabc@example.test", E.NAME), ["Shivam Thakur"])
        self.assertEqual(self.values(S.CONTACT, "Name:\nShivam Thakur\nEmail:\nabc@example.test", E.EMAIL), ["abc@example.test"])

    def test_labeled_dates_validate_values(self):
        self.assertEqual(self.values(S.CERTIFICATIONS, "Date: Jan 2022", E.DATE), ["Jan 2022"])
        self.assertEqual(self.values(S.EDUCATION, "Graduation year: 2022", E.DATE), ["2022"])
        self.assertEqual(self.values(S.EXPERIENCE, "Dates: 2020 - 2022", E.DATE_RANGE), ["2020 - 2022"])
        self.assertEqual(self.values(S.CERTIFICATIONS, "Date: someday", E.DATE), [])

    def test_provenance_offsets_and_determinism(self):
        sections, logical, text = inputs(S.SKILLS, "  .NET Core; C#\r\nAngular  ")
        before = (asdict(sections), asdict(logical), dict(text))
        result = extract_entities(sections, logical, text)
        self.assertEqual(before, (asdict(sections), asdict(logical), text))
        self.assertEqual(result.to_json(), extract_entities(sections, logical, text).to_json())
        block = logical.pages[0].logical_blocks[0]
        for entity in result.entities:
            self.assertEqual(entity.raw_value, text[entity.logical_block_id][entity.text_start:entity.text_end])
            self.assertIs(entity.sources, block.sources)
            self.assertEqual(entity.source_block_ids, block.source_block_ids)
            self.assertEqual(entity.section_type, S.SKILLS)

    def test_no_duplicate_entities_from_labels(self):
        entities = self.entities(S.CONTACT, "Email: abc@example.test")
        self.assertEqual(len(entities), 1)
        self.assertEqual(len({e.entity_id for e in entities}), len(entities))

    def test_invalid_inputs_are_rejected(self):
        sections, logical, text = inputs(S.SKILLS, "Python")
        with self.assertRaises(ValueError):
            extract_entities(sections, logical, {})
        with self.assertRaises(ValueError):
            extract_entities(replace(sections, document_id="other"), logical, text)
        with self.assertRaises(ValueError):
            extract_entities(replace(sections, blocks=sections.blocks * 2), logical, text)

    def test_heading_is_not_an_entity(self):
        sections, logical, text = inputs(S.SKILLS, "Skills")
        sections = replace(sections, blocks=(replace(sections.blocks[0], is_heading=True),))
        self.assertEqual(extract_entities(sections, logical, text).entities, ())

    def test_multicolumn_and_multipage_use_m5_assignments(self):
        logical, text = fixture([("Python", False, 350, 100), ("Senior Engineer", False, 50, 100)],
                                 [("ABC Technologies", False, 50, 50)])
        kinds = (S.SKILLS, S.EXPERIENCE, S.EXPERIENCE)
        blocks = [b for p in logical.pages for b in p.logical_blocks]
        assignments = tuple(SectionBlock(b.logical_block_id, b.page_number, kind, None, False,
                                          Evidence(.7, ("fixture",), ("Assigned section",))) for b, kind in zip(blocks, kinds))
        sections = SectionDocumentIR("fixture", "sections-ir/1", logical.stage_version, logical.config_fingerprint, "sections/1", assignments)
        result = extract_entities(sections, logical, text)
        self.assertEqual([e.entity_type for e in result.entities], [E.SKILL, E.ROLE, E.COMPANY])
        self.assertEqual([e.page_number for e in result.entities], [0, 0, 1])

    def test_m5_to_m6_pdf_integration(self):
        import fitz
        from app.parsing.pdf_objects import extract_pdf
        from app.parsing.layout import analyze_layout
        from app.parsing.reading_order import analyze_reading_order
        from app.parsing.logical_blocks import analyze_logical_blocks
        from app.parsing.sections import detect_sections, resolve_logical_text
        with fitz.open() as pdf:
            page = pdf.new_page(width=600, height=800)
            page.insert_text((50, 100), "Skills", fontsize=16)
            page.insert_text((50, 150), ".NET Core, Angular, SQL Server", fontsize=10)
            page.insert_text((50, 200), "Azure", fontsize=10)
            source = extract_pdf(pdf.tobytes())
        layout = analyze_layout(source)
        order = analyze_reading_order(layout)
        logical = analyze_logical_blocks(source, layout, order)
        text = resolve_logical_text(logical, source)
        sections = detect_sections(logical, text)
        result = extract_entities(sections, logical, text)
        self.assertEqual([e.raw_value for e in result.entities], [".NET Core", "Angular", "SQL Server", "Azure"])
        native = {b.source.object_id: b.source for b in source.pages[0].blocks}
        for entity in result.entities:
            for ref in entity.sources:
                self.assertIs(ref, native[ref.object_id])


if __name__ == "__main__":
    unittest.main()

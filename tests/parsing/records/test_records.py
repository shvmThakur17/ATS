from pathlib import Path
import unittest

from app.parsing.entities.models import Entity, EntityDocumentIR, EntityType as E
from app.parsing.models import Box, Evidence, SourceRef
from app.parsing.records import build_records
from app.parsing.sections import SectionType as S


def ir(*specs):
    """specs: (EntityType, raw_value, SectionType, block_index)."""
    entities = []
    for index, (kind, value, section, block) in enumerate(specs):
        block_id = f"p0/lb{block}"
        source = SourceRef("fixture", 0, f"p0/b{block}", Box(0, 0, 10, 10))
        entities.append(Entity(f"{block_id}/e{index}", kind, value, block_id, 0, section,
                               (f"p0/b{block}",), (source,), 0, len(value),
                               Evidence(.8, ("m6/fixture/1",), ("fixture",))))
    return EntityDocumentIR("fixture", "entities-ir/1", "entities/1", "sections/1", "cfg", tuple(entities))


X, C, ED, P = S.EXPERIENCE, S.CONTACT, S.EDUCATION, S.PROJECTS


class RecordTests(unittest.TestCase):
    def test_contact_record(self):
        r = build_records(ir((E.NAME, "KUKOO", S.UNKNOWN, 0), (E.EMAIL, "a@b.com", C, 1),
                             (E.PHONE, "123456", C, 1), (E.LOCATION, "Mohali", C, 2))).contact
        self.assertEqual((r.name, r.email, r.phone, r.location), ("KUKOO", "a@b.com", "123456", "Mohali"))
        self.assertIsNone(r.linkedin)
        self.assertIsNone(r.github)
        self.assertIsNone(r.website)

    def test_empty_input_gives_empty_contact(self):
        r = build_records(ir())
        self.assertIsNone(r.contact.name)
        self.assertEqual(r.contact.raw_text, "")
        self.assertEqual(r.work_experience, ())
        self.assertIsNone(r.skills)

    def test_multiple_work_records_role_first(self):
        r = build_records(ir((E.ROLE, "Dev", X, 0), (E.DATE_RANGE, "2022", X, 0), (E.COMPANY, "A", X, 1),
                             (E.ROLE, "Lead", X, 2), (E.COMPANY, "B", X, 3))).work_experience
        self.assertEqual([(w.role, w.company, w.date_range) for w in r],
                         [("Dev", "A", "2022"), ("Lead", "B", None)])

    def test_multiple_work_records_company_first(self):
        r = build_records(ir((E.COMPANY, "A", X, 0), (E.ROLE, "Dev", X, 1), (E.DATE_RANGE, "2022", X, 1),
                             (E.DESCRIPTION, "Built", X, 2), (E.COMPANY, "B", X, 3),
                             (E.ROLE, "Lead", X, 4))).work_experience
        self.assertEqual([(w.company, w.role, w.description) for w in r],
                         [("A", "Dev", "Built"), ("B", "Lead", None)])

    def test_work_missing_company(self):
        r = build_records(ir((E.ROLE, "Dev", X, 0), (E.DATE_RANGE, "2022", X, 0))).work_experience
        self.assertEqual(len(r), 1)
        self.assertIsNone(r[0].company)

    def test_work_missing_role(self):
        r = build_records(ir((E.COMPANY, "A", X, 0), (E.DATE_RANGE, "2022", X, 0))).work_experience
        self.assertEqual(len(r), 1)
        self.assertIsNone(r[0].role)

    def test_multiple_education_records(self):
        r = build_records(ir((E.DEGREE, "10th", ED, 0), (E.INSTITUTION, "School", ED, 1),
                             (E.INSTITUTION, "College", ED, 2), (E.DEGREE, "Diploma", ED, 2),
                             (E.FIELD_OF_STUDY, "CS", ED, 2), (E.DEGREE, "BSc", ED, 3))).education
        self.assertEqual([(e.institution, e.degree, e.field_of_study) for e in r],
                         [("School", "10th", None), ("College", "Diploma", "CS"), (None, "BSc", None)])

    def test_education_date_range_stays_in_education(self):
        r = build_records(ir((E.INSTITUTION, "College", ED, 0), (E.DATE_RANGE, "2015-2018", ED, 0)))
        self.assertEqual(r.education[0].date_range, "2015-2018")
        self.assertEqual(r.work_experience, ())

    def test_project_grouping_and_descriptions(self):
        r = build_records(ir((E.PROJECT_NAME, "A", P, 0), (E.PROJECT_DESCRIPTION, "desc A", P, 1),
                             (E.TECHNOLOGY, "C#", P, 1), (E.TECHNOLOGY, "SQL", P, 1),
                             (E.PROJECT_NAME, "B", P, 2), (E.PROJECT_DESCRIPTION, "desc B", P, 3))).projects
        self.assertEqual([(p.name, p.description, p.technology) for p in r],
                         [("A", "desc A", ("C#", "SQL")), ("B", "desc B", ())])

    def test_education_entity_does_not_leak_into_project(self):
        r = build_records(ir((E.PROJECT_NAME, "A", P, 0), (E.PROJECT_DESCRIPTION, "desc A", P, 1),
                             (E.DEGREE, "BSc", ED, 2), (E.INSTITUTION, "College", ED, 2)))
        self.assertEqual(len(r.projects), 1)
        self.assertEqual(r.projects[0].raw_text, "A\ndesc A")
        self.assertEqual(len(r.education), 1)

    def test_section_change_closes_record(self):
        r = build_records(ir((E.PROJECT_NAME, "A", P, 0), (E.ROLE, "Dev", X, 1),
                             (E.PROJECT_DESCRIPTION, "late", P, 2)))
        self.assertEqual(len(r.projects), 2)

    def test_skills_languages_interests_awards(self):
        r = build_records(ir((E.SKILL, ".NET", S.SKILLS, 0), (E.SKILL, "Angular", S.SKILLS, 0),
                             (E.SKILL, ".NET", S.SKILLS, 1), (E.SKILL, "SQL Server", S.SKILLS, 1),
                             (E.LANGUAGE, "English", S.LANGUAGES, 2), (E.LANGUAGE, "Hindi", S.LANGUAGES, 2),
                             (E.INTEREST, "Chess", S.INTERESTS, 3),
                             (E.AWARD, "Awarded Best\nEmployee", S.AWARDS, 4), (E.AWARD, "Won X", S.AWARDS, 5)))
        self.assertEqual(r.skills.skills, (".NET", "Angular", "SQL Server"))
        self.assertEqual(len(r.skills.entities), 4)  # exact duplicate kept in provenance
        self.assertEqual(r.languages.languages, ("English", "Hindi"))
        self.assertEqual(r.interests.interests, ("Chess",))
        self.assertEqual([a.text for a in r.awards], ["Awarded Best\nEmployee", "Won X"])

    def test_certifications(self):
        r = build_records(ir((E.CERTIFICATION, "AZ-900", S.CERTIFICATIONS, 0),
                             (E.ISSUING_ORGANIZATION, "Microsoft", S.CERTIFICATIONS, 0),
                             (E.CREDENTIAL_ID, "ID1", S.CERTIFICATIONS, 0),
                             (E.CERTIFICATION, "AWS", S.CERTIFICATIONS, 1))).certifications
        self.assertEqual([(c.certification, c.issuing_organization, c.credential_id, c.credential_url)
                          for c in r], [("AZ-900", "Microsoft", "ID1", None), ("AWS", None, None, None)])

    def test_missing_entities_are_not_fabricated(self):
        r = build_records(ir((E.PROJECT_NAME, "A", P, 0)))
        self.assertIsNone(r.projects[0].description)
        self.assertEqual(r.projects[0].technology, ())
        self.assertIsNone(r.skills)
        self.assertIsNone(r.languages)
        self.assertEqual(r.awards, ())

    def test_provenance_preserved(self):
        source = ir((E.ROLE, "Dev", X, 7), (E.COMPANY, "A", X, 8))
        record = build_records(source).work_experience[0]
        self.assertEqual(record.logical_block_ids, ("p0/lb7", "p0/lb8"))
        self.assertEqual(record.raw_text, "Dev\nA")
        for ref, entity in zip(record.entities, source.entities):
            self.assertEqual((ref.entity_id, ref.entity_type, ref.raw_value, ref.logical_block_id,
                              ref.section_type, ref.page_number, ref.source_block_ids, ref.text_start,
                              ref.text_end, ref.sources, ref.evidence),
                             (entity.entity_id, entity.entity_type, entity.raw_value, entity.logical_block_id,
                              entity.section_type, entity.page_number, entity.source_block_ids,
                              entity.text_start, entity.text_end, entity.sources, entity.evidence))

    def test_deterministic(self):
        source = ir((E.NAME, "N", S.UNKNOWN, 0), (E.ROLE, "Dev", X, 1), (E.COMPANY, "A", X, 2),
                    (E.SKILL, "C#", S.SKILLS, 3), (E.UNKNOWN, "?", S.UNKNOWN, 4))
        self.assertEqual(build_records(source).model_dump_json(), build_records(source).model_dump_json())

    def test_unknown_entities_remain_unassigned(self):
        r = build_records(ir((E.UNKNOWN, "DOT NET DEVELOPER", S.UNKNOWN, 1), (E.DATE, "2020", X, 2),
                             (E.DESCRIPTION, "summary text", S.SUMMARY, 3)))
        self.assertEqual([(u.entity_type, u.raw_value) for u in r.unassigned],
                         [(E.UNKNOWN, "DOT NET DEVELOPER"), (E.DATE, "2020")])
        self.assertEqual(r.work_experience, ())

    def test_summary_record(self):
        text = ".NET Developer with 2.6+ years\nof experience."
        source = ir((E.UNKNOWN, "DOT NET DEVELOPER", S.UNKNOWN, 1), (E.DESCRIPTION, text, S.SUMMARY, 23))
        r = build_records(source)
        self.assertEqual(r.summary.text, text)
        self.assertEqual(r.summary.raw_text, text)
        self.assertEqual(r.summary.logical_block_ids, ("p0/lb23",))
        ref, entity = r.summary.entities[0], source.entities[1]
        self.assertEqual((ref.entity_id, ref.entity_type, ref.section_type, ref.page_number,
                          ref.source_block_ids, ref.text_start, ref.text_end, ref.sources, ref.evidence),
                         (entity.entity_id, E.DESCRIPTION, S.SUMMARY, 0, entity.source_block_ids,
                          entity.text_start, entity.text_end, entity.sources, entity.evidence))
        self.assertEqual([u.raw_value for u in r.unassigned], ["DOT NET DEVELOPER"])

    def test_summary_not_used_as_work_or_project_description(self):
        r = build_records(ir((E.ROLE, "Dev", X, 0), (E.DESCRIPTION, "summary", S.SUMMARY, 1),
                             (E.PROJECT_NAME, "A", P, 2)))
        self.assertIsNone(r.work_experience[0].description)
        self.assertEqual(r.work_experience[0].raw_text, "Dev")
        self.assertIsNone(r.projects[0].description)
        self.assertEqual(r.summary.text, "summary")

    def test_no_summary_entity_gives_none(self):
        self.assertIsNone(build_records(ir((E.ROLE, "Dev", X, 0))).summary)

    def test_surplus_contact_value_is_not_dropped(self):
        r = build_records(ir((E.EMAIL, "a@b.com", C, 0), (E.EMAIL, "c@d.com", C, 0)))
        self.assertEqual(r.contact.email, "a@b.com")
        self.assertEqual([u.raw_value for u in r.unassigned], ["c@d.com"])

    def test_real_kukoo_resume(self):
        pdf = Path(__file__).resolve().parents[3] / "uploads" / "Kukoo.pdf"
        if not pdf.is_file():
            self.skipTest("uploads/Kukoo.pdf not available")
        from app.parsing.pdf_objects import extract_pdf
        from app.parsing.layout import analyze_layout
        from app.parsing.reading_order import analyze_reading_order
        from app.parsing.logical_blocks import analyze_logical_blocks
        from app.parsing.sections import detect_sections, resolve_logical_text
        from app.parsing.entities import extract_entities
        document = extract_pdf(pdf)
        layout = analyze_layout(document)
        logical = analyze_logical_blocks(document, layout, analyze_reading_order(layout))
        text = resolve_logical_text(logical, document)
        entities = extract_entities(detect_sections(logical, text), logical, text)
        r = build_records(entities)
        self.assertEqual(r.contact.name, "KUKOO RATHEE")
        self.assertEqual(len(r.work_experience), 2)
        self.assertEqual(len(r.education), 3)
        self.assertEqual(len(r.projects), 7)
        self.assertEqual([(u.entity_type, u.raw_value) for u in r.unassigned], [(E.UNKNOWN, "DOT NET DEVELOPER")])
        self.assertTrue(r.summary.text.startswith(".NET Developer with 2.6+ years"))
        self.assertIsNone(r.work_experience[0].description)


if __name__ == "__main__":
    unittest.main()

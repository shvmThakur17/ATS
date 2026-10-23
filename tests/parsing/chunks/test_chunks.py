import json
from pathlib import Path
import unittest

from app.parsing.chunks import ChunkType as T, compile_chunks
from app.parsing.entities.models import EntityType as E
from app.parsing.sections import SectionType as S
from tests.parsing.canonical.test_canonical import canon

X, C, ED, P = S.EXPERIENCE, S.CONTACT, S.EDUCATION, S.PROJECTS
DASH = "–"

FULL = (
    (E.NAME, "KUKOO RATHEE", S.UNKNOWN, 0), (E.UNKNOWN, "DOT NET DEVELOPER", S.UNKNOWN, 1),
    (E.PHONE, "8683077463", C, 2), (E.EMAIL, "kukoo@gmail.com", C, 2), (E.LOCATION, "Mohali Sector 71", C, 3),
    (E.DESCRIPTION, ".NET Developer with 2.6+ years of experience.", S.SUMMARY, 4),
    (E.ROLE, "Software Engineer", X, 5), (E.DATE_RANGE, f"July 2022 {DASH} Jan 2025", X, 5),
    (E.COMPANY, "Ginlytics IT Solutions Pvt. Ltd.", X, 6),
    (E.ROLE, "Software Engineer", X, 7), (E.DATE_RANGE, f"Jan 2025{DASH} Present", X, 7),
    (E.COMPANY, "Off Beat Software Solutions Pvt. Ltd.", X, 8),
    (E.PROJECT_NAME, "Shop Findy (API)", P, 9),
    (E.PROJECT_DESCRIPTION, "Developed RESTful APIs, using SQL Server for data.", P, 10),
    (E.PROJECT_NAME, "GST Vendor", P, 11),
    (E.DEGREE, "DIPLOMA", ED, 12), (E.FIELD_OF_STUDY, "CS", ED, 12), (E.INSTITUTION, "Govt. Polytechnic", ED, 13),
    (E.DEGREE, "10th", ED, 14), (E.INSTITUTION, "Govt. School", ED, 15),
    (E.SKILL, "ASP.Net Core", S.SKILLS, 16), (E.SKILL, "JQuery", S.SKILLS, 16), (E.SKILL, "SQL", S.SKILLS, 17),
    (E.CERTIFICATION, "AZ-900", S.CERTIFICATIONS, 18), (E.ISSUING_ORGANIZATION, "Microsoft", S.CERTIFICATIONS, 18),
    (E.CREDENTIAL_ID, "ID-77", S.CERTIFICATIONS, 18),
    (E.AWARD, "Awarded Best\nEmployee", S.AWARDS, 19),
    (E.LANGUAGE, "English", S.LANGUAGES, 20), (E.LANGUAGE, "Hindi", S.LANGUAGES, 20),
    (E.INTEREST, "Chess", S.INTERESTS, 21),
)


def of(chunks, chunk_type):
    return [c for c in chunks if c.chunk_type == chunk_type]


class ChunkTests(unittest.TestCase):
    def setUp(self):
        self.resume = canon(*FULL)
        self.chunks = compile_chunks(self.resume)

    def test_counts_per_type(self):
        counts = {t: len(of(self.chunks, t)) for t in T}
        self.assertEqual(counts, {T.SUMMARY: 1, T.WORK_EXPERIENCE: 2, T.PROJECT: 2, T.EDUCATION: 2, T.SKILLS: 1,
                                  T.CERTIFICATION: 1, T.AWARD: 1, T.LANGUAGE: 1, T.INTEREST: 1, T.CONTACT: 0})

    def test_summary_chunk(self):
        c = of(self.chunks, T.SUMMARY)[0]
        self.assertEqual((c.text, c.canonical_path, c.source_fields, c.chunk_id),
                         (".NET Developer with 2.6+ years of experience.", "summary", ("summary",), "resume-summary-001"))

    def test_work_chunks_one_per_record(self):
        a, b = of(self.chunks, T.WORK_EXPERIENCE)
        self.assertEqual(a.text, "Role: Software Engineer\nCompany: Ginlytics IT Solutions Pvt. Ltd.\n"
                                 "Employment: 2022-07 to 2025-01")
        self.assertIn("Company: Off Beat Software Solutions Pvt. Ltd.", b.text)
        self.assertNotIn("Ginlytics", b.text)

    def test_current_employment_is_present(self):
        b = of(self.chunks, T.WORK_EXPERIENCE)[1]
        self.assertIn("Employment: 2025-01 to Present", b.text)
        self.assertIs(b.metadata["current"], True)

    def test_project_chunks_no_inferred_technologies(self):
        a, b = of(self.chunks, T.PROJECT)
        self.assertEqual(a.text, "Project: Shop Findy (API)\nDescription: Developed RESTful APIs, using SQL Server for data.")
        self.assertNotIn("Technologies", a.text)
        self.assertNotIn("technologies", a.metadata)  # M9 has none, so none are listed
        self.assertEqual(b.text, "Project: GST Vendor")

    def test_project_technologies_only_from_m9(self):
        resume = canon((E.PROJECT_NAME, "A", P, 0), (E.TECHNOLOGY, "JQuery", P, 1), (E.TECHNOLOGY, "C#", P, 1))
        c = compile_chunks(resume)[0]
        self.assertEqual(c.text, "Project: A\nTechnologies: jQuery, C#")
        self.assertEqual(c.metadata["technologies"], ["jQuery", "C#"])

    def test_education_chunks(self):
        a, b = of(self.chunks, T.EDUCATION)
        self.assertEqual(a.text, "Degree: Diploma\nField of Study: CS\nInstitution: Govt. Polytechnic")
        self.assertEqual(b.text, "Degree: 10th\nInstitution: Govt. School")

    def test_skills_one_chunk_with_individual_values_in_metadata(self):
        (c,) = of(self.chunks, T.SKILLS)
        self.assertEqual(c.text, "ASP.NET Core, jQuery, SQL")
        self.assertEqual(c.metadata["skills"], ["ASP.NET Core", "jQuery", "SQL"])
        self.assertEqual(c.metadata["raw_skills"], ["ASP.Net Core", "JQuery", "SQL"])
        self.assertNotIn("SQL Server", c.text)

    def test_certification_award_language_interest(self):
        (cert,) = of(self.chunks, T.CERTIFICATION)
        self.assertEqual(cert.text, "Certification: AZ-900\nIssuer: Microsoft")
        self.assertNotIn("ID-77", cert.text)
        self.assertEqual(cert.metadata["credential_id"], "ID-77")
        self.assertEqual(of(self.chunks, T.AWARD)[0].text, "Awarded Best Employee")
        self.assertEqual(of(self.chunks, T.AWARD)[0].metadata["raw_award"], "Awarded Best\nEmployee")
        self.assertEqual(of(self.chunks, T.LANGUAGE)[0].text, "English, Hindi")
        self.assertEqual(of(self.chunks, T.INTEREST)[0].text, "Chess")

    def test_empty_sections_create_no_chunks(self):
        chunks = compile_chunks(canon((E.NAME, "N", S.UNKNOWN, 0)))
        self.assertEqual(chunks, [])
        chunks = compile_chunks(canon((E.NAME, "N", S.UNKNOWN, 0), (E.SKILL, "C#", S.SKILLS, 1)))
        self.assertEqual([c.chunk_type for c in chunks], [T.SKILLS])
        self.assertTrue(all(c.text.strip() for c in self.chunks))

    def test_contact_disabled_by_default_and_excludes_email_phone(self):
        self.assertEqual(of(self.chunks, T.CONTACT), [])
        chunks = compile_chunks(self.resume, include_contact=True)
        (c,) = of(chunks, T.CONTACT)
        self.assertEqual(c.text, "Name: KUKOO RATHEE\nLocation: Mohali Sector 71")
        self.assertNotIn("8683077463", c.text)
        self.assertNotIn("@", c.text)
        self.assertEqual((c.metadata["email"], c.metadata["phone"]), ("kukoo@gmail.com", "8683077463"))
        self.assertEqual(c.chunk_id, "resume-contact-001")
        self.assertEqual(chunks[-1], c)  # contact is last

    def test_contact_chunk_absent_when_no_name_or_location(self):
        chunks = compile_chunks(canon((E.EMAIL, "a@b.com", C, 0)), include_contact=True)
        self.assertEqual(chunks, [])

    def test_unknown_is_not_a_chunk(self):
        resume = canon((E.UNKNOWN, "DOT NET DEVELOPER", S.UNKNOWN, 1))
        self.assertEqual(compile_chunks(resume), [])
        self.assertTrue(all("DOT NET DEVELOPER" not in c.text and "DOT NET DEVELOPER" not in json.dumps(c.metadata)
                            for c in self.chunks))

    def test_ids_and_order(self):
        self.assertEqual([c.chunk_id for c in self.chunks],
                         ["resume-summary-001", "resume-work-001", "resume-work-002", "resume-project-001",
                          "resume-project-002", "resume-education-001", "resume-education-002", "resume-skills-001",
                          "resume-certification-001", "resume-award-001", "resume-language-001", "resume-interest-001"])
        self.assertEqual(len({c.chunk_id for c in self.chunks}), len(self.chunks))
        self.assertEqual([c.source_record_id for c in of(self.chunks, T.WORK_EXPERIENCE)],
                         ["work-experience-001", "work-experience-002"])

    def test_deterministic(self):
        again = compile_chunks(canon(*FULL))
        self.assertEqual([c.model_dump_json() for c in self.chunks], [c.model_dump_json() for c in again])

    def test_document_order_not_sorted(self):
        resume = canon((E.PROJECT_NAME, "Zeta", P, 0), (E.PROJECT_NAME, "Alpha", P, 1),
                       (E.SKILL, "Zed", S.SKILLS, 2), (E.SKILL, "Alpha", S.SKILLS, 2))
        chunks = compile_chunks(resume)
        self.assertEqual([c.text for c in of(chunks, T.PROJECT)], ["Project: Zeta", "Project: Alpha"])
        self.assertEqual(of(chunks, T.SKILLS)[0].text, "Zed, Alpha")

    def test_identical_records_stay_separate(self):
        chunks = compile_chunks(canon((E.PROJECT_NAME, "Same", P, 0), (E.PROJECT_NAME, "Same", P, 1)))
        self.assertEqual([c.chunk_id for c in chunks], ["resume-project-001", "resume-project-002"])

    def test_canonical_path_and_source_fields(self):
        a, b = of(self.chunks, T.WORK_EXPERIENCE)
        self.assertEqual((a.canonical_path, b.canonical_path), ("work_experience[0]", "work_experience[1]"))
        self.assertEqual(a.source_fields, ("role", "company", "start_date", "end_date"))
        self.assertEqual(b.source_fields, ("role", "company", "start_date", "current"))
        self.assertEqual(of(self.chunks, T.PROJECT)[1].canonical_path, "projects[1]")

    def test_logical_block_provenance(self):
        a = of(self.chunks, T.WORK_EXPERIENCE)[0]
        self.assertEqual(a.provenance.logical_block_ids, ("p0/lb5", "p0/lb6"))
        self.assertEqual(a.provenance.page_numbers, (0,))
        self.assertEqual(a.provenance.source_block_ids, ("p0/b5", "p0/b6"))
        self.assertEqual(a.metadata["logical_block_ids"], ["p0/lb5", "p0/lb6"])
        (skills,) = of(self.chunks, T.SKILLS)
        self.assertEqual(skills.provenance.logical_block_ids, ("p0/lb16", "p0/lb17"))
        (summary,) = of(self.chunks, T.SUMMARY)
        self.assertEqual(summary.provenance.logical_block_ids, ("p0/lb4",))

    def test_every_chunk_keeps_m9_provenance_and_raw_values(self):
        for c in self.chunks:
            self.assertTrue(c.provenance.fields, c.chunk_id)
            self.assertTrue(c.provenance.logical_block_ids, c.chunk_id)
            self.assertEqual(set(c.provenance.raw_values), set(c.provenance.fields))
            for provenance in c.provenance.fields.values():
                self.assertTrue(provenance.entities)
        a = of(self.chunks, T.WORK_EXPERIENCE)[0]
        self.assertEqual(a.provenance.raw_values["date_range"], f"July 2022 {DASH} Jan 2025")
        self.assertEqual((a.metadata["raw_role"], a.metadata["raw_company"], a.metadata["raw_date_range"]),
                         ("Software Engineer", "Ginlytics IT Solutions Pvt. Ltd.", f"July 2022 {DASH} Jan 2025"))
        self.assertEqual(a.provenance.fields["role"].entities[0].entity_id, "p0/lb5/e6")

    def test_no_invented_information(self):
        resume = canon((E.ROLE, "Dev", X, 0))
        (c,) = compile_chunks(resume)
        self.assertEqual(c.text, "Role: Dev")
        self.assertNotIn("company", c.metadata)
        self.assertNotIn("start_date", c.metadata)
        lone = canon((E.COMPANY, "A", X, 0), (E.DATE_RANGE, "2018", X, 0))
        self.assertEqual(compile_chunks(lone)[0].text, "Company: A")  # lone date is not turned into a range

    def test_no_json_or_debug_in_text(self):
        for c in self.chunks:
            self.assertNotIn("{", c.text)
            self.assertNotIn("}", c.text)
            self.assertNotIn("entity_id", c.text)
            self.assertNotIn("p0/lb", c.text)
            self.assertNotIn("None", c.text)

    def test_metadata_is_flat_and_json_safe(self):
        for c in self.chunks:
            json.dumps(c.metadata)
            for value in c.metadata.values():
                self.assertIsNotNone(value)
                self.assertTrue(isinstance(value, (str, int, bool)) or
                                (isinstance(value, list) and all(isinstance(v, (str, int)) for v in value)), c.chunk_id)

    def test_long_record_is_never_split_or_truncated(self):
        long_text = "Sentence number one is here. " * 400
        resume = canon((E.PROJECT_NAME, "Big", P, 0), (E.PROJECT_DESCRIPTION, long_text, P, 1))
        chunks = compile_chunks(resume)
        self.assertEqual(len(chunks), 1)
        self.assertIn(long_text.strip(), chunks[0].text)

    def test_real_kukoo(self):
        pdf = Path(__file__).resolve().parents[3] / "uploads" / "Kukoo.pdf"
        if not pdf.is_file():
            self.skipTest("uploads/Kukoo.pdf not available")
        from app.parsing.pdf_objects import extract_pdf
        from app.parsing.layout import analyze_layout
        from app.parsing.reading_order import analyze_reading_order
        from app.parsing.logical_blocks import analyze_logical_blocks
        from app.parsing.sections import detect_sections, resolve_logical_text
        from app.parsing.entities import extract_entities
        from app.parsing.records import build_records
        from app.parsing.normalization import normalize_records
        from app.parsing.canonical import build_canonical_resume
        document = extract_pdf(pdf)
        layout = analyze_layout(document)
        logical = analyze_logical_blocks(document, layout, analyze_reading_order(layout))
        text = resolve_logical_text(logical, document)
        entities = extract_entities(detect_sections(logical, text), logical, text)
        resume = build_canonical_resume(normalize_records(build_records(entities)), pdf.name, len(document.pages),
                                        sum(len(p.logical_blocks) for p in logical.pages), len(entities.entities))
        chunks = compile_chunks(resume)
        self.assertEqual({t: len(of(chunks, t)) for t in T if of(chunks, t)},
                         {T.SUMMARY: 1, T.WORK_EXPERIENCE: 2, T.PROJECT: 7, T.EDUCATION: 3, T.SKILLS: 1,
                          T.AWARD: 1, T.LANGUAGE: 1})
        self.assertEqual(len(compile_chunks(resume, include_contact=True)), 17)
        self.assertEqual(of(chunks, T.SKILLS)[0].text.split(", ")[:2], ["HTML", "CSS"])
        self.assertTrue(all("DOT NET DEVELOPER" not in c.text for c in chunks))
        self.assertEqual([c.model_dump_json() for c in chunks], [c.model_dump_json() for c in compile_chunks(resume)])


if __name__ == "__main__":
    unittest.main()

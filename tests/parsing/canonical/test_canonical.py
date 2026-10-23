import json
from pathlib import Path
import unittest

from app.parsing.canonical import SCHEMA_VERSION, CanonicalResume, build_canonical_resume
from app.parsing.entities.models import EntityType as E
from app.parsing.normalization import ValidationStatus, normalize_records
from app.parsing.records import build_records
from app.parsing.sections import SectionType as S
from tests.parsing.records.test_records import ir

X, C, ED, P = S.EXPERIENCE, S.CONTACT, S.EDUCATION, S.PROJECTS
DASH = "–"


def canon(*specs):
    entities = ir(*specs)
    result = normalize_records(build_records(entities))
    return build_canonical_resume(result, "fixture.pdf", 1, 10, len(entities.entities))


class CanonicalTests(unittest.TestCase):
    def test_construction_and_schema_version(self):
        r = canon((E.NAME, "N", S.UNKNOWN, 0))
        self.assertIsInstance(r, CanonicalResume)
        self.assertEqual(r.schema_version, "1.0")
        self.assertEqual(SCHEMA_VERSION, "1.0")
        self.assertEqual((r.metadata.source_file_name, r.metadata.page_count, r.metadata.logical_block_count,
                          r.metadata.extracted_entity_count), ("fixture.pdf", 1, 10, 1))

    def test_candidate(self):
        r = canon((E.NAME, "  KUKOO  RATHEE", S.UNKNOWN, 0), (E.EMAIL, "Kukoo@Gmail.com", C, 1),
                  (E.PHONE, "8683077463", C, 1), (E.LOCATION, "Mohali Sector 71", C, 2))
        c = r.candidate
        self.assertEqual((c.name, c.email, c.phone, c.location),
                         ("KUKOO RATHEE", "kukoo@gmail.com", "8683077463", "Mohali Sector 71"))
        self.assertEqual((c.linkedin, c.github, c.website), (None, None, None))
        self.assertEqual(list(c.provenance), ["name", "email", "phone", "location"])
        self.assertEqual(c.provenance["name"].raw_value, "  KUKOO  RATHEE")
        self.assertEqual(c.provenance["email"].normalization_rules, ("email/lowercase",))

    def test_invalid_contact_value_is_none_but_raw_is_kept(self):
        r = canon((E.EMAIL, "not-an-email", C, 0))
        self.assertIsNone(r.candidate.email)
        self.assertEqual(r.candidate.provenance["email"].raw_value, "not-an-email")
        self.assertEqual(r.candidate.provenance["email"].normalization_status, "INVALID")

    def test_summary(self):
        r = canon((E.DESCRIPTION, ".NET Developer with\n  2.6+ years.", S.SUMMARY, 3))
        self.assertEqual(r.summary.value, ".NET Developer with 2.6+ years.")
        self.assertEqual(r.summary.raw_value, ".NET Developer with\n  2.6+ years.")
        self.assertEqual(r.summary.provenance.entities[0].logical_block_id, "p0/lb3")
        self.assertIsNone(canon((E.NAME, "N", S.UNKNOWN, 0)).summary)

    def test_skills_raw_and_normalized(self):
        r = canon((E.SKILL, "ASP.Net Core", S.SKILLS, 0), (E.SKILL, "SQL", S.SKILLS, 0),
                  (E.SKILL, "C#", S.SKILLS, 1), (E.SKILL, "Angular", S.SKILLS, 1))
        self.assertEqual([(s.value, s.raw_value) for s in r.skills],
                         [("ASP.NET Core", "ASP.Net Core"), ("SQL", "SQL"), ("C#", "C#"), ("Angular", "Angular")])
        self.assertEqual(r.skills[0].provenance.normalization_rules, ("skill-mapping/asp.net core",))
        self.assertNotIn("SQL Server", [s.value for s in r.skills])
        self.assertNotIn(".NET", [s.value for s in r.skills])

    def test_work_experience_in_order(self):
        r = canon((E.ROLE, "Dev", X, 0), (E.DATE_RANGE, f"July 2022 {DASH} Jan 2025", X, 0), (E.COMPANY, "A", X, 1),
                  (E.ROLE, "Lead", X, 2), (E.COMPANY, "B", X, 3), (E.DESCRIPTION, "Did  things", X, 3))
        a, b = r.work_experience
        self.assertEqual((a.company, a.role, a.start_date, a.end_date, a.current), ("A", "Dev", "2022-07", "2025-01", False))
        self.assertEqual((b.company, b.role, b.start_date, b.end_date, b.description), ("B", "Lead", None, None, "Did things"))
        self.assertEqual(a.raw_text, f"Dev\nJuly 2022 {DASH} Jan 2025\nA")
        self.assertEqual(a.logical_block_ids, ("p0/lb0", "p0/lb1"))

    def test_current_employment_has_no_artificial_end(self):
        r = canon((E.ROLE, "Dev", X, 0), (E.DATE_RANGE, f"Jan 2025{DASH} Present", X, 0), (E.COMPANY, "A", X, 1))
        w = r.work_experience[0]
        self.assertEqual((w.start_date, w.end_date, w.current), ("2025-01", None, True))
        self.assertEqual(w.provenance["date_range"].raw_value, f"Jan 2025{DASH} Present")

    def test_year_only_range_has_no_invented_months(self):
        w = canon((E.ROLE, "Dev", X, 0), (E.DATE_RANGE, "2022 - 2024", X, 0)).work_experience[0]
        self.assertEqual((w.start_date, w.end_date), ("2022", "2024"))

    def test_lone_or_bad_date_is_not_mapped(self):
        for raw in ("2018", "sometime", "2024 - 2022"):
            w = canon((E.ROLE, "Dev", X, 0), (E.DATE_RANGE, raw, X, 0)).work_experience[0]
            self.assertEqual((w.start_date, w.end_date, w.current), (None, None, False), raw)
            self.assertEqual(w.provenance["date_range"].raw_value, raw)

    def test_education(self):
        r = canon((E.DEGREE, "DIPLOMA", ED, 0), (E.FIELD_OF_STUDY, "CS", ED, 0), (E.INSTITUTION, "Poly", ED, 1),
                  (E.DEGREE, "10th", ED, 2), (E.INSTITUTION, "School", ED, 3))
        a, b = r.education
        self.assertEqual((a.institution, a.degree, a.field_of_study, a.start_date, a.end_date),
                         ("Poly", "Diploma", "CS", None, None))
        self.assertEqual(a.provenance["degree"].raw_value, "DIPLOMA")
        self.assertEqual((b.degree, b.field_of_study), ("10th", None))

    def test_education_dates_only_when_provided(self):
        e = canon((E.INSTITUTION, "Poly", ED, 0), (E.DATE_RANGE, "2015 - 2018", ED, 0)).education[0]
        self.assertEqual((e.start_date, e.end_date), ("2015", "2018"))

    def test_projects_and_empty_technologies(self):
        r = canon((E.PROJECT_NAME, "A (API)", P, 0), (E.PROJECT_DESCRIPTION, "using SQL Server\nand C#", P, 1),
                  (E.PROJECT_NAME, "B", P, 2))
        a, b = r.projects
        self.assertEqual((a.name, a.description), ("A (API)", "using SQL Server and C#"))
        self.assertEqual(a.technologies, [])
        self.assertEqual(b.technologies, [])
        self.assertNotIn("SQL Server", [t.value for t in a.technologies])

    def test_project_technologies_from_m8_only(self):
        p = canon((E.PROJECT_NAME, "A", P, 0), (E.TECHNOLOGY, "JQuery", P, 1)).projects[0]
        self.assertEqual([(t.value, t.raw_value) for t in p.technologies], [("jQuery", "JQuery")])

    def test_certification(self):
        r = canon((E.CERTIFICATION, "AZ-900", S.CERTIFICATIONS, 0), (E.ISSUING_ORGANIZATION, "Microsoft", S.CERTIFICATIONS, 0),
                  (E.CREDENTIAL_URL, "example.com/c/1", S.CERTIFICATIONS, 0))
        c = r.certifications[0]
        self.assertEqual((c.name, c.issuing_organization, c.credential_id, c.credential_url),
                         ("AZ-900", "Microsoft", None, "https://example.com/c/1"))

    def test_award_language_interest(self):
        r = canon((E.AWARD, "Awarded Best\nEmployee", S.AWARDS, 0), (E.LANGUAGE, "English", S.LANGUAGES, 1),
                  (E.LANGUAGE, "Hindi", S.LANGUAGES, 1), (E.INTEREST, "Chess", S.INTERESTS, 2))
        self.assertEqual((r.awards[0].value, r.awards[0].raw_text), ("Awarded Best Employee", "Awarded Best\nEmployee"))
        self.assertEqual(r.awards[0].provenance.raw_value, "Awarded Best\nEmployee")
        self.assertEqual([(x.value, x.raw_value) for x in r.languages], [("English", "English"), ("Hindi", "Hindi")])
        self.assertEqual([x.value for x in r.interests], ["Chess"])
        self.assertEqual(r.certifications, [])  # an award never appears as a certification

    def test_provenance_traces_to_m6(self):
        source = ir((E.SKILL, "JQuery", S.SKILLS, 7))
        entity = source.entities[0]
        r = build_canonical_resume(normalize_records(build_records(source)), "f.pdf", 1, 1, 1)
        ref = r.skills[0].provenance.entities[0]
        self.assertEqual((ref.entity_id, ref.entity_type, ref.logical_block_id, ref.page_number, ref.source_block_ids,
                          ref.sources, ref.evidence, ref.text_start, ref.text_end),
                         (entity.entity_id, E.SKILL, "p0/lb7", 0, ("p0/b7",), entity.sources, entity.evidence, 0, 6))
        self.assertEqual(r.skills[0].provenance.raw_value, "JQuery")

    def test_validation_status_propagation(self):
        r = canon((E.UNKNOWN, "DOT NET DEVELOPER", S.UNKNOWN, 1))
        self.assertEqual(r.validation.status, ValidationStatus.VALID_WITH_WARNINGS)
        self.assertEqual(r.metadata.validation_status, ValidationStatus.VALID_WITH_WARNINGS)
        self.assertIn("UNRESOLVED_ENTITY", [i.code for i in r.validation.issues])
        r = canon((E.EMAIL, "bad", C, 0))
        self.assertEqual(r.validation.status, ValidationStatus.INVALID)
        self.assertEqual(canon((E.NAME, "N", S.UNKNOWN, 0), (E.EMAIL, "a@b.com", C, 1), (E.PHONE, "8683077463", C, 1)).validation.status,
                         ValidationStatus.VALID)

    def test_unknown_entity_is_not_a_role(self):
        r = canon((E.UNKNOWN, "DOT NET DEVELOPER", S.UNKNOWN, 1))
        self.assertEqual([(u.entity_type, u.raw_value) for u in r.validation.unresolved], [(E.UNKNOWN, "DOT NET DEVELOPER")])
        self.assertEqual(r.work_experience, [])
        self.assertFalse(hasattr(r.candidate, "role"))
        self.assertNotIn("DOT NET DEVELOPER", r.model_dump_json(exclude={"validation"}))

    def test_empty_collections_and_none_singulars(self):
        r = canon((E.NAME, "N", S.UNKNOWN, 0))
        for name in ("skills", "work_experience", "education", "projects", "certifications", "awards",
                     "languages", "interests"):
            self.assertEqual(getattr(r, name), [], name)
        self.assertIsNone(r.summary)
        self.assertIsNone(r.candidate.email)

    def test_nothing_fabricated(self):
        r = canon((E.PROJECT_NAME, "A", P, 0))
        p = r.projects[0]
        self.assertEqual((p.description, p.technologies), (None, []))
        self.assertEqual(list(p.provenance), ["name"])
        self.assertEqual((r.skills, r.work_experience, r.education), ([], [], []))

    def test_deterministic_ordering_and_json(self):
        specs = ((E.NAME, "N", S.UNKNOWN, 0), (E.SKILL, "Zed", S.SKILLS, 1), (E.SKILL, "Alpha", S.SKILLS, 1),
                 (E.PROJECT_NAME, "Second?", P, 2), (E.PROJECT_NAME, "First?", P, 3))
        a, b = canon(*specs), canon(*specs)
        self.assertEqual(a.model_dump_json(), b.model_dump_json())
        self.assertEqual([s.value for s in a.skills], ["Zed", "Alpha"])  # document order, not sorted
        self.assertEqual([p.name for p in a.projects], ["Second?", "First?"])

    def test_json_serialization(self):
        r = canon((E.NAME, "N", S.UNKNOWN, 0), (E.SKILL, "C#", S.SKILLS, 1))
        data = json.loads(r.model_dump_json())
        self.assertEqual(data["schema_version"], "1.0")
        self.assertEqual(data["candidate"]["name"], "N")
        self.assertEqual(data["skills"][0]["provenance"]["entities"][0]["entity_id"], "p0/lb1/e1")
        self.assertEqual(data["metadata"]["validation_status"], "VALID")  # missing contact is INFO only
        self.assertEqual(json.loads(json.dumps(r.model_dump(mode="json"))), data)

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
        document = extract_pdf(pdf)
        layout = analyze_layout(document)
        logical = analyze_logical_blocks(document, layout, analyze_reading_order(layout))
        text = resolve_logical_text(logical, document)
        entities = extract_entities(detect_sections(logical, text), logical, text)
        result = normalize_records(build_records(entities))
        r = build_canonical_resume(result, pdf.name, len(document.pages),
                                   sum(len(p.logical_blocks) for p in logical.pages), len(entities.entities))
        self.assertEqual((r.candidate.name, r.candidate.email, r.candidate.phone, r.candidate.location),
                         ("KUKOO RATHEE", "kukoorathee@gmail.com", "8683077463", "Mohali Sector 71"))
        self.assertTrue(r.summary.value.startswith(".NET Developer with 2.6+ years"))
        self.assertEqual([len(r.work_experience), len(r.education), len(r.projects), len(r.certifications),
                          len(r.awards), len(r.skills), len(r.languages), len(r.interests)], [2, 3, 7, 0, 1, 13, 2, 0])
        self.assertEqual([(w.start_date, w.end_date, w.current) for w in r.work_experience],
                         [("2022-07", "2025-01", False), ("2025-01", None, True)])
        self.assertTrue(all(p.technologies == [] for p in r.projects))
        self.assertEqual([u.raw_value for u in r.validation.unresolved], ["DOT NET DEVELOPER"])
        self.assertEqual(r.validation.status, ValidationStatus.VALID_WITH_WARNINGS)
        self.assertEqual(r.model_dump_json(), r.model_dump_json())


if __name__ == "__main__":
    unittest.main()

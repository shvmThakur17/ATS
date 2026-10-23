from pathlib import Path
import unittest

from app.parsing.entities.models import EntityType as E
from app.parsing.normalization import (NormalizationStatus as N, Severity, ValidationStatus,
                                       normalize_records)
from app.parsing.normalization.contact_normalizer import normalize_email, normalize_phone, normalize_url
from app.parsing.normalization.date_normalizer import normalize_date_range
from app.parsing.normalization.skill_normalizer import normalize_skill
from app.parsing.normalization.text_normalizer import normalize_text
from app.parsing.records import SkillsRecord, build_records
from app.parsing.sections import SectionType as S
from tests.parsing.records.test_records import ir

X, C, ED, P = S.EXPERIENCE, S.CONTACT, S.EDUCATION, S.PROJECTS
DASH = "–"


def run(*specs):
    return normalize_records(build_records(ir(*specs)))


def codes(result, severity=None):
    return [i.code for i in result.issues if severity is None or i.severity == severity]


class TextTests(unittest.TestCase):
    def test_whitespace(self):
        v = normalize_text("  Ginlytics \n  IT Solutions   Pvt. Ltd. ")
        self.assertEqual(v.normalized_value, "Ginlytics IT Solutions Pvt. Ltd.")
        self.assertEqual(v.status, N.NORMALIZED)
        self.assertIn("text/collapse-whitespace", v.rules)

    def test_unicode_nfc_and_zero_width(self):
        v = normalize_text("Café Na​me")
        self.assertEqual(v.normalized_value, "Café Name")
        self.assertEqual(set(v.rules), {"text/unicode-nfc", "text/remove-zero-width"})

    def test_punctuation_and_case_preserved(self):
        raw = f"Baskin Robbins {DASH} Online Ordering (API), C# & .NET!"
        v = normalize_text(raw)
        self.assertEqual(v.normalized_value, raw)
        self.assertEqual(v.status, N.UNCHANGED)

    def test_raw_preserved(self):
        raw = "  A   B "
        v = normalize_text(raw)
        self.assertEqual(v.raw_value, raw)

    def test_u_circumflex_is_never_replaced_in_text(self):
        v = normalize_text("Baskin Robbins û Online")
        self.assertEqual(v.normalized_value, "Baskin Robbins û Online")
        self.assertTrue(v.warnings[0].startswith("suspicious-character"))

    def test_empty_value(self):
        v = normalize_text("   ")
        self.assertEqual((v.status, v.normalized_value), (N.UNKNOWN, None))
        self.assertEqual(v.raw_value, "   ")


class DateTests(unittest.TestCase):
    def fields(self, raw):
        d = normalize_date_range(raw)
        return (d.start_year, d.start_month, d.end_year, d.end_month, d.is_current, d.normalized_value)

    def test_month_present(self):
        self.assertEqual(self.fields("Jan 2025 - Present"), (2025, 1, None, None, True, "2025-01 / Present"))

    def test_full_month_range(self):
        self.assertEqual(self.fields("January 2022 - March 2024"), (2022, 1, 2024, 3, False, "2022-01 / 2024-03"))

    def test_year_range_does_not_invent_months(self):
        self.assertEqual(self.fields("2022 - 2024"), (2022, None, 2024, None, False, "2022 / 2024"))
        self.assertEqual(self.fields("2022-2024"), (2022, None, 2024, None, False, "2022 / 2024"))

    def test_single_year_and_formats(self):
        self.assertEqual(self.fields("2022"), (2022, None, None, None, False, "2022"))
        self.assertEqual(self.fields("01/2022"), (2022, 1, None, None, False, "2022-01"))
        self.assertEqual(self.fields("2022-01"), (2022, 1, None, None, False, "2022-01"))
        self.assertEqual(normalize_date_range("2022-01").status, N.UNCHANGED)
        self.assertEqual(self.fields("2022 - Present"), (2022, None, None, None, True, "2022 / Present"))

    def test_real_en_dash_is_a_normal_separator(self):
        d = normalize_date_range(f"July 2022 {DASH} Jan 2025")
        self.assertEqual((d.normalized_value, d.warnings), ("2022-07 / 2025-01", ()))
        self.assertIn("date-range/separator-dash", d.rules)
        d = normalize_date_range(f"Jan 2025{DASH} Present")
        self.assertEqual((d.normalized_value, d.is_current), ("2025-01 / Present", True))

    def test_corrupted_separator_between_valid_dates(self):
        for raw in ("July 2022 û Jan 2025", "Jan 2025û Present"):
            d = normalize_date_range(raw)
            self.assertEqual(d.status, N.NORMALIZED)
            self.assertEqual(d.raw_value, raw)
            self.assertIn("date-range/separator-artifact", d.rules)
            self.assertTrue(d.warnings[0].startswith("separator-artifact"))
        self.assertEqual(normalize_date_range("Jan 2025û Present").normalized_value, "2025-01 / Present")

    def test_corrupted_separator_not_assumed_elsewhere(self):
        for raw in ("Jan 2025 û", "û Present", "Backend û 2022", "Jan û Feb"):
            d = normalize_date_range(raw)
            self.assertIn(d.status, (N.UNKNOWN, N.INVALID))
            self.assertIsNone(d.normalized_value)
            self.assertEqual(d.raw_value, raw)

    def test_malformed(self):
        for raw in ("sometime in 2022", "Jan 2022 -", "Present", "Present - 2022", "Foo 2022", ""):
            d = normalize_date_range(raw)
            self.assertEqual((d.status, d.normalized_value), (N.UNKNOWN, None), raw)

    def test_invalid_month_and_year(self):
        d = normalize_date_range("13/2022")
        self.assertEqual((d.status, d.normalized_value), (N.INVALID, None))
        self.assertTrue(d.warnings[0].startswith("invalid-month"))
        self.assertEqual(normalize_date_range("2022-00").status, N.INVALID)
        self.assertEqual(normalize_date_range("1800 - 2000").status, N.INVALID)

    def test_end_before_start_not_fixed(self):
        d = normalize_date_range("March 2024 - January 2022")
        self.assertEqual((d.status, d.normalized_value), (N.INVALID, None))
        self.assertTrue(d.warnings[0].startswith("end-before-start"))
        self.assertEqual((d.start_year, d.end_year), (2024, 2022))
        self.assertEqual(normalize_date_range("Mar 2024 - Jan 2024").status, N.INVALID)
        self.assertEqual(normalize_date_range("2024 - Mar 2024").status, N.NORMALIZED)

    def test_deterministic(self):
        self.assertEqual(normalize_date_range("Jan 2025 - Present"), normalize_date_range("Jan 2025 - Present"))


class ContactTests(unittest.TestCase):
    def test_email(self):
        v = normalize_email("  Kukoo.Rathee@Gmail.COM ")
        self.assertEqual((v.normalized_value, v.status), ("kukoo.rathee@gmail.com", N.NORMALIZED))
        self.assertEqual(normalize_email("a@b.com").status, N.UNCHANGED)

    def test_invalid_email(self):
        for raw in ("not-an-email", "a@b", "a b@c.com", "a@@b.com"):
            v = normalize_email(raw)
            self.assertEqual((v.status, v.normalized_value, v.raw_value), (N.INVALID, None, raw))

    def test_phone(self):
        self.assertEqual(normalize_phone("8683077463").status, N.UNCHANGED)
        v = normalize_phone("+91 (868) 307-7463")
        self.assertEqual(v.normalized_value, "+918683077463")
        self.assertEqual(v.raw_value, "+91 (868) 307-7463")

    def test_invalid_phone(self):
        for raw in ("123", "86830abc463", "12+345678901", "1" * 16):
            self.assertEqual(normalize_phone(raw).status, N.INVALID, raw)

    def test_url(self):
        v = normalize_url("linkedin.com/in/Example")
        self.assertEqual(v.normalized_value, "https://linkedin.com/in/Example")
        self.assertEqual(normalize_url("HTTPS://GitHub.com/Example").normalized_value, "https://github.com/Example")
        self.assertEqual(normalize_url("http://example.test/a").normalized_value, "http://example.test/a")
        self.assertEqual(normalize_url("https://example.com/x").status, N.UNCHANGED)
        v = normalize_url("my portfolio")
        self.assertEqual((v.status, v.normalized_value), (N.UNKNOWN, None))

    def test_location_whitespace(self):
        result = run((E.LOCATION, "Mohali   Sector 71 ", C, 0))
        self.assertEqual(result.resume.contact.location.normalized_value, "Mohali Sector 71")
        self.assertEqual(result.resume.contact.location.raw_value, "Mohali   Sector 71 ")


class SkillTests(unittest.TestCase):
    def test_explicit_mappings(self):
        self.assertEqual(normalize_skill("ASP.Net Core").normalized_value, "ASP.NET Core")
        self.assertEqual(normalize_skill(" JQuery ").normalized_value, "jQuery")
        self.assertEqual(normalize_skill("VB NET").normalized_value, "VB.NET")
        self.assertEqual(normalize_skill("ASP.Net MVC").normalized_value, "ASP.NET MVC")
        self.assertIn("skill-mapping/asp.net core", normalize_skill("ASP.Net Core").rules)

    def test_no_semantic_inference(self):
        for raw in ("SQL", "C#", "Web APIs", "DEBUGGING", "WEB FORMS"):
            v = normalize_skill(raw)
            self.assertTrue(v.normalized_value in (raw, "SQL"), raw)
        self.assertEqual(normalize_skill("C#").normalized_value, "C#")
        self.assertEqual(normalize_skill("Web APIs").normalized_value, "Web APIs")
        self.assertEqual(normalize_skill("SQL").normalized_value, "SQL")

    def test_raw_skill_preserved(self):
        v = normalize_skill(" JQuery ")
        self.assertEqual(v.raw_value, " JQuery ")

    def test_duplicate_exact_skills(self):
        records = build_records(ir((E.NAME, "N", S.UNKNOWN, 0)))
        records = records.model_copy(update={"skills": SkillsRecord(
            raw_text="", logical_block_ids=(), entities=(), skills=("C#", "C#", "JQuery", "jquery"))})
        result = normalize_records(records)
        self.assertEqual(codes(result).count("SKILL_DUPLICATE"), 2)  # C# and jQuery
        self.assertEqual(len(result.resume.skills.skills), 4)  # nothing is merged or removed


class RecordTests(unittest.TestCase):
    def test_summary(self):
        raw = ".NET Developer with 2.6+ years\n  of experience."
        result = run((E.DESCRIPTION, raw, S.SUMMARY, 0))
        self.assertEqual(result.resume.summary.text.normalized_value, ".NET Developer with 2.6+ years of experience.")
        self.assertEqual(result.resume.summary.text.raw_value, raw)

    def test_project(self):
        result = run((E.PROJECT_NAME, f"Shop  Findy (API) {DASH} Web", P, 0),
                     (E.PROJECT_DESCRIPTION, "Built\nRESTful APIs.", P, 1))
        project = result.resume.projects[0]
        self.assertEqual(project.name.normalized_value, f"Shop Findy (API) {DASH} Web")
        self.assertEqual(project.description.normalized_value, "Built RESTful APIs.")

    def test_company_keeps_legal_suffix(self):
        result = run((E.COMPANY, "Off Beat Software  Solutions Pvt. Ltd.", X, 0))
        self.assertEqual(result.resume.work_experience[0].company.normalized_value,
                         "Off Beat Software Solutions Pvt. Ltd.")

    def test_education(self):
        result = run((E.DEGREE, "DIPLOMA", ED, 0), (E.FIELD_OF_STUDY, "CS", ED, 0),
                     (E.INSTITUTION, "Govt. Polytechnic", ED, 1),
                     (E.DEGREE, "10TH", ED, 2), (E.INSTITUTION, "School", ED, 3),
                     (E.DEGREE, "12th", ED, 4))
        a, b, c = result.resume.education
        self.assertEqual((a.degree.normalized_value, a.field_of_study.normalized_value), ("Diploma", "CS"))
        self.assertEqual(b.degree.normalized_value, "10th")
        self.assertEqual(c.degree.status, N.UNCHANGED)
        self.assertEqual(a.degree.raw_value, "DIPLOMA")

    def test_award(self):
        result = run((E.AWARD, "Awarded Best\nEmployee of  the month", S.AWARDS, 0))
        self.assertEqual(result.resume.awards[0].text.normalized_value, "Awarded Best Employee of the month")

    def test_provenance_preserved(self):
        source = ir((E.ROLE, "Dev", X, 7), (E.DATE_RANGE, f"Jan 2025{DASH} Present", X, 7), (E.COMPANY, "A", X, 8))
        result = normalize_records(build_records(source))
        record = result.resume.work_experience[0]
        date = record.date_range
        entity = source.entities[1]
        self.assertEqual(record.logical_block_ids, ("p0/lb7", "p0/lb8"))
        self.assertEqual(date.normalized_value, "2025-01 / Present")
        self.assertEqual(date.sources[0].entity_id, entity.entity_id)
        self.assertEqual((date.sources[0].logical_block_id, date.sources[0].page_number, date.sources[0].sources,
                          date.sources[0].evidence), (entity.logical_block_id, 0, entity.sources, entity.evidence))
        self.assertEqual(record.company.sources[0].raw_value, "A")

    def test_m7_records_not_mutated(self):
        records = build_records(ir((E.COMPANY, "  A  ", X, 0)))
        before = records.model_dump_json()
        normalize_records(records)
        self.assertEqual(records.model_dump_json(), before)

    def test_deterministic(self):
        records = build_records(ir((E.ROLE, "Dev", X, 0), (E.DATE_RANGE, "Jan 2022 - Present", X, 0),
                                   (E.SKILL, "JQuery", S.SKILLS, 1)))
        self.assertEqual(normalize_records(records).model_dump_json(), normalize_records(records).model_dump_json())


class ValidationTests(unittest.TestCase):
    def test_valid_resume(self):
        result = run((E.NAME, "KUKOO", S.UNKNOWN, 0), (E.EMAIL, "a@b.com", C, 1), (E.PHONE, "8683077463", C, 1),
                     (E.COMPANY, "A", X, 2), (E.ROLE, "Dev", X, 2), (E.DATE_RANGE, "Jan 2022 - Mar 2024", X, 2),
                     (E.PROJECT_NAME, "P", P, 3), (E.SKILL, "C#", S.SKILLS, 4), (E.LANGUAGE, "English", S.LANGUAGES, 5))
        self.assertEqual(result.status, ValidationStatus.VALID)
        self.assertEqual(codes(result, Severity.ERROR), [])

    def test_missing_optional_fields(self):
        result = run((E.NAME, "KUKOO", S.UNKNOWN, 0))
        self.assertNotEqual(result.status, ValidationStatus.INVALID)
        self.assertEqual(set(codes(result)), {"CONTACT_EMAIL_MISSING", "CONTACT_PHONE_MISSING"})
        self.assertTrue(all(i.severity == Severity.INFO for i in result.issues))

    def test_invalid_contact(self):
        result = run((E.EMAIL, "nope", C, 0), (E.PHONE, "12", C, 0))
        self.assertEqual(result.status, ValidationStatus.INVALID)
        self.assertIn("INVALID_EMAIL", codes(result, Severity.ERROR))
        self.assertIn("INVALID_PHONE", codes(result, Severity.ERROR))

    def test_invalid_date_range(self):
        result = run((E.ROLE, "Dev", X, 0), (E.DATE_RANGE, "2024 - 2022", X, 0))
        self.assertEqual(result.status, ValidationStatus.INVALID)
        self.assertIn("END_BEFORE_START", codes(result, Severity.ERROR))
        self.assertEqual(result.resume.work_experience[0].date_range.raw_value, "2024 - 2022")

    def test_invalid_month(self):
        result = run((E.ROLE, "Dev", X, 0), (E.DATE_RANGE, "13/2022 - 2024", X, 0))
        self.assertIn("INVALID_MONTH", codes(result, Severity.ERROR))

    def test_current_employment_is_info(self):
        result = run((E.ROLE, "Dev", X, 0), (E.COMPANY, "A", X, 0), (E.DATE_RANGE, "Jan 2025 - Present", X, 0))
        self.assertIn("CURRENT_EMPLOYMENT", codes(result, Severity.INFO))

    def test_unparseable_date_is_warning(self):
        result = run((E.ROLE, "Dev", X, 0), (E.COMPANY, "A", X, 0), (E.DATE_RANGE, "sometime", X, 0))
        self.assertIn("MALFORMED_DATE", codes(result, Severity.WARNING))

    def test_unresolved_unknown_entity(self):
        result = run((E.UNKNOWN, "DOT NET DEVELOPER", S.UNKNOWN, 1))
        issue = next(i for i in result.issues if i.code == "UNRESOLVED_ENTITY")
        self.assertEqual(issue.severity, Severity.WARNING)
        self.assertEqual(result.resume.unassigned[0].entity_type, E.UNKNOWN)
        self.assertEqual(result.resume.unassigned[0].raw_value, "DOT NET DEVELOPER")
        self.assertEqual(result.resume.work_experience, ())  # never turned into a ROLE
        self.assertNotEqual(result.status, ValidationStatus.INVALID)

    def test_empty_project_name(self):
        result = run((E.PROJECT_NAME, "   ", P, 0))
        self.assertIn("PROJECT_NAME_MISSING", codes(result, Severity.ERROR))
        self.assertEqual(result.resume.projects[0].name.raw_value, "   ")
        result = run((E.PROJECT_DESCRIPTION, "orphan description", P, 0))
        self.assertIn("PROJECT_NAME_MISSING", codes(result, Severity.ERROR))

    def test_empty_award_skill_language(self):
        result = run((E.AWARD, " ", S.AWARDS, 0), (E.SKILL, " ", S.SKILLS, 1), (E.LANGUAGE, " ", S.LANGUAGES, 2))
        self.assertEqual({"AWARD_TEXT_MISSING", "SKILL_EMPTY", "LANGUAGE_EMPTY"} - set(codes(result)), set())

    def test_education_structure(self):
        result = run((E.DEGREE, "10th", ED, 0))
        self.assertIn("EDUCATION_INSTITUTION_MISSING", codes(result, Severity.WARNING))
        result = run((E.DATE_RANGE, "2015", ED, 0))
        self.assertIn("EDUCATION_EMPTY", codes(result, Severity.ERROR))

    def test_suspicious_character_warning(self):
        result = run((E.PROJECT_NAME, "Baskin û Online", P, 0))
        self.assertIn("SUSPICIOUS_CHARACTER", codes(result, Severity.WARNING))

    def test_raw_and_normalized_remain_traceable(self):
        result = run((E.SKILL, "ASP.Net Core", S.SKILLS, 0))
        value = result.resume.skills.skills[0]
        self.assertEqual((value.raw_value, value.normalized_value, value.status), ("ASP.Net Core", "ASP.NET Core", N.NORMALIZED))
        self.assertEqual(value.sources[0].entity_id, "p0/lb0/e0")
        self.assertEqual(codes(result, Severity.ERROR), [])

    def test_silent_replacement_is_detected(self):
        from app.parsing.normalization.models import NormalizedValue
        from app.parsing.normalization.validators import check_value
        found = []
        bad = NormalizedValue(raw_value="a", normalized_value="b", status=N.UNCHANGED)
        check_value(lambda *args, **kw: found.append(args[1]), "skills", None, "skills", bad)
        self.assertIn("UNDOCUMENTED_NORMALIZATION", found)


class RealResumeTests(unittest.TestCase):
    def test_kukoo(self):
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
        records = build_records(extract_entities(detect_sections(logical, text), logical, text))
        result = normalize_records(records)
        self.assertEqual(codes(result, Severity.ERROR), [])
        dates = [w.date_range.normalized_value for w in result.resume.work_experience]
        self.assertEqual(dates, ["2022-07 / 2025-01", "2025-01 / Present"])
        skills = {v.raw_value: v.normalized_value for v in result.resume.skills.skills}
        self.assertEqual(skills["ASP.Net Core"], "ASP.NET Core")
        self.assertEqual(skills["JQuery"], "jQuery")
        self.assertEqual(skills["VB NET"], "VB.NET")
        self.assertEqual(skills["SQL"], "SQL")
        self.assertEqual([u.raw_value for u in result.resume.unassigned], ["DOT NET DEVELOPER"])
        self.assertEqual(result.resume.education[2].field_of_study.normalized_value, "CS")


if __name__ == "__main__":
    unittest.main()

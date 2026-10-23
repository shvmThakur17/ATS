from pathlib import Path
import unittest

from app.matching import (EmptyJobDescriptionError, MissingJDSectionsError, RequirementLevel as L,
                          build_query_units, MatchingConfig, parse_job_description)

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "jds" / "dotnet_backend_jd.txt"


def parse(text):
    return parse_job_description(text)


class JDParserTests(unittest.TestCase):
    def setUp(self):
        self.text = FIXTURE.read_text(encoding="utf-8")
        self.jd = parse(self.text)

    def test_fixture_structure(self):
        jd = self.jd
        self.assertEqual(jd.title, ".NET Backend Developer")
        self.assertEqual(jd.raw_text, self.text)  # raw text always preserved
        self.assertEqual([s.normalized for s in jd.required_skills],
                         ["C#", "ASP.NET Core", ".NET", "SQL Server", "REST APIs"])
        self.assertEqual([s.normalized for s in jd.preferred_skills], ["Azure", "Docker", "Angular"])
        self.assertEqual(len(jd.responsibilities), 6)
        self.assertEqual(jd.experience_requirements[0].min_years, 3)
        self.assertEqual(jd.education_requirements, ("Bachelor's degree in Computer Science or equivalent",))

    def test_required_vs_preferred_are_distinct(self):
        self.assertTrue(all(s.requirement == L.REQUIRED for s in self.jd.required_skills))
        self.assertTrue(all(s.requirement == L.PREFERRED for s in self.jd.preferred_skills))
        names = {s.normalized for s in self.jd.required_skills}
        self.assertTrue(names.isdisjoint({s.normalized for s in self.jd.preferred_skills}))

    def test_skill_normalization_reuses_m8_terminology(self):
        jd = parse("Required Skills\n- ASP.Net Core\n- JQuery\n- VB NET\n\nResponsibilities\n- Build things")
        self.assertEqual([(s.raw, s.normalized) for s in jd.required_skills],
                         [("ASP.Net Core", "ASP.NET Core"), ("JQuery", "jQuery"), ("VB NET", "VB.NET")])

    def test_unrecognized_text_is_kept(self):
        jd = self.jd
        self.assertIn("Benefits", jd.unclassified_text)
        self.assertIn("Flexible working hours", jd.unclassified_text)
        self.assertIn("About the role", jd.unclassified_text)
        self.assertNotIn("Flexible working hours", [s.normalized for s in jd.preferred_skills])

    def test_inline_heading_and_sentence_items(self):
        jd = parse("Backend Dev\nRequired Skills: C#, .NET, SQL Server\nNice to have: Docker and Kubernetes\n"
                   "Responsibilities:\n- Build APIs\nRequirements\n- Strong experience with Python, FastAPI and PostgreSQL.")
        self.assertEqual([s.normalized for s in jd.required_skills], ["C#", ".NET", "SQL Server", "Python", "FastAPI", "PostgreSQL"])
        self.assertEqual([s.normalized for s in jd.preferred_skills], ["Docker", "Kubernetes"])

    def test_skill_in_both_sections_stays_required(self):
        jd = parse("Required Skills\n- Docker\n\nPreferred Skills\n- Docker\n- Azure\n\nResponsibilities\n- Work")
        self.assertEqual([s.normalized for s in jd.required_skills], ["Docker"])
        self.assertEqual([s.normalized for s in jd.preferred_skills], ["Azure"])

    def test_years_and_technology_specific_requirement(self):
        jd = parse("Required Skills\n- ASP.NET Core\n\nExperience\n- 3-5 years of ASP.NET Core experience\n"
                   "- At least 2 years in a team\n\nResponsibilities\n- Work")
        first, second = jd.experience_requirements
        self.assertEqual((first.min_years, first.mentions_skills), (3, ("ASP.NET Core",)))
        self.assertEqual((second.min_years, second.mentions_skills), (2, ()))

    def test_education_items_not_treated_as_skills(self):
        jd = parse("Requirements\n- Bachelor's degree in Computer Science\n- C#\n\nResponsibilities\n- Work")
        self.assertEqual([s.normalized for s in jd.required_skills], ["C#"])
        self.assertEqual(len(jd.education_requirements), 1)

    def test_soft_skills_and_responsibility_skills_are_not_extracted(self):
        jd = parse("Required Skills\n- C#\n- Good communication\n- Team player\n\nResponsibilities\n- Build Docker images")
        self.assertEqual([s.normalized for s in jd.required_skills], ["C#"])
        self.assertEqual(jd.preferred_skills, ())

    def test_wrapped_lines_join_the_previous_item(self):
        jd = parse("Responsibilities\n- Design and build\n  reliable services for customers\n\nRequired Skills\n- C#")
        self.assertEqual(jd.responsibilities, ("Design and build reliable services for customers",))

    def test_deterministic_and_stable_job_id(self):
        again = parse(self.text)
        self.assertEqual(self.jd, again)
        self.assertEqual(self.jd.job_id, again.job_id)
        self.assertTrue(self.jd.job_id.startswith("jd-"))
        self.assertEqual(parse_job_description(self.text, job_id="job-7").job_id, "job-7")

    def test_invalid_jds(self):
        for text in ("", "   \n\n"):
            with self.assertRaises(EmptyJobDescriptionError):
                parse(text)
        with self.assertRaises(MissingJDSectionsError):
            parse("We are a great company with a great culture.\nJoin us.")

    def test_query_units_only_for_sections_with_content(self):
        units = build_query_units(self.jd, MatchingConfig())
        self.assertEqual([u.query_type.value for u in units],
                         ["REQUIRED_SKILLS", "PREFERRED_SKILLS", "RESPONSIBILITIES", "EXPERIENCE", "EDUCATION"])
        self.assertEqual(units[0].text, "Required skills: C#, ASP.NET Core, .NET, SQL Server, REST APIs")
        minimal = build_query_units(parse("Required Skills\n- C#\n\nResponsibilities\n- Build"), MatchingConfig())
        self.assertEqual([u.query_type.value for u in minimal], ["REQUIRED_SKILLS", "RESPONSIBILITIES"])
        self.assertEqual(units[2].chunk_types, ("WORK_EXPERIENCE", "PROJECT", "SUMMARY"))


if __name__ == "__main__":
    unittest.main()

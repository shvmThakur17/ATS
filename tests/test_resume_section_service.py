import unittest

from app.services.resume_section_service import ResumeSectionService


class ResumeSectionDetectionTests(unittest.TestCase):
    def split(self, text: str) -> dict:
        lines = ResumeSectionService._prepare_lines(text)
        return ResumeSectionService._split_sections(lines)

    def test_profile_summary_alias_is_detected(self):
        sections = self.split(
            "PROFILE SUMMARY\nBackend engineer building reliable APIs.\n"
            "WORK EXPERIENCE\nExample Ltd\nSenior Engineer"
        )

        self.assertEqual(
            sections["summary"],
            ["Backend engineer building reliable APIs."],
        )
        self.assertEqual(
            sections["work_experience"],
            ["Example Ltd", "Senior Engineer"],
        )

    def test_heading_punctuation_and_numbering_are_normalized(self):
        sections = self.split(
            "01. PROFESSIONAL SUMMARY:\nA concise summary.\n"
            "02 | TECHNICAL SKILLS\nPython, PostgreSQL"
        )

        self.assertEqual(sections["summary"], ["A concise summary."])
        self.assertEqual(sections["skills"], ["Python, PostgreSQL"])

    def test_inline_heading_preserves_content(self):
        sections = self.split(
            "Technical Skills: Python, SQL, FastAPI\n"
            "Experience: Example Ltd"
        )

        self.assertEqual(sections["skills"], ["Python, SQL, FastAPI"])
        self.assertEqual(sections["work_experience"], ["Example Ltd"])

    def test_short_uppercase_skills_do_not_reset_section(self):
        sections = self.split("SKILLS\nC#\nSQL\nGITHUB\nWEB FORMS")

        self.assertEqual(
            sections["skills"],
            ["C#", "SQL", "GITHUB", "WEB FORMS"],
        )

    def test_ignored_section_stops_relevant_section(self):
        sections = self.split(
            "SKILLS\nPython, SQL\nCERTIFICATIONS\nAWS Practitioner"
        )

        self.assertEqual(sections["skills"], ["Python, SQL"])
        self.assertEqual(sections["ignored"], ["AWS Practitioner"])

    def test_repeated_sections_are_merged(self):
        sections = self.split(
            "SKILLS\nPython\nWORK EXPERIENCE\nExample Ltd\n"
            "CORE SKILLS\nPostgreSQL"
        )

        self.assertEqual(sections["skills"], ["Python", "PostgreSQL"])


if __name__ == "__main__":
    unittest.main()

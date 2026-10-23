import unittest

from app.matching import (EvidenceKind, EvidenceStrength, MatchEvidence, MatchingConfig, QueryType,
                          RequirementLevel as L, SkillStatus as St, parse_job_description)
from app.matching.skill_matching import coverage, match_skills, mentions
from app.parsing.entities.models import EntityType as E
from app.parsing.sections import SectionType as S
from tests.parsing.canonical.test_canonical import canon

CONFIG = MatchingConfig(semantic_match_threshold=0.5, strong_evidence_threshold=0.8,
                        required_skill_match_threshold=0.5, preferred_skill_match_threshold=0.5)


def jd(required=(), preferred=()):
    text = "Required Skills\n" + "".join(f"- {s}\n" for s in required) + "\nPreferred Skills\n" + \
           "".join(f"- {s}\n" for s in preferred) + "\nResponsibilities\n- Build things\n"
    return parse_job_description(text)


def resume(skills=(), technologies=()):
    specs = [(E.NAME, "N", S.UNKNOWN, 0)]
    specs += [(E.SKILL, s, S.SKILLS, 1) for s in skills]
    if technologies:
        specs += [(E.PROJECT_NAME, "P", S.PROJECTS, 2)] + [(E.TECHNOLOGY, t, S.PROJECTS, 3) for t in technologies]
    return canon(*specs)


def chunk(text, score, chunk_id="resume-project-001", path="projects[0]", kind="PROJECT"):
    return MatchEvidence(document_id="d", namespace="resume:d", chunk_id=chunk_id, chunk_type=kind, canonical_path=path,
                         similarity_score=score, query_type=QueryType.REQUIRED_SKILLS, matched_text=text,
                         evidence_strength=EvidenceStrength.MODERATE, logical_block_ids=("p0/lb7",))


def statuses(matches):
    return {m.skill: m.status for m in matches}


class SkillMatchingTests(unittest.TestCase):
    def test_explicit_match_after_normalization(self):
        matches = match_skills(jd(["ASP.Net Core", "JQuery"]), resume(["ASP.NET Core", "jQuery"]), [], CONFIG)
        self.assertEqual(statuses(matches), {"ASP.NET Core": St.EXPLICIT_SKILL, "jQuery": St.EXPLICIT_SKILL})
        first = matches[0]
        self.assertEqual((first.raw_jd_skill, first.resume_value, first.canonical_path, first.evidence_kind),
                         ("ASP.Net Core", "ASP.NET Core", "skills[0]", EvidenceKind.SKILLS_LIST))
        self.assertTrue(first.logical_block_ids)

    def test_project_technology_is_explicit(self):
        (m,) = match_skills(jd(["Kubernetes"]), resume(technologies=["Kubernetes"]), [], CONFIG)
        self.assertEqual((m.status, m.evidence_kind, m.canonical_path),
                         (St.EXPLICIT_SKILL, EvidenceKind.PROJECT_TECHNOLOGY, "projects[0].technologies[0]"))

    def test_not_found(self):
        (m,) = match_skills(jd(["Azure"]), resume(["C#"]), [], CONFIG)
        self.assertEqual((m.status, m.evidence_kind, m.canonical_path, m.similarity_score), (St.NOT_FOUND, None, None, None))

    def test_unknown_without_canonical_resume(self):
        (m,) = match_skills(jd(["Azure"]), None, [], CONFIG)
        self.assertEqual(m.status, St.UNKNOWN)

    def test_text_mention_in_chunk_is_explicit_text_evidence(self):
        evidence = [chunk("Built services using SQL Server for data.", 0.7)]
        (m,) = match_skills(jd(["SQL Server"]), resume(["SQL"]), evidence, CONFIG)
        self.assertEqual((m.status, m.evidence_kind, m.canonical_path, m.similarity_score, m.supporting_chunk_id),
                         (St.EXPLICIT_TEXT_EVIDENCE, EvidenceKind.TEXT_MENTION, "projects[0]", 0.7, "resume-project-001"))
        self.assertTrue(m.term_in_text)
        self.assertEqual(m.logical_block_ids, ("p0/lb7",))
        self.assertEqual((m.supporting_text, m.matched_term), ("Built services using SQL Server for data.", "SQL Server"))
        self.assertIsNone(m.resume_value)

    def test_text_mention_has_no_similarity_gate(self):
        (m,) = match_skills(jd(["SQL Server"]), resume(["SQL"]), [chunk("uses SQL Server", 0.01)], CONFIG)
        self.assertEqual(m.status, St.EXPLICIT_TEXT_EVIDENCE)

    def test_canonical_text_fields_are_explicit_text_evidence_with_provenance(self):
        cases = [
            ("summary", [(E.DESCRIPTION, ".NET Developer with 2 years of experience", S.SUMMARY, 1)], ".NET", "summary"),
            ("role", [(E.ROLE, "Senior .NET Engineer", S.EXPERIENCE, 1)], ".NET", "work_experience[0].role"),
            ("work", [(E.ROLE, "Engineer", S.EXPERIENCE, 1), (E.DESCRIPTION, "Tuned SQL Server queries", S.EXPERIENCE, 1)],
             "SQL Server", "work_experience[0].description"),
            ("project", [(E.PROJECT_NAME, "Shop", S.PROJECTS, 2),
                         (E.PROJECT_DESCRIPTION, "Built REST APIs for a store", S.PROJECTS, 3)],
             "REST APIs", "projects[0].description"),
        ]
        for label, specs, skill, path in cases:
            with self.subTest(label):
                r = canon((E.NAME, "N", S.UNKNOWN, 0), *specs)
                (m,) = match_skills(jd([skill]), r, [], CONFIG)     # no retrieval at all: canonical fields suffice
                self.assertEqual((m.status, m.canonical_path), (St.EXPLICIT_TEXT_EVIDENCE, path))
                self.assertIn(skill, m.supporting_text)
                self.assertTrue(m.logical_block_ids and m.entity_ids and m.page_numbers)

    def test_rest_api_singular_counts_as_the_same_term(self):
        r = canon((E.NAME, "N", S.UNKNOWN, 0), (E.PROJECT_NAME, "Shop", S.PROJECTS, 2),
                  (E.PROJECT_DESCRIPTION, "Developed a REST API with caching", S.PROJECTS, 3))
        (m,) = match_skills(jd(["REST APIs"]), r, [], CONFIG)
        self.assertEqual((m.status, m.matched_term), (St.EXPLICIT_TEXT_EVIDENCE, "REST API"))

    def test_restful_is_the_same_term_as_rest(self):
        r = canon((E.NAME, "N", S.UNKNOWN, 0), (E.PROJECT_NAME, "Shop", S.PROJECTS, 2),
                  (E.PROJECT_DESCRIPTION, "Developed RESTful APIs in a layered architecture", S.PROJECTS, 3))
        (m,) = match_skills(jd(["REST APIs"]), r, [], CONFIG)
        self.assertEqual((m.status, m.matched_term, m.canonical_path),
                         (St.EXPLICIT_TEXT_EVIDENCE, "RESTful APIs", "projects[0].description"))

    def test_structured_skill_wins_over_text(self):
        r = canon((E.NAME, "N", S.UNKNOWN, 0), (E.DESCRIPTION, "Expert in C# and SQL", S.SUMMARY, 1),
                  (E.SKILL, "C#", S.SKILLS, 2))
        (m,) = match_skills(jd(["C#"]), r, [], CONFIG)
        self.assertEqual((m.status, m.evidence_kind), (St.EXPLICIT_SKILL, EvidenceKind.SKILLS_LIST))

    def test_semantic_match_is_partial_terminology_above_threshold(self):
        evidence = [chunk("Developed Web APIs with a layered design.", 0.7)]
        (m,) = match_skills(jd(["REST APIs"]), resume(["Web APIs"]), evidence, CONFIG)
        self.assertEqual((m.status, m.similarity_score, m.supporting_chunk_id, m.matched_term),
                         (St.SEMANTIC_MATCH, 0.7, "resume-project-001", "APIs"))
        self.assertFalse(m.term_in_text)
        self.assertEqual(m.supporting_text, "Developed Web APIs with a layered design.")
        self.assertEqual(m.logical_block_ids, ("p0/lb7",))

    def test_semantic_is_never_promoted_to_explicit(self):
        evidence = [chunk("Developed Web APIs with a layered design.", 0.99)]
        (m,) = match_skills(jd(["REST APIs"]), resume(["Web APIs"]), evidence, CONFIG)
        self.assertEqual(m.status, St.SEMANTIC_MATCH)
        self.assertIsNone(m.resume_value)
        self.assertNotIn(m.status, (St.EXPLICIT_SKILL, St.EXPLICIT_TEXT_EVIDENCE))

    def test_semantic_below_threshold_and_generic_tokens_are_not_found(self):
        (low,) = match_skills(jd(["REST APIs"]), resume(["C#"]), [chunk("Developed Web APIs", 0.3)], CONFIG)
        self.assertEqual(low.status, St.NOT_FOUND)
        (generic,) = match_skills(jd(["SQL Server"]), resume(["C#"]), [chunk("Administered Windows Server", 0.9)], CONFIG)
        self.assertEqual(generic.status, St.NOT_FOUND)
        (core,) = match_skills(jd(["ASP.NET Core"]), resume(["C#"]), [chunk("Used Entity Framework Core", 0.9)], CONFIG)
        self.assertEqual(core.status, St.NOT_FOUND)

    def test_unknown_when_resume_has_no_content(self):
        empty = canon((E.NAME, "N", S.UNKNOWN, 0))
        (m,) = match_skills(jd(["Azure"]), empty, [], CONFIG)
        self.assertEqual(m.status, St.UNKNOWN)

    def test_text_is_not_matched_inside_longer_terms(self):
        r = canon((E.NAME, "N", S.UNKNOWN, 0), (E.DESCRIPTION, "Built ASP.NET Core sites on NoSQL and Dockerfile", S.SUMMARY, 1))
        for skill in (".NET", "SQL", "Docker"):
            (m,) = match_skills(jd([skill]), r, [], CONFIG)
            self.assertEqual(m.status, St.NOT_FOUND, skill)

    def test_results_are_deterministic(self):
        evidence = [chunk("uses SQL Server", 0.7, "resume-project-002"), chunk("SQL Server DBA", 0.7, "resume-project-001")]
        runs = [match_skills(jd(["SQL Server", "Azure"]), resume(["SQL"]), list(reversed(evidence)) if i else evidence,
                             CONFIG) for i in range(2)]
        self.assertEqual(runs[0], runs[1])
        self.assertEqual(runs[0][0].supporting_chunk_id, "resume-project-001")

    def test_high_similarity_without_a_mention_proves_nothing(self):
        (m,) = match_skills(jd(["Kubernetes"]), resume(["Docker"]), [chunk("Deployed containers with Docker.", 0.95)], CONFIG)
        self.assertEqual(m.status, St.NOT_FOUND)

    def test_no_skill_is_inferred_from_another(self):
        pairs = [("SQL Server", ["SQL"]), ("SQL", ["SQL Server"]), (".NET", ["C#"]), ("C#", [".NET"]),
                 ("Angular", ["ASP.NET Core"]), ("Azure", ["AWS"]), ("Kubernetes", ["Docker", "Azure"]),
                 ("REST APIs", ["Web APIs"]), ("CS", ["Computer Science"])]
        for wanted, have in pairs:
            (m,) = match_skills(jd([wanted]), resume(have), [], CONFIG)
            self.assertEqual(m.status, St.NOT_FOUND, f"{wanted} must not be inferred from {have}")

    def test_mentions_is_whole_term(self):
        self.assertFalse(mentions(".NET", "Built with ASP.NET Core"))
        self.assertTrue(mentions(".NET", ".NET Developer with experience"))
        self.assertFalse(mentions("SQL", "NoSQL stores"))
        self.assertTrue(mentions("C#", "Programming in C#, Java"))
        self.assertFalse(mentions("C", "C# only"))

    def test_preferred_threshold_is_separate(self):
        config = MatchingConfig(required_skill_match_threshold=0.9, preferred_skill_match_threshold=0.4,
                                semantic_match_threshold=0.4, strong_evidence_threshold=0.9)
        evidence = [chunk("Built Web APIs", 0.5)]
        required = match_skills(jd(["REST APIs"]), resume(["C#"]), evidence, config)[0]
        preferred = match_skills(jd([], ["REST APIs"]), resume(["C#"]), evidence, config)[0]
        self.assertEqual((required.status, preferred.status), (St.NOT_FOUND, St.SEMANTIC_MATCH))

    def test_coverage_counts_explicit_skill_and_text_evidence_only(self):
        matches = match_skills(jd(["C#", "SQL Server", "Azure", "Docker"], ["Angular", "Git"]),
                               resume(["C#", "Git"]), [chunk("uses SQL Server", 0.9)], CONFIG)
        self.assertEqual(statuses(matches), {"C#": St.EXPLICIT_SKILL, "SQL Server": St.EXPLICIT_TEXT_EVIDENCE,
                                             "Azure": St.NOT_FOUND, "Docker": St.NOT_FOUND,
                                             "Angular": St.NOT_FOUND, "Git": St.EXPLICIT_SKILL})
        self.assertEqual(coverage(matches, L.REQUIRED), 0.5)    # C# (skill) + SQL Server (text)
        self.assertEqual(coverage(matches, L.PREFERRED), 0.5)
        self.assertIsNone(coverage([], L.REQUIRED))

    def test_missing_requirements_are_all_reported(self):
        matches = match_skills(jd(["A1", "B2", "C3"]), resume(["A1"]), [], CONFIG)
        self.assertEqual(len(matches), 3)
        self.assertEqual([m.status for m in matches], [St.EXPLICIT_SKILL, St.NOT_FOUND, St.NOT_FOUND])

    def test_explicit_match_carries_the_supporting_skills_chunk_score(self):
        evidence = [chunk("C#, HTML", 0.66, "resume-skills-001", "skills", "SKILLS")]
        (m,) = match_skills(jd(["C#"]), resume(["C#"]), evidence, CONFIG)
        self.assertEqual((m.status, m.similarity_score, m.supporting_chunk_id), (St.EXPLICIT_SKILL, 0.66, "resume-skills-001"))


if __name__ == "__main__":
    unittest.main()

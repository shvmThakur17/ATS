"""Skills-only ranking: required coverage, then preferred coverage, then semantic evidence. Nothing else ranks."""

import unittest

from app.matching import (JDMatcher, MatchingConfig, RequirementLevel as L, SkillMatch, SkillStatus as St,
                          parse_job_description)
from app.matching.ranking import assign_ranks, skill_score
from app.parsing.entities.models import EntityType as E
from app.parsing.sections import SectionType as S
from tests.parsing.matching.fakes import CountingEmbedder, build_candidate, ingest, make_store

CONFIG = MatchingConfig(top_k=10, semantic_match_threshold=0.05, strong_evidence_threshold=0.5,
                        required_skill_match_threshold=0.05, preferred_skill_match_threshold=0.05, as_of="2026-10")
X, ED, P = S.EXPERIENCE, S.EDUCATION, S.PROJECTS
DASH = "–"


def jd(required, preferred=()):
    text = "Required Skills\n" + "".join(f"- {s}\n" for s in required)
    if preferred:
        text += "\nPreferred Skills\n" + "".join(f"- {s}\n" for s in preferred)
    return parse_job_description(text + "\nResponsibilities\n- Build software\n")


def spec(name, skills=(), description=None, degree="B.Tech", dates="Jan 2020 - Jan 2024", summary=None):
    specs = [(E.NAME, name, S.UNKNOWN, 0)]
    if summary:
        specs.append((E.DESCRIPTION, summary, S.SUMMARY, 1))
    specs += [(E.ROLE, "Engineer", X, 2), (E.DATE_RANGE, dates, X, 2), (E.COMPANY, "Co", X, 3)]
    if description:
        specs += [(E.PROJECT_NAME, "Proj", P, 4), (E.PROJECT_DESCRIPTION, description, P, 5)]
    specs += [(E.DEGREE, degree, ED, 6), (E.INSTITUTION, "Uni", ED, 7)]
    specs += [(E.SKILL, skill, S.SKILLS, 8) for skill in skills]
    return tuple(specs)


def rank(job, candidates, with_resumes=True):
    """candidates: {document_id: specs}. Returns {document_id: CandidateMatch} and the ordered ids."""
    store, resumes = make_store(), {}
    for document_id, specs in candidates.items():
        resume, _, records = build_candidate(document_id, f"{document_id}.pdf", specs)
        ingest(store, records)
        resumes[document_id] = resume
    result = JDMatcher(store, CountingEmbedder(), CONFIG).match(job, resumes if with_resumes else None)
    return {c.document_id: c for c in result.candidates}, [c.document_id for c in result.candidates]


def match(skill, status, level=L.REQUIRED):
    return SkillMatch(skill=skill, raw_jd_skill=skill, requirement=level, status=status)


class SkillScoreFormulaTests(unittest.TestCase):
    def test_key_is_the_documented_formula(self):
        matches = [match("a", St.EXPLICIT_SKILL), match("b", St.EXPLICIT_TEXT_EVIDENCE), match("c", St.SEMANTIC_MATCH),
                   match("d", St.NOT_FOUND), match("p", St.EXPLICIT_SKILL, L.PREFERRED),
                   match("q", St.UNKNOWN, L.PREFERRED)]
        score = skill_score(matches)
        self.assertEqual((score.required_total, score.required_explicit, score.required_semantic), (4, 2, 1))
        self.assertEqual((score.preferred_total, score.preferred_explicit, score.preferred_semantic), (2, 1, 0))
        self.assertEqual(score.key, (0.5, 0.5, 0.25, 0.0))

    def test_only_explicit_statuses_count_as_covered(self):
        for status, covered in ((St.EXPLICIT_SKILL, 1), (St.EXPLICIT_TEXT_EVIDENCE, 1), (St.SEMANTIC_MATCH, 0),
                                (St.NOT_FOUND, 0), (St.UNKNOWN, 0)):
            self.assertEqual(skill_score([match("a", status)]).required_explicit, covered, status)

    def test_levels_absent_from_the_jd_are_none_and_zero_in_the_key(self):
        score = skill_score([match("a", St.EXPLICIT_SKILL)])
        self.assertIsNone(score.preferred_coverage)
        self.assertEqual(score.key, (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(skill_score([]).key, (0.0, 0.0, 0.0, 0.0))

    def test_competition_ranks_share_ties(self):
        a, b, c = (skill_score([match("a", s)]) for s in (St.EXPLICIT_SKILL, St.EXPLICIT_SKILL, St.NOT_FOUND))
        self.assertEqual(assign_ranks([a, b, c]), [1, 1, 3])


class SkillRankingTests(unittest.TestCase):
    def test_more_required_coverage_ranks_higher(self):
        by, order = rank(jd(["C#", "SQL", "Docker"]), {
            "low": spec("L", ["C#"]), "high": spec("H", ["C#", "SQL", "Docker"]), "mid": spec("M", ["C#", "SQL"])})
        self.assertEqual(order, ["high", "mid", "low"])
        self.assertEqual([by[d].rank for d in order], [1, 2, 3])
        self.assertEqual([by[d].skill_score.required_coverage for d in order], [1.0, 2 / 3, 1 / 3])

    def test_preferred_skills_are_secondary(self):
        job = jd(["C#", "SQL"], ["Docker", "Azure"])
        by, order = rank(job, {
            "more_preferred": spec("A", ["C#", "SQL", "Docker", "Azure"]),
            "less_preferred": spec("B", ["C#", "SQL", "Docker"]),
            "no_preferred": spec("C", ["C#", "SQL"])})
        self.assertEqual(order, ["more_preferred", "less_preferred", "no_preferred"])
        self.assertEqual({d: by[d].skill_score.required_coverage for d in order}, {d: 1.0 for d in order})

    def test_preferred_never_outweighs_required(self):
        job = jd(["C#", "SQL"], ["Docker", "Azure", "Angular"])
        by, order = rank(job, {"required_only": spec("A", ["C#", "SQL"]),
                               "all_preferred": spec("B", ["C#", "Docker", "Azure", "Angular"])})
        self.assertEqual(order, ["required_only", "all_preferred"])

    def test_jd_without_preferred_skills_ranks_on_required_alone(self):
        by, order = rank(jd(["C#", "SQL"]), {"a": spec("A", ["C#", "SQL"]), "b": spec("B", ["C#"])})
        self.assertEqual(order, ["a", "b"])
        self.assertIsNone(by["a"].skill_score.preferred_coverage)
        self.assertEqual(by["a"].skill_score.key[1], 0.0)

    def test_missing_ambiguous_and_related_skills_are_not_confirmed(self):
        job = jd(["SQL", "Java", "ASP.NET Core"])
        by, _ = rank(job, {"c": spec("C", ["SQL Server", "JavaScript", "Entity Framework Core"])})
        statuses = {m.skill: m.status for m in by["c"].skill_matches}
        self.assertEqual(statuses, {"SQL": St.NOT_FOUND, "Java": St.NOT_FOUND, "ASP.NET Core": St.NOT_FOUND})
        self.assertEqual(by["c"].skill_score.key, (0.0, 0.0, 0.0, 0.0))

    def test_unknown_skill_evidence_counts_as_nothing(self):
        by, _ = rank(jd(["Azure"]), {"c": spec("C", ["C#"])}, with_resumes=False)
        (m,) = by["c"].skill_matches
        self.assertEqual(m.status, St.UNKNOWN)
        self.assertEqual(by["c"].skill_score.required_explicit, 0)

    def test_explicit_beats_semantic_only(self):
        job = jd(["REST APIs"])
        by, order = rank(job, {"semantic": spec("S", ["Web APIs"]), "explicit": spec("E", ["REST APIs"])})
        self.assertEqual(order, ["explicit", "semantic"])
        self.assertEqual(by["semantic"].skill_matches[0].status, St.SEMANTIC_MATCH)
        self.assertEqual(by["semantic"].skill_score.required_explicit, 0)
        self.assertEqual(by["semantic"].skill_score.required_semantic_share, 1.0)

    def test_semantic_evidence_only_separates_otherwise_identical_candidates(self):
        job = jd(["C#", "REST APIs"])
        by, order = rank(job, {"related": spec("A", ["C#", "Web APIs"]), "plain": spec("B", ["C#"])})
        self.assertEqual(by["related"].skill_score.key[:2], by["plain"].skill_score.key[:2])
        self.assertEqual(order, ["related", "plain"])
        self.assertEqual([by[d].rank for d in order], [1, 2])

    def test_structured_and_text_evidence_rank_the_same(self):
        job = jd(["C#", "SQL Server"])
        by, order = rank(job, {
            "structured": spec("A", ["C#", "SQL Server"]),
            "stated_in_text": spec("B", ["C#"], description="Built reports on SQL Server")})
        self.assertEqual(by["stated_in_text"].skill_matches[1].status, St.EXPLICIT_TEXT_EVIDENCE)
        self.assertEqual(by["structured"].skill_score, by["stated_in_text"].skill_score)
        self.assertEqual([by[d].rank for d in order], [1, 1])

    def test_education_and_experience_do_not_change_the_ranking(self):
        job = jd(["C#", "SQL"])      # the JD asks for neither, but the resumes differ a lot
        candidates = {
            "senior_degree": spec("A", ["C#", "SQL"], degree="M.Tech", dates="Jan 2010 - Jan 2024",
                                  summary="Engineer with 14 years of experience"),
            "junior_diploma": spec("B", ["C#", "SQL"], degree="Diploma", dates="Jan 2025 - Jul 2025")}
        by, order = rank(job, candidates)
        self.assertEqual(by["senior_degree"].skill_score, by["junior_diploma"].skill_score)
        self.assertEqual([by[d].rank for d in order], [1, 1])
        self.assertEqual(order, sorted(order))                      # listing order is only the document_id tie-break
        job_with = parse_job_description("Required Skills\n- C#\n- SQL\n\nResponsibilities\n- Build\n\n"
                                         "Experience\n- 5+ years of experience\n\nEducation\n- Bachelor's degree")
        by2, _ = rank(job_with, candidates)
        self.assertEqual(by2["senior_degree"].experience.status.value != by2["junior_diploma"].experience.status.value,
                         True)                                       # the evidence differs ...
        self.assertEqual([by2[d].rank for d in by2], [1, 1])         # ... the ranking does not

    def test_responsibility_text_and_similarity_do_not_rank(self):
        job = jd(["C#"])
        by, _ = rank(job, {
            "relevant_text": spec("A", ["C#"], description="Build software software software"),
            "unrelated_text": spec("B", ["C#"], description="Cooked pasta")})
        self.assertEqual(by["relevant_text"].skill_score, by["unrelated_text"].skill_score)
        self.assertEqual(by["relevant_text"].rank, by["unrelated_text"].rank)

    def test_retrieval_scores_and_depth_do_not_change_the_key(self):
        from dataclasses import replace
        job = jd(["C#", "SQL"], ["Docker"])
        candidates = {"a": spec("A", ["C#", "SQL"], description="Build software"), "b": spec("B", ["C#", "SQL"])}
        baseline, _ = rank(job, candidates)
        for config in (replace(CONFIG, top_k=1), replace(CONFIG, top_k=50, semantic_match_threshold=0.01,
                                                        strong_evidence_threshold=0.9)):
            store, resumes = make_store(), {}
            for document_id, specs in candidates.items():
                resume, _, records = build_candidate(document_id, f"{document_id}.pdf", specs)
                ingest(store, records)
                resumes[document_id] = resume
            result = JDMatcher(store, CountingEmbedder(), config).match(job, resumes)
            for c in result.candidates:
                self.assertEqual(c.skill_score.key, baseline[c.document_id].skill_score.key)
                self.assertEqual(c.rank, 1)

    def test_ranking_module_depends_only_on_skill_statuses(self):
        import inspect
        import app.matching.ranking as ranking
        source = inspect.getsource(ranking)
        for forbidden in ("scoring", "retrieval", "experience", "education", "similarity_score", "evidence_strength"):
            code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("#", '"')))
            code = code.split('"""')[0] + code.split('"""')[-1] if '"""' in code else code
            self.assertNotIn(forbidden, code.split("def share")[1], forbidden)
        self.assertEqual(list(inspect.signature(ranking.skill_score).parameters), ["matches"])

    def test_same_skill_profile_gets_the_same_score_and_rank(self):
        job = jd(["C#", "SQL"], ["Docker"])
        by, order = rank(job, {"z": spec("Z", ["C#", "Docker"]), "a": spec("A", ["Docker", "C#"]),
                               "m": spec("M", ["C#", "Docker", "HTML"])})
        self.assertEqual(len({by[d].skill_score.key for d in order}), 1)
        self.assertEqual([by[d].rank for d in order], [1, 1, 1])
        self.assertEqual(order, ["a", "m", "z"])

    def test_result_is_deterministic(self):
        job = jd(["C#", "SQL"], ["Docker"])
        candidates = {"a": spec("A", ["C#"]), "b": spec("B", ["C#", "SQL"]), "c": spec("C", ["Docker"])}
        first = rank(job, candidates)[1]
        second = rank(job, dict(reversed(list(candidates.items()))))[1]
        self.assertEqual(first, second)

    def test_provenance_survives_in_the_ranked_matches(self):
        by, _ = rank(jd(["C#", "SQL Server"]), {"c": spec("C", ["C#"], description="Used SQL Server daily")})
        structured, text = by["c"].skill_matches
        self.assertEqual((structured.status, structured.canonical_path), (St.EXPLICIT_SKILL, "skills[0]"))
        self.assertTrue(structured.logical_block_ids)
        self.assertEqual((text.status, text.canonical_path), (St.EXPLICIT_TEXT_EVIDENCE, "projects[0].description"))
        self.assertEqual(text.supporting_text, "Used SQL Server daily")
        self.assertTrue(text.logical_block_ids and text.entity_ids and text.page_numbers)


if __name__ == "__main__":
    unittest.main()

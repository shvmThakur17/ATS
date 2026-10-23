import unittest

from app.matching import (EvidenceStrength, InvalidConfigError, MatchEvidence, MatchingConfig, QueryType,
                          EducationStatus as Ed, EducationStatus, ExperienceStatus, SkillStatus, SkillStatus as St, parse_job_description)
from app.matching.scoring import education_match, experience_match, month_index, retrieval_stats, strength
from app.parsing.entities.models import EntityType as E
from app.parsing.sections import SectionType as S
from tests.parsing.canonical.test_canonical import canon

X, ED = S.EXPERIENCE, S.EDUCATION
DASH = "–"


def ev(chunk_id, score, query_type=QueryType.REQUIRED_SKILLS):
    return MatchEvidence(document_id="d", namespace="resume:d", chunk_id=chunk_id, chunk_type="PROJECT",
                         canonical_path="projects[0]", similarity_score=score, query_type=query_type, matched_text="t",
                         evidence_strength=EvidenceStrength.MODERATE)


def work(*ranges):
    specs = []
    for i, raw in enumerate(ranges):
        specs += [(E.ROLE, f"Dev{i}", X, 2 * i), (E.DATE_RANGE, raw, X, 2 * i), (E.COMPANY, f"Co{i}", X, 2 * i + 1)]
    return canon(*specs)


def jd(extra):
    return parse_job_description("Required Skills\n- C#\n\n" + extra + "\n\nResponsibilities\n- Build things")


class ConfigTests(unittest.TestCase):
    def test_defaults_are_documented_and_valid(self):
        config = MatchingConfig()
        self.assertEqual((config.top_k, config.semantic_match_threshold), (10, 0.60))
        for removed in ("weights", "missing_component_policy", "relevance_top_n"):
            self.assertFalse(hasattr(config, removed), removed)     # ranking is skills-only: nothing to weight

    def test_invalid_config_is_rejected(self):
        for kwargs in ({"top_k": 0}, {"semantic_match_threshold": 1.5}, 
                       {"strong_evidence_threshold": 0.1, "semantic_match_threshold": 0.5}, {"as_of": "2025"}):
            with self.assertRaises(InvalidConfigError):
                MatchingConfig(**kwargs)

    def test_from_env(self):
        import os
        old = {k: os.environ.get(k) for k in ("M13_TOP_K", "M13_AS_OF")}
        try:
            os.environ.update(M13_TOP_K="4", M13_AS_OF="2026-10")
            config = MatchingConfig.from_env()
            self.assertEqual((config.top_k, config.as_of), (4, "2026-10"))
        finally:
            for k, v in old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


class ScoringTests(unittest.TestCase):
    def test_strength_bands(self):
        config = MatchingConfig(semantic_match_threshold=0.5, strong_evidence_threshold=0.8)
        self.assertEqual([strength(s, config) for s in (0.9, 0.8, 0.6, 0.5, 0.4)],
                         [EvidenceStrength.STRONG, EvidenceStrength.STRONG, EvidenceStrength.MODERATE,
                          EvidenceStrength.MODERATE, EvidenceStrength.WEAK])

    def test_retrieval_stats(self):
        config = MatchingConfig(top_k=2, semantic_match_threshold=0.5, strong_evidence_threshold=0.8)
        evidence = [ev("a", 0.9), ev("a", 0.6, QueryType.RESPONSIBILITIES), ev("b", 0.7), ev("c", 0.2)]
        stats = retrieval_stats(evidence, config)
        self.assertEqual((stats.retrieval_count, stats.unique_chunk_count, stats.top_score), (4, 3, 0.9))
        self.assertAlmostEqual(stats.average_top_k_score, 0.8)      # mean of best a=0.9 and b=0.7
        self.assertEqual(stats.relevant_chunk_count, 2)
        self.assertEqual(stats.per_query_counts, {"REQUIRED_SKILLS": 3, "RESPONSIBILITIES": 1})
        empty = retrieval_stats([], config)
        self.assertEqual((empty.top_score, empty.average_top_k_score, empty.relevant_chunk_count), (None, None, 0))

    def test_month_index(self):
        self.assertEqual(month_index("2025-01") - month_index("2022-07"), 30)
        self.assertIsNone(month_index("2022"))
        self.assertIsNone(month_index(None))


class ExperienceTests(unittest.TestCase):
    def test_overlapping_roles_are_merged(self):
        match = experience_match(jd("Experience\n- 3+ years of backend experience"),
                                 work(f"July 2022 {DASH} Jan 2025", f"Jan 2025{DASH} Present"),
                                 MatchingConfig(as_of="2026-10"))
        self.assertEqual((match.candidate_months, match.meets, match.required_years), (51, True, 3))
        self.assertIn("not technology-specific", match.note)

    def test_current_role_without_as_of_is_excluded_not_guessed(self):
        match = experience_match(jd("Experience\n- 3+ years"), work(f"Jan 2024{DASH} Present"), MatchingConfig())
        self.assertEqual((match.candidate_months, match.meets), (0, None))
        self.assertIn("work_experience[0]", match.note)

    def test_short_experience_does_not_meet(self):
        match = experience_match(jd("Experience\n- 5+ years"), work("Jan 2020 - Jan 2022"), MatchingConfig())
        self.assertEqual((match.candidate_months, match.meets), (24, False))

    def test_technology_specific_years_are_flagged_not_inferred(self):
        match = experience_match(jd("Experience\n- 3+ years of C# experience"), work("Jan 2015 - Jan 2022"), MatchingConfig())
        self.assertTrue(match.meets)
        self.assertIn("technology-specific duration is not inferred", match.note)

    def test_no_requirement_or_no_resume(self):
        self.assertIsNone(experience_match(jd(""), work("Jan 2020 - Jan 2022"), MatchingConfig()))
        match = experience_match(jd("Experience\n- 3+ years"), None, MatchingConfig())
        self.assertEqual((match.candidate_months, match.meets), (None, None))

    def test_year_only_dates_are_not_used(self):
        match = experience_match(jd("Experience\n- 3+ years"), work("2015 - 2022"), MatchingConfig())
        self.assertEqual(match.meets, None)


class EducationTests(unittest.TestCase):
    def resume(self, *degrees):
        return canon(*[(E.DEGREE, d, ED, i) for i, d in enumerate(degrees)])

    def test_required_bachelor_vs_diploma_is_not_found_and_equivalent_not_evaluated(self):
        match = education_match(jd("Education\n- Bachelor's degree in Computer Science or equivalent"),
                                self.resume("10th", "DIPLOMA"))
        self.assertEqual((match.required_levels, match.candidate_levels, match.status),
                         (("bachelor",), ("diploma",), Ed.NOT_FOUND))
        self.assertIn("equivalence is not evaluated", match.note)

    def test_explicit_bachelor_matches(self):
        match = education_match(jd("Education\n- Bachelor's degree"), self.resume("B.Tech"))
        self.assertEqual((match.candidate_levels, match.status), (("bachelor",), Ed.EXPLICIT_MATCH))

    def test_generic_degree_accepts_any_degree_level_but_not_diploma(self):
        self.assertEqual(education_match(jd("Education\n- A degree"), self.resume("M.Tech")).status, Ed.EXPLICIT_MATCH)
        self.assertEqual(education_match(jd("Education\n- A degree"), self.resume("Diploma")).status, Ed.NOT_FOUND)

    def test_degrees_are_not_inferred(self):
        match = education_match(jd("Education\n- Master's degree"), self.resume("B.Tech"))
        self.assertEqual(match.status, Ed.NOT_FOUND)

    def test_unknown_cases(self):
        self.assertIsNone(education_match(jd(""), self.resume("B.Tech")))
        self.assertEqual(education_match(jd("Education\n- Bachelor's degree"), None).status, Ed.UNKNOWN)
        self.assertEqual(education_match(jd("Education\n- Bachelor's degree"), canon((E.NAME, "N", S.UNKNOWN, 0))).status,
                         Ed.UNKNOWN)


class StatedExperienceTests(unittest.TestCase):
    REQ = "Experience\n- 3+ years of experience"

    def resume(self, summary, *ranges):
        specs = [(E.NAME, "N", S.UNKNOWN, 0), (E.DESCRIPTION, summary, S.SUMMARY, 1)]
        for i, raw in enumerate(ranges):
            specs += [(E.ROLE, f"Dev{i}", X, 2 * i + 2), (E.DATE_RANGE, raw, X, 2 * i + 2),
                      (E.COMPANY, f"Co{i}", X, 2 * i + 3)]
        return canon(*specs)

    def match(self, summary, *ranges, req=REQ):
        return experience_match(jd(req), self.resume(summary, *ranges), MatchingConfig(as_of="2026-10"))

    def test_both_figures_are_preserved_and_conflict_is_flagged(self):
        m = self.match(".NET Developer with 2.6+ years of experience", "Jul 2022 - Oct 2026")
        self.assertEqual((m.stated_text, m.stated_months, m.candidate_months), ("2.6+ years", 31, 51))
        self.assertTrue(m.discrepancy)
        self.assertIn("neither is treated as authoritative", m.discrepancy_note)
        self.assertIn("2.6+ years", m.discrepancy_note)
        self.assertIn("51 months", m.discrepancy_note)
        self.assertEqual((m.meets, m.stated_meets), (True, False))   # each figure judged on its own, none chosen

    def test_stated_claim_keeps_its_source_and_provenance(self):
        m = self.match(".NET Developer with 2.6+ years of experience", "Jul 2022 - Oct 2026")
        self.assertEqual(m.stated_source_path, "summary")
        self.assertEqual(m.stated_source_text, ".NET Developer with 2.6+ years of experience")
        self.assertTrue(m.logical_block_ids and m.entity_ids and m.page_numbers)

    def test_no_discrepancy_when_figures_agree(self):
        m = self.match("Engineer with 4 years of experience", "Jul 2022 - Oct 2026")
        self.assertEqual((m.stated_months, m.candidate_months, m.discrepancy, m.discrepancy_note), (48, 51, False, None))

    def test_no_stated_claim_means_no_stated_fields(self):
        m = self.match("Engineer", "Jul 2022 - Oct 2026")
        self.assertEqual((m.stated_text, m.stated_months, m.stated_meets, m.discrepancy), (None, None, None, False))

    def test_cannot_compare_without_dates(self):
        m = self.match("Engineer with 2 years of experience")
        self.assertEqual((m.stated_months, m.candidate_months, m.discrepancy), (24, None, False))

    def test_experience_is_never_technology_specific(self):
        m = self.match(".NET Developer with 5 years", "Jan 2020 - Jan 2024", req="Experience\n- 3+ years of experience with C#")
        self.assertIn("technology-specific duration is not inferred", m.note)
        self.assertEqual(m.candidate_months, 48)


class EducationEvidenceTests(unittest.TestCase):
    def resume(self, degree, field="CS"):
        return canon((E.NAME, "N", S.UNKNOWN, 0), (E.DEGREE, degree, ED, 1), (E.FIELD_OF_STUDY, field, ED, 1),
                     (E.INSTITUTION, "Govt. Polytechnic", ED, 2))

    def test_status_type_is_education_specific(self):
        values = {s.value for s in EducationStatus}
        self.assertEqual(values, {"EXPLICIT_MATCH", "NOT_FOUND", "UNKNOWN"})
        self.assertNotIn("EXPLICIT_SKILL", values)
        for resume in (self.resume("B.Tech"), self.resume("Diploma"), None):
            self.assertIsInstance(education_match(jd("Education\n- Bachelor's degree"), resume).status, EducationStatus)

    def test_three_categories(self):
        requirement = jd("Education\n- Bachelor's degree")
        self.assertEqual(education_match(requirement, self.resume("B.Tech")).status, EducationStatus.EXPLICIT_MATCH)
        self.assertEqual(education_match(requirement, self.resume("Diploma")).status, EducationStatus.NOT_FOUND)
        self.assertEqual(education_match(requirement, None).status, EducationStatus.UNKNOWN)

    def test_degree_evidence_and_provenance_are_preserved(self):
        match = education_match(jd("Education\n- Bachelor's degree"), self.resume("Diploma"))
        (evidence,) = match.evidence
        self.assertEqual((evidence.canonical_path, evidence.degree, evidence.field_of_study, evidence.institution),
                         ("education[0]", "Diploma", "CS", "Govt. Polytechnic"))
        self.assertEqual(evidence.levels, ("diploma",))
        self.assertTrue(evidence.raw_text and evidence.logical_block_ids and evidence.entity_ids and evidence.page_numbers)

    def test_matching_degree_evidence_is_kept_too(self):
        match = education_match(jd("Education\n- Bachelor's degree"), self.resume("B.Tech"))
        self.assertEqual(match.evidence[0].levels, ("bachelor",))
        self.assertEqual(match.evidence[0].degree, "B.Tech")

    def test_diploma_never_satisfies_a_bachelor_requirement(self):
        self.assertEqual(education_match(jd("Education\n- Bachelor's degree"), self.resume("Diploma")).status,
                         EducationStatus.NOT_FOUND)


class ExperienceStatusTests(StatedExperienceTests):
    def test_conflicting_verdicts_are_unknown_not_met(self):
        m = self.match("Engineer with 2.6+ years of experience", "Jul 2022 - Oct 2026")   # dates: 51 months; claim: 31
        self.assertEqual((m.meets, m.stated_meets, m.status), (True, False, ExperienceStatus.UNKNOWN))
        self.assertIn("disagree", m.note)

    def test_agreeing_verdicts_are_met_even_with_a_discrepancy(self):
        m = self.match("Engineer with 9 years of experience", "Jul 2022 - Oct 2026")
        self.assertTrue(m.discrepancy)
        self.assertEqual(m.status, ExperienceStatus.MET)

    def test_not_met_and_unknown(self):
        short = self.match("Engineer with 1 year of experience", "Jan 2025 - Oct 2026")
        self.assertEqual(short.status, ExperienceStatus.NOT_MET)
        self.assertEqual(self.match("Engineer").status, ExperienceStatus.UNKNOWN)

    def test_both_values_survive_the_conflict(self):
        m = self.match("Engineer with 2.6+ years of experience", "Jul 2022 - Oct 2026")
        self.assertEqual((m.stated_months, m.candidate_months), (31, 51))


class EducationAuditTests(unittest.TestCase):
    def resume(self, degree):
        return canon((E.NAME, "N", S.UNKNOWN, 0), (E.DEGREE, degree, ED, 1), (E.INSTITUTION, "Uni", ED, 2))

    def test_btech_matches_a_bachelor_requirement(self):
        for degree in ("B.Tech", "BCA", "Bachelor of Technology"):
            match = education_match(jd("Education\n- Bachelor's degree in Computer Science"), self.resume(degree))
            self.assertEqual((match.status, match.candidate_levels), (Ed.EXPLICIT_MATCH, ("bachelor",)), degree)
            self.assertEqual(match.evidence[0].degree, self.resume(degree).education[0].degree)

    def test_diploma_is_not_equivalent_even_when_the_jd_mentions_equivalent(self):
        match = education_match(jd("Education\n- Bachelor's degree or equivalent"), self.resume("Diploma"))
        self.assertEqual(match.status, Ed.NOT_FOUND)
        self.assertIn("equivalence is not evaluated", match.note)

    def test_education_states_are_not_skill_states(self):
        match = education_match(jd("Education\n- Bachelor's degree"), self.resume("B.Tech"))
        self.assertIsInstance(match.status, EducationStatus)
        self.assertNotIsInstance(match.status, SkillStatus)
        self.assertNotEqual(match.status.value, SkillStatus.EXPLICIT_SKILL.value)


if __name__ == "__main__":
    unittest.main()

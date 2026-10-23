import unittest

from app.matching import (EvidenceStrength, InvalidConfigError, MatchEvidence, MatchingConfig, QueryType,
                          EducationStatus as Ed, EducationStatus, ExperienceStatus, SkillStatus, SkillStatus as St, parse_job_description)
from app.matching.models import DEFAULT_WEIGHTS, ExperienceMatch
from app.matching.scoring import (build_components, not_applicable_components, education_match, experience_match, experience_component, relevance, retrieval_stats,
                                  strength, weighted,
                                  month_index)
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
        self.assertAlmostEqual(sum(config.weights.values()), 1.0)
        # similarity components are deliberately a minority of the weight
        self.assertLess(config.weights["semantic_relevance"] + config.weights["responsibility_relevance"], 0.5)
        self.assertGreater(config.weights["required_skill_coverage"], config.weights["semantic_relevance"])

    def test_invalid_config_is_rejected(self):
        for kwargs in ({"top_k": 0}, {"semantic_match_threshold": 1.5}, {"weights": {"x": 1.0}},
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

    def test_relevance_components(self):
        config = MatchingConfig(relevance_top_n=2)
        evidence = [ev("a", 0.9), ev("b", 0.5), ev("c", 0.1), ev("r", 0.7, QueryType.RESPONSIBILITIES)]
        self.assertAlmostEqual(relevance(evidence, config), 0.8)
        self.assertAlmostEqual(relevance(evidence, config, QueryType.RESPONSIBILITIES), 0.7)
        self.assertIsNone(relevance([], config))
        self.assertIsNone(relevance(evidence, config, QueryType.EDUCATION))
        self.assertEqual(relevance([ev("n", -0.4)], config), 0.0)   # negative cosine is clamped

    def test_weighted_renormalizes_over_present_components(self):
        weights = {"a": 0.5, "b": 0.3, "c": 0.2}
        self.assertAlmostEqual(weighted({"a": 1.0, "b": 0.0, "c": 0.5}, weights, ("a", "b", "c")), 0.6)
        self.assertAlmostEqual(weighted({"a": 1.0, "b": None, "c": 0.5}, weights, ("a", "b", "c")), (0.5 + 0.1) / 0.7)
        self.assertIsNone(weighted({"a": None}, weights, ("a",)))
        self.assertEqual(weighted({"a": 0.0}, weights, ("a",)), 0.0)

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

    def test_conflicting_verdicts_make_the_component_undetermined(self):
        conflict = self.match("Engineer with 2 years of experience", "Jul 2022 - Oct 2026")
        agree = self.match("Engineer with 4 years of experience", "Jul 2022 - Oct 2026")
        self.assertIsNone(experience_component(conflict))
        self.assertEqual(experience_component(agree), 1.0)

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


class MissingComponentPolicyTests(unittest.TestCase):
    W = {"a": 0.5, "b": 0.3, "c": 0.2}

    def test_zero_policy_unavailable_cannot_raise_the_score(self):
        components = {"a": 1.0, "b": None, "c": 1.0}
        self.assertAlmostEqual(weighted(components, self.W, ("a", "b", "c"), "zero"), 0.7)
        self.assertAlmostEqual(weighted(components, self.W, ("a", "b", "c"), "renormalize"), 1.0)
        self.assertLess(weighted(components, self.W, ("a", "b", "c"), "zero"),
                        weighted({"a": 1.0, "b": 1.0, "c": 1.0}, self.W, ("a", "b", "c"), "zero"))

    def test_not_applicable_is_excluded_under_both_policies(self):
        components = {"a": 1.0, "b": None, "c": 0.5}
        for policy in ("zero", "renormalize"):
            self.assertAlmostEqual(weighted(components, self.W, ("a", "b", "c"), policy, {"b"}), (0.5 + 0.1) / 0.7)

    def test_real_zero_is_counted_and_all_missing_is_none(self):
        self.assertEqual(weighted({"a": 0.0, "b": 0.0, "c": 0.0}, self.W, ("a", "b", "c"), "zero"), 0.0)
        self.assertIsNone(weighted({"a": None, "b": None, "c": None}, self.W, ("a", "b", "c"), "zero"))
        self.assertIsNone(weighted({"a": None, "b": None, "c": None}, self.W, ("a", "b", "c"), "zero", {"a", "b", "c"}))

    def test_a_zero_and_a_missing_component_are_not_confused(self):
        zero = weighted({"a": 1.0, "b": 0.0, "c": 1.0}, self.W, ("a", "b", "c"), "zero")
        missing = weighted({"a": 1.0, "b": None, "c": 1.0}, self.W, ("a", "b", "c"), "zero")
        self.assertEqual(zero, missing)          # conservative policy: unknown is treated like no evidence

    def test_invalid_policy_is_rejected(self):
        with self.assertRaises(InvalidConfigError):
            MatchingConfig(missing_component_policy="ignore")

    def test_not_applicable_detection(self):
        self.assertEqual(not_applicable_components(None, 0.5, None, None),
                         {"required_skill_coverage", "experience_evidence", "education_evidence"})
        self.assertEqual(not_applicable_components(1.0, 0.5, object(), object()), frozenset())


class ExperienceStatusTests(StatedExperienceTests):
    def test_conflicting_verdicts_are_unknown_not_met(self):
        m = self.match("Engineer with 2.6+ years of experience", "Jul 2022 - Oct 2026")   # dates: 51 months; claim: 31
        self.assertEqual((m.meets, m.stated_meets, m.status), (True, False, ExperienceStatus.UNKNOWN))
        self.assertIn("disagree", m.note)
        self.assertIsNone(experience_component(m))

    def test_agreeing_verdicts_are_met_even_with_a_discrepancy(self):
        m = self.match("Engineer with 9 years of experience", "Jul 2022 - Oct 2026")
        self.assertTrue(m.discrepancy)
        self.assertEqual(m.status, ExperienceStatus.MET)

    def test_not_met_and_unknown(self):
        short = self.match("Engineer with 1 year of experience", "Jan 2025 - Oct 2026")
        self.assertEqual(short.status, ExperienceStatus.NOT_MET)
        self.assertEqual(experience_component(short), 0.0)
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


class ScoringPolicyTests(unittest.TestCase):
    NAMES = tuple(DEFAULT_WEIGHTS)
    BASES = (
        {n: 1.0 for n in NAMES},
        {n: 0.5 for n in NAMES},
        {"required_skill_coverage": 0.8, "preferred_skill_coverage": 0.0, "semantic_relevance": 0.75,
         "responsibility_relevance": 0.69, "experience_evidence": 1.0, "education_evidence": 0.0},
    )

    def test_exact_default_weights(self):
        self.assertEqual(DEFAULT_WEIGHTS, {
            "required_skill_coverage": 0.40, "preferred_skill_coverage": 0.10, "semantic_relevance": 0.15,
            "responsibility_relevance": 0.15, "experience_evidence": 0.10, "education_evidence": 0.10})
        self.assertEqual(MatchingConfig().weights, DEFAULT_WEIGHTS)
        self.assertAlmostEqual(sum(DEFAULT_WEIGHTS.values()), 1.0)

    def test_default_policy_is_the_conservative_one(self):
        self.assertEqual(MatchingConfig().missing_component_policy, "zero")

    def test_omitting_a_component_never_beats_supplying_it_at_zero(self):
        # Under the documented default policy "unavailable" is exactly "no evidence" (0), for every component.
        for base in self.BASES:
            for name in self.NAMES:
                with self.subTest(name=name, base=base[name]):
                    missing = dict(base, **{name: None})
                    zero = dict(base, **{name: 0.0})
                    one = dict(base, **{name: 1.0})
                    got = weighted(missing, DEFAULT_WEIGHTS, self.NAMES, "zero")
                    self.assertAlmostEqual(got, weighted(zero, DEFAULT_WEIGHTS, self.NAMES, "zero"))
                    self.assertLessEqual(got, weighted(one, DEFAULT_WEIGHTS, self.NAMES, "zero") + 1e-12)

    def test_omission_raises_the_score_only_under_the_explicit_renormalize_policy(self):
        base = dict(self.BASES[2], education_evidence=None)        # a known-bad component made unavailable
        full = dict(self.BASES[2])
        self.assertGreater(weighted(base, DEFAULT_WEIGHTS, self.NAMES, "renormalize"),
                           weighted(full, DEFAULT_WEIGHTS, self.NAMES, "renormalize"))
        self.assertAlmostEqual(weighted(base, DEFAULT_WEIGHTS, self.NAMES, "zero"),
                               weighted(full, DEFAULT_WEIGHTS, self.NAMES, "zero"))

    def test_default_call_matches_the_documented_formula(self):
        components = self.BASES[2]
        expected = sum(DEFAULT_WEIGHTS[n] * components[n] for n in self.NAMES)   # weights sum to 1
        self.assertAlmostEqual(weighted(components, DEFAULT_WEIGHTS, self.NAMES, "zero"), expected)
        self.assertAlmostEqual(expected, 0.4 * 0.8 + 0.15 * 0.75 + 0.15 * 0.69 + 0.1)

    def test_not_applicable_never_lowers_or_raises_the_score(self):
        components = dict(self.BASES[2], preferred_skill_coverage=None)
        na = {"preferred_skill_coverage"}
        without = {n: w for n, w in DEFAULT_WEIGHTS.items() if n not in na}
        reference = sum(without[n] * components[n] for n in without) / sum(without.values())
        for policy in ("zero", "renormalize"):
            self.assertAlmostEqual(weighted(components, DEFAULT_WEIGHTS, self.NAMES, policy, na), reference)

    def test_sub_scores_follow_the_same_policy(self):
        components = {"semantic_relevance": 0.8, "responsibility_relevance": None}
        pair = ("semantic_relevance", "responsibility_relevance")
        self.assertAlmostEqual(weighted(components, DEFAULT_WEIGHTS, pair, "zero"), 0.4)
        self.assertAlmostEqual(weighted(components, DEFAULT_WEIGHTS, pair, "renormalize"), 0.8)

    def test_unknown_statuses_map_to_unavailable_components(self):
        unknown_edu = education_match(jd("Education\n- Bachelor's degree"), None)
        self.assertEqual(unknown_edu.status, EducationStatus.UNKNOWN)
        components = build_components(1.0, None, [], None, unknown_edu, MatchingConfig())
        self.assertIsNone(components["education_evidence"])
        self.assertIsNone(components["experience_evidence"])
        self.assertIsNone(experience_component(ExperienceMatch(requirement_text="t", required_years=3,
                                                               candidate_months=None, meets=None, note="")))


if __name__ == "__main__":
    unittest.main()

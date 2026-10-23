
import unittest

from app.services.m14_scoring import calculate_match_score


def skills(covered_flags):
    return [{"covered": value} for value in covered_flags]


class M14ScoringTests(unittest.TestCase):
    def test_required_4_of_5_and_preferred_3_of_5_scores_76(self):
        result = calculate_match_score(
            required_skills=skills([True, True, True, True, False]),
            preferred_skills=skills([True, True, True, False, False]),
        )

        self.assertEqual(result.required_points, 64.0)
        self.assertEqual(result.preferred_points, 12.0)
        self.assertEqual(result.overall_score, 76.0)

    def test_required_5_of_5_and_preferred_1_of_5_scores_84(self):
        result = calculate_match_score(
            required_skills=skills([True] * 5),
            preferred_skills=skills([True, False, False, False, False]),
        )

        self.assertEqual(result.overall_score, 84.0)

    def test_semantic_only_evidence_does_not_count_as_confirmed(self):
        result = calculate_match_score(
            required_skills=[
                {"covered": False, "status": "SEMANTIC_MATCH"},
                {"covered": True, "status": "EXPLICIT_SKILL"},
            ],
            preferred_skills=[],
        )

        self.assertEqual(result.required_matched, 1)
        self.assertEqual(result.overall_score, 50.0)

    def test_no_skills_returns_unavailable_score(self):
        result = calculate_match_score([], [])

        self.assertIsNone(result.overall_score)
        self.assertIsNone(result.required_points)
        self.assertIsNone(result.preferred_points)

    def test_only_required_skills_uses_full_100_point_scale(self):
        result = calculate_match_score(
            required_skills=skills([True, True, False, False]),
            preferred_skills=[],
        )

        self.assertEqual(result.overall_score, 50.0)

    def test_only_preferred_skills_uses_full_100_point_scale(self):
        result = calculate_match_score(
            required_skills=[],
            preferred_skills=skills([True, True, False, False]),
        )

        self.assertEqual(result.overall_score, 50.0)


if __name__ == "__main__":
    unittest.main()

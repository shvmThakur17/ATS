
import unittest
from unittest.mock import patch

from app.services.ai.m14_reasoning_service import M14ReasoningService


class M14ReasoningTests(unittest.TestCase):
    def analyze(self):
        return M14ReasoningService.analyze(
            job_description="Python developer",
            required_skills=[
                {
                    "skill": "Python",
                    "status": "EXPLICIT_SKILL",
                    "covered": True,
                }
            ],
            preferred_skills=[],
            missing_required=[],
            missing_preferred=[],
        )

    @patch("app.services.ai.m14_reasoning_service.OllamaService.analyze")
    def test_valid_response_is_parsed(self, mock_analyze):
        mock_analyze.return_value = {
            "strengths": ["Python appears in the skills list"],
            "gaps": [],
            "experience_summary": "",
            "recommendation": "Good Match",
        }

        result = self.analyze()

        self.assertEqual(result.status, "AVAILABLE")
        self.assertEqual(result.strengths, ["Python appears in the skills list"])
        self.assertEqual(result.recommendation, "Good Match")

    @patch("app.services.ai.m14_reasoning_service.OllamaService.analyze")
    def test_invalid_response_returns_unavailable(self, mock_analyze):
        mock_analyze.return_value = {
            "match_score": 95,
            "strengths": [],
            "gaps": [],
            "experience_summary": "",
            "recommendation": "Strong Match",
        }

        result = self.analyze()

        self.assertEqual(result.status, "UNAVAILABLE")
        self.assertEqual(result.recommendation, "Insufficient Evidence")

    @patch(
        "app.services.ai.m14_reasoning_service.OllamaService.analyze",
        side_effect=RuntimeError("simulated outage"),
    )
    def test_ollama_failure_returns_unavailable(self, mock_analyze):
        result = self.analyze()

        self.assertEqual(result.status, "UNAVAILABLE")
        self.assertEqual(result.strengths, [])
        self.assertEqual(result.gaps, [])


if __name__ == "__main__":
    unittest.main()

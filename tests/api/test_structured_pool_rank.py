import unittest
from unittest.mock import patch

from pydantic import ValidationError

from app.api.v1.jd_pool import build_structured_jd
from app.schemas.m14_reasoning import CandidateReasoning
from app.schemas.pool_api import PoolRankRequest, StructuredJD
from tests.api.test_pool_rank import client_for, pool_service
from tests.parsing.matching.test_skill_ranking import spec


class StructuredPoolRankTests(unittest.TestCase):
    def structured(self, **overrides):
        return {
            "role": "Backend Developer",
            "required_skills": ["C#", "SQL"],
            "preferred_skills": ["Docker"],
            "experience": {"minimum_years": 2.5, "maximum_years": 6},
            "education": {"required": True, "description": "BSc"},
            "responsibilities": ["Build APIs"],
            "additional_preferences": ["Azure experience"],
            **overrides,
        }

    def client(self):
        return client_for(pool_service({
            "a": spec("A", ["C#", "SQL", "Docker"]),
            "b": spec("B", ["C#"]),
            "c": spec("C", ["Python"]),
        }))

    def test_schema_accepts_structured_only_and_rejects_empty_requests(self):
        request = PoolRankRequest.model_validate({"structured_jd": self.structured()})
        self.assertEqual(request.job_description, "")
        self.assertFalse(request.include_reasoning)
        for payload in ({}, {"job_description": " "}, {"job_description": "x" * 20001}):
            with self.subTest(payload_length=len(payload.get("job_description", ""))):
                with self.assertRaises(ValidationError):
                    PoolRankRequest.model_validate(payload)

    def test_maps_experience_and_education_without_truncating_fractional_years(self):
        structured = StructuredJD.model_validate(self.structured())
        jd = build_structured_jd(structured, "")
        self.assertEqual(jd.title, structured.role)
        self.assertEqual(jd.education_requirements, ("BSc",))
        self.assertEqual(jd.responsibilities, ("Build APIs",))
        self.assertEqual(jd.qualifications, ("Azure experience",))
        self.assertIn("2.5 years", jd.experience_requirements[0].text)
        self.assertIn("6 years", jd.experience_requirements[0].text)
        self.assertIsNone(jd.experience_requirements[0].min_years)
        self.assertIn("Education (required): BSc", jd.raw_text)
        structured.experience.minimum_years = 3.0
        other = build_structured_jd(structured, "")
        self.assertEqual(other.experience_requirements[0].min_years, 3)
        self.assertNotEqual(jd.job_id, other.job_id)

    @patch("app.api.v1.jd_pool.M14ReasoningService.analyze")
    def test_structured_only_search_keeps_categories_scores_and_zero_score_candidates(self, analyze):
        response = self.client().post("/api/v1/jd/rank", json={"structured_jd": self.structured()})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["jd"]["required_skills"], ["C#", "SQL"])
        self.assertEqual(body["jd"]["preferred_skills"], ["Docker"])
        self.assertEqual([c["overall_match_score"] for c in body["candidates"]], [100, 40, 0])
        self.assertEqual(body["pool"]["ranked"], 3)
        self.assertTrue(all(c["reasoning"]["status"] == "NOT_REQUESTED" for c in body["candidates"]))
        analyze.assert_not_called()

    def test_single_skill_category_is_normalized_to_full_score(self):
        for category in ("required_skills", "preferred_skills"):
            with self.subTest(category=category):
                structured = self.structured(required_skills=[], preferred_skills=[])
                structured[category] = ["C#"]
                response = self.client().post("/api/v1/jd/rank", json={"structured_jd": structured})
                self.assertEqual(response.status_code, 200, response.text)
                scores = [c["overall_match_score"] for c in response.json()["candidates"]]
                self.assertEqual(scores, [100, 100, 0])

    @patch("app.api.v1.jd_pool.M14ReasoningService.analyze")
    def test_reasoning_receives_structured_context_without_changing_rank_or_score(self, analyze):
        analyze.return_value = CandidateReasoning(status="AVAILABLE", strengths=["Confirmed skills"])
        client = self.client()
        payload = {"structured_jd": self.structured(), "job_description": "Required skills: Python"}
        baseline = client.post("/api/v1/jd/rank", json=payload).json()
        response = client.post("/api/v1/jd/rank", json={**payload, "include_reasoning": True})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["jd"]["required_skills"], ["C#", "SQL"])
        fields = ("document_id", "rank", "ranking_key", "overall_match_score")
        self.assertEqual(
            [[candidate[field] for field in fields] for candidate in baseline["candidates"]],
            [[candidate[field] for field in fields] for candidate in result["candidates"]],
        )
        self.assertIn("2.5 years", analyze.call_args.kwargs["job_description"])
        self.assertIn("Education (required): BSc", analyze.call_args.kwargs["job_description"])
        self.assertEqual(result["candidates"][0]["reasoning"]["status"], "AVAILABLE")

    @patch("app.api.v1.jd_pool.M14ReasoningService.analyze", side_effect=RuntimeError("offline"))
    def test_reasoning_failure_does_not_fail_ranking(self, analyze):
        response = self.client().post("/api/v1/jd/rank", json={
            "structured_jd": self.structured(), "include_reasoning": True,
        })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["pool"]["ranked"], 3)
        self.assertTrue(all(c["reasoning"]["status"] == "UNAVAILABLE" for c in response.json()["candidates"]))

    def test_invalid_structured_fields_are_rejected(self):
        client = self.client()
        for structured in (
            self.structured(required_skills=[], preferred_skills=[]),
            self.structured(experience={"minimum_years": 5, "maximum_years": 2}),
            self.structured(experience={"minimum_years": -1}),
            self.structured(education={"description": "x" * 501}),
        ):
            with self.subTest(structured=structured):
                response = client.post("/api/v1/jd/rank", json={"structured_jd": structured})
                self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()

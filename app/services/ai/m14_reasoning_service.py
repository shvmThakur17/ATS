
"""M14 Ollama reasoning service.

Ollama provides qualitative explanations only. Python owns match scoring,
and M13 owns ranking order.
"""

import logging

from app.core.config import settings
from app.schemas.m14_reasoning import CandidateReasoning
from app.services.ai.m14_prompt_builder import M14PromptBuilder
from app.services.ai.ollama_service import OllamaService

logger = logging.getLogger("app.api")


class M14ReasoningService:
    @staticmethod
    def analyze(job_description: str,required_skills: list[dict],preferred_skills: list[dict],missing_required: list[str],missing_preferred: list[str],) -> CandidateReasoning:
        prompt = M14PromptBuilder.build(job_description=job_description,required_skills=required_skills,preferred_skills=preferred_skills,missing_required=missing_required,missing_preferred=missing_preferred,)
        try:
            raw = OllamaService.analyze(prompt)
            result = CandidateReasoning.model_validate(raw)
            return result
        except Exception as exc:
            # Log the error type only. Never log resume content or the raw response.
            logger.warning("M14 reasoning unavailable type=%s", type(exc).__name__)
            return CandidateReasoning(status="UNAVAILABLE",recommendation="Insufficient Evidence",)

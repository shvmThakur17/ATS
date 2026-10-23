from pydantic import BaseModel

from app.schemas.candidate_response import CandidateResponse
from app.services.ai.ai_response_parser import AIAnalysisResponse


class JobSearchResponse(BaseModel):
    candidate: CandidateResponse
    analysis: AIAnalysisResponse
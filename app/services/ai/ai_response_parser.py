from pydantic import BaseModel


class AIAnalysisResponse(BaseModel):
    match_score: int
    strengths: list[str]
    missing_skills: list[str]
    experience_summary: str
    recommendation: str
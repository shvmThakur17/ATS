from pydantic import BaseModel

class CandidateResponse(BaseModel):
    file_name: str
    average_score: float
    best_score: float
    matched_chunks: int
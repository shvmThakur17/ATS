from fastapi import APIRouter
from app.schemas.job_search_request import JobSearchRequest
from app.services.job_search_service import JobSearchService

router = APIRouter()
@router.post("/search")
async def search_candidates(request: JobSearchRequest):
    return JobSearchService.search_candidates(request.job_description)
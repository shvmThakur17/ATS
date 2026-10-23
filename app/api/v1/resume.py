from pathlib import Path
from fastapi import APIRouter, UploadFile, File
from app.services.resume_parser import ResumeParser
from app.services.resume_section_service import ResumeSectionService
from app.utils.text_cleaner import clean_text
from app.schemas.normalized_resume_response import (
    NormalizedResumeResponse
)


router = APIRouter()

UPLOAD_FOLDER = "uploads"


@router.post(
    "/upload",
    response_model=NormalizedResumeResponse
)
async def upload_resume(file: UploadFile = File(...)):

    file_path = Path(UPLOAD_FOLDER) / file.filename

    with open(file_path, "wb") as buffer:buffer.write(await file.read())
    resume_text = ResumeParser.parse(str(file_path))
    resume_text = clean_text(resume_text)
    normalized_resume = (ResumeSectionService.extract_relevant_content(resume_text))

    return {
        "status": "processed",
        "file_name": file.filename,
        "resume": normalized_resume
    }
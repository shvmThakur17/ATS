from pydantic import BaseModel, Field


class ProjectItem(BaseModel):
    name: str = ""
    description: str = ""
    raw_text: str = ""


class WorkExperienceItem(BaseModel):
    company: str = ""
    role: str = ""
    description: str = ""
    raw_text: str = ""


class NormalizedResume(BaseModel):
    summary: str = ""
    skills: list[str] = Field(default_factory=list)
    projects: list[ProjectItem] = Field(default_factory=list)
    work_experience: list[WorkExperienceItem] = Field(
        default_factory=list
    )


class NormalizedResumeResponse(BaseModel):
    status: str
    file_name: str
    resume: NormalizedResume
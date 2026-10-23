
"""Request and response models for shared-pool JD ranking."""

from typing import Literal

from pydantic import Field, model_validator

from app.schemas.m14_reasoning import CandidateReasoning
from .pipeline_api import (
    MAX_JD_CHARS,
    CandidateResult,
    JdSummary,
    RetrievalSummary,
    StrictRequest,
)


class ExperienceRequirement(StrictRequest):
    minimum_years: float | None = Field(default=None, ge=0, le=60)
    maximum_years: float | None = Field(default=None, ge=0, le=60)

    @model_validator(mode="after")
    def validate_range(self):
        if (
            self.minimum_years is not None
            and self.maximum_years is not None
            and self.minimum_years > self.maximum_years
        ):
            raise ValueError("minimum_years cannot exceed maximum_years")
        return self


class EducationRequirement(StrictRequest):
    required: bool = False
    description: str | None = Field(default=None, max_length=500)


class StructuredJD(StrictRequest):
    role: str = Field(min_length=1, max_length=200)
    experience: ExperienceRequirement = Field(
        default_factory=ExperienceRequirement
    )
    required_skills: list[str] = Field(default_factory=list, max_length=100)
    preferred_skills: list[str] = Field(default_factory=list, max_length=100)
    education: EducationRequirement = Field(
        default_factory=EducationRequirement
    )
    responsibilities: list[str] = Field(default_factory=list, max_length=50)
    additional_preferences: list[str] = Field(
        default_factory=list, max_length=50
    )

    @model_validator(mode="after")
    def validate_skills(self):
        self.required_skills = list(
            dict.fromkeys(skill.strip() for skill in self.required_skills if skill.strip())
        )
        self.preferred_skills = list(
            dict.fromkeys(skill.strip() for skill in self.preferred_skills if skill.strip())
        )

        required_lower = {skill.casefold() for skill in self.required_skills}
        self.preferred_skills = [
            skill for skill in self.preferred_skills
            if skill.casefold() not in required_lower
        ]

        if not self.required_skills and not self.preferred_skills:
            raise ValueError("At least one required or preferred skill is needed")

        return self


class PoolRankRequest(StrictRequest):
    # Keep existing free-text requests working.
    job_description: str = Field(default="", max_length=MAX_JD_CHARS)

    # Optional structured JD. When supplied, it takes precedence for skill extraction.
    structured_jd: StructuredJD | None = None
    include_reasoning: bool = False

    @model_validator(mode="after")
    def validate_description(self):
        if self.structured_jd is None and not self.job_description.strip():
            raise ValueError("Provide a job description or structured JD")
        return self


class PoolCandidateResult(CandidateResult):
    source_file: str | None = Field(
        default=None,
        description="Original PDF file name, when the index holds one",
    )
    required_skill_points: float | None = Field(
        default=None,
        description="Confirmed required-skill points",
    )
    preferred_skill_points: float | None = Field(
        default=None,
        description="Confirmed preferred-skill points",
    )
    overall_match_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Separate deterministic M14 score from 0 to 100",
    )
    reasoning: CandidateReasoning = Field(
        default_factory=lambda: CandidateReasoning(status="NOT_REQUESTED"),
        description="Qualitative explanation; not used for ranking or scoring",
    )


class SkippedResume(StrictRequest):
    document_id: str
    reason: Literal["EMPTY", "STALE", "TOO_LARGE"] = Field(
        description="Why a resume was not ranked",
    )


class PoolSummary(StrictRequest):
    discovered: int
    considered: int
    ranked: int
    skipped: int
    truncated: bool
    max_considered: int


class PoolRankResponse(StrictRequest):
    ranking_basis: Literal["skills-only"] = "skills-only"
    jd: JdSummary
    retrieval: RetrievalSummary
    pool: PoolSummary
    skipped: list[SkippedResume]
    candidates: list[PoolCandidateResult]

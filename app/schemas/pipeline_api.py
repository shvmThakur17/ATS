"""Request and response models for the N3 resume-ingestion and skills-matching endpoints."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
MAX_JD_CHARS = 20_000      # single source for the schema and the request-size cap
MAX_CANDIDATES = 100


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")        # unknown fields (for example `namespace`) are rejected, not ignored


class ResumeUploadResponse(BaseModel):
    status: Literal["INGESTED", "PENDING_INDEXING"]
    ready: bool = Field(description="False while the vectors are not yet visible in the index: wait, then match")
    document_id: str
    namespace: str
    chunk_count: int
    vectors_upserted: int
    page_count: int
    replaced_existing: bool | None = Field(description="True when this exact PDF had been ingested before")
    embedding_model: str
    embedding_dimension: int
    access_token: str = Field(description="Capability for this resume; required to match it. Treat as a secret.")


class CandidateReference(StrictRequest):
    document_id: str
    access_token: str

    @field_validator("document_id", "access_token")
    @classmethod
    def must_be_sha256_hex(cls, value: str) -> str:
        if not HEX_SHA256.match(value):
            raise ValueError("must be 64 lowercase hex characters")
        return value


class JdMatchRequest(StrictRequest):
    job_description: str = Field(min_length=1, max_length=MAX_JD_CHARS)
    candidates: list[CandidateReference] = Field(min_length=1, max_length=MAX_CANDIDATES)


class SkillEvidence(BaseModel):
    kind: str | None = None                 # SKILLS_LIST / PROJECT_TECHNOLOGY / TEXT_MENTION
    canonical_path: str | None = None
    matched_term: str | None = None
    excerpt: str | None = None              # short, whitespace-collapsed, emails/phones redacted
    page_numbers: list[str] = []
    chunk_id: str | None = None


class SkillResult(BaseModel):
    skill: str
    requirement: Literal["REQUIRED", "PREFERRED"]
    status: str                             # M13 status, unchanged
    covered: bool                           # EXPLICIT_SKILL or EXPLICIT_TEXT_EVIDENCE: the only statuses that count
    evidence: SkillEvidence | None = None


class MissingSkills(BaseModel):
    required: list[str]
    preferred: list[str]


class CandidateResult(BaseModel):
    rank: int
    document_id: str
    required_skill_coverage: float | None
    preferred_skill_coverage: float | None
    ranking_key: list[float] = Field(description="M13 lexicographic key: required, preferred, required-semantic, "
                                                 "preferred-semantic. Larger is better.")
    required_skills: list[SkillResult]
    preferred_skills: list[SkillResult]
    missing_skills: MissingSkills           # not covered: NOT_FOUND, UNKNOWN or SEMANTIC_MATCH only
    warnings: list[str]


class JdSummary(BaseModel):
    job_id: str
    title: str | None
    required_skills: list[str]
    preferred_skills: list[str]


class RetrievalSummary(BaseModel):
    effective_top_k: int
    max_retrieval_depth: int


class JdMatchResponse(BaseModel):
    ranking_basis: Literal["skills-only"] = "skills-only"
    jd: JdSummary
    retrieval: RetrievalSummary
    candidates: list[CandidateResult]

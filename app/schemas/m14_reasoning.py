
"""Validated output schema for M14 candidate reasoning.

Ollama provides explanations only. Python remains responsible for scoring
and M13 remains responsible for candidate ranking.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CandidateReasoning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["AVAILABLE", "UNAVAILABLE", "NOT_REQUESTED"] = "AVAILABLE"
    strengths: list[str] = Field(default_factory=list, max_length=3)
    gaps: list[str] = Field(default_factory=list, max_length=3)
    experience_summary: str = Field(default="", max_length=500)
    recommendation: Literal[
        "Strong Match",
        "Good Match",
        "Average Match",
        "Weak Match",
        "Insufficient Evidence",
    ] = "Insufficient Evidence"

    @field_validator("strengths", "gaps")
    @classmethod
    def validate_items(cls, values: list[str]) -> list[str]:
        cleaned = []
        for value in values:
            item = value.strip()
            if not item:
                raise ValueError("Items must not be empty")
            if len(item) > 300:
                raise ValueError("Items must not exceed 300 characters")
            cleaned.append(item)
        return cleaned

    @field_validator("experience_summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        return value.strip()


"""M14 deterministic overall match score.

This score is separate from M13 ranking and must never change candidate order.
Only confirmed skill evidence contributes to coverage.
"""

from dataclasses import dataclass
from typing import Any, Sequence


REQUIRED_WEIGHT = 80.0
PREFERRED_WEIGHT = 20.0


@dataclass(frozen=True)
class MatchScore:
    required_matched: int
    required_total: int
    required_points: float | None
    preferred_matched: int
    preferred_total: int
    preferred_points: float | None
    overall_score: float | None


def _confirmed_count(skills: Sequence[Any]) -> int:
    """Count only confirmed skill results; semantic matches do not count."""
    return sum(
        1
        for skill in skills
        if (
            skill.get("covered", False)
            if isinstance(skill, dict)
            else getattr(skill, "covered", False)
        )
    )


def calculate_match_score(required_skills: Sequence[Any],preferred_skills: Sequence[Any],
) -> MatchScore:
    """Calculate an overall score out of 100 using 80 required / 20 preferred."""
    required_total = len(required_skills)
    preferred_total = len(preferred_skills)

    required_matched = _confirmed_count(required_skills)
    preferred_matched = _confirmed_count(preferred_skills)

    required_points = (round(required_matched / required_total * REQUIRED_WEIGHT, 2)
        if required_total
        else None
    )
    preferred_points = (
        round(preferred_matched / preferred_total * PREFERRED_WEIGHT, 2)
        if preferred_total
        else None
    )

    if not required_total and not preferred_total:
        overall_score = None
    elif not required_total:
        # If the JD has only preferred skills, use the available preferred
        # coverage as the full score rather than capping the candidate at 20.
        overall_score = round(preferred_matched / preferred_total * 100, 2)
    elif not preferred_total:
        # If the JD has only required skills, required coverage is the full score.
        overall_score = round(required_matched / required_total * 100, 2)
    else:
        overall_score = round(required_points + preferred_points, 2)

    abv = MatchScore(
        required_matched=required_matched,
        required_total=required_total,
        required_points=required_points,
        preferred_matched=preferred_matched,
        preferred_total=preferred_total,
        preferred_points=preferred_points,
        overall_score=overall_score,
    )
    return abv
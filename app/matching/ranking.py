"""Skills-only candidate ranking.

Inputs are the per-skill statuses from skill_matching.py and nothing else: experience, education,
responsibilities, retrieval similarity and resume completeness are not ranking criteria.

Per candidate, with R = required skills in the JD and P = preferred skills in the JD:
    covered(skill)   = status in {EXPLICIT_SKILL, EXPLICIT_TEXT_EVIDENCE}     (evidence of the skill itself)
    semantic(skill)  = status == SEMANTIC_MATCH                               (related wording only, NOT covered)
    NOT_FOUND / UNKNOWN count as neither: missing evidence is never a confirmed match.

    key = ( |covered R| / |R|,  |covered P| / |P|,  |semantic R| / |R|,  |semantic P| / |P| )

Candidates are ordered by `key` compared lexicographically (larger first). A level with no skills in the JD
contributes 0.0 for every candidate. Why lexicographic and not a weighted sum: it needs no weights, so there is nothing
arbitrary to justify. Required coverage always dominates; preferred skills only separate candidates whose required
coverage is identical; semantic evidence only separates candidates who are otherwise identical. Ties on the full
key share a rank (competition ranking: 1, 1, 3); document_id only fixes the listing order.
"""

from .models import RequirementLevel, SkillMatch, SkillScore, SkillStatus

COVERED = (SkillStatus.EXPLICIT_SKILL, SkillStatus.EXPLICIT_TEXT_EVIDENCE)


def share(count: int, total: int) -> float | None:
    return count / total if total else None


def skill_score(matches: list[SkillMatch]) -> SkillScore:
    counts = {}
    for level in RequirementLevel:
        pool = [m for m in matches if m.requirement == level]
        counts[level] = (len(pool), sum(m.status in COVERED for m in pool),
                         sum(m.status == SkillStatus.SEMANTIC_MATCH for m in pool))
    (r_total, r_explicit, r_semantic) = counts[RequirementLevel.REQUIRED]
    (p_total, p_explicit, p_semantic) = counts[RequirementLevel.PREFERRED]
    values = (share(r_explicit, r_total), share(p_explicit, p_total), share(r_semantic, r_total),
              share(p_semantic, p_total))
    return SkillScore(
        required_total=r_total, required_explicit=r_explicit, required_semantic=r_semantic,
        preferred_total=p_total, preferred_explicit=p_explicit, preferred_semantic=p_semantic,
        required_coverage=values[0], preferred_coverage=values[1], required_semantic_share=values[2],
        preferred_semantic_share=values[3], key=tuple(v or 0.0 for v in values))


def assign_ranks(scores: list[SkillScore]) -> list[int]:
    """Competition ranks for scores already sorted best-first."""
    ranks = []
    for i, score in enumerate(scores):
        ranks.append(ranks[-1] if i and score.key == scores[i - 1].key else i + 1)
    return ranks

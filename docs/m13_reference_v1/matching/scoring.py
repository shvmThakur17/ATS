"""Transparent scoring components. Nothing here is a hiring probability.

Components (each 0..1, or None):
  required_skill_coverage   required skills with explicit evidence (structured or stated) / required skills
  preferred_skill_coverage  same for preferred skills
  semantic_relevance        mean of the `relevance_top_n` best chunk similarities across all queries
  responsibility_relevance  same, restricted to the RESPONSIBILITIES query
  experience_evidence       MET 1, NOT_MET 0, UNKNOWN None (incl. stated vs calculated durations giving opposite verdicts)
  education_evidence        EXPLICIT_MATCH 1, NOT_FOUND 0, UNKNOWN None

evidence_index = sum(w_i * c_i) / sum(w_i). A None component is either
  not applicable (the JD has no such requirement)   -> excluded from numerator and denominator, or
  unavailable (the JD asks, the resume cannot say)  -> per MatchingConfig.missing_component_policy:
      "zero" (default): c_i = 0 and w_i stays in the denominator, so omission can never raise the score;
      "renormalize": w_i is dropped (mean of what is known); evidence_completeness shows how much was known.
A real 0.0 is always counted. Weights come from MatchingConfig. Nothing here is a hiring probability.
"""

import re

from ..parsing.canonical.models import CanonicalResume
from .models import (EducationEvidence, EducationMatch, EducationStatus, EvidenceStrength, ExperienceMatch, ExperienceStatus, JobDescription, MatchEvidence, MatchingConfig,
                     QueryType, RetrievalStats)

# Education levels, checked on both the JD text and canonical degrees. Order = specificity.
LEVEL_PATTERNS = (
    ("doctorate", re.compile(r"\b(?:ph\.?d|doctorate|doctoral)\b", re.IGNORECASE)),
    ("master", re.compile(r"\b(?:master'?s?|m\.?\s?tech|m\.?\s?sc|mca|mba)\b", re.IGNORECASE)),
    ("bachelor", re.compile(r"\b(?:bachelor'?s?|b\.?\s?tech|b\.?\s?sc|bca|b\.?\s?com|b\.?\s?e\.?)\b", re.IGNORECASE)),
    ("diploma", re.compile(r"\bdiploma\b", re.IGNORECASE)),
)
GENERIC_DEGREE = re.compile(r"\bdegree\b", re.IGNORECASE)
DEGREE_LEVELS = {"bachelor", "master", "doctorate"}


def strength(score: float, config: MatchingConfig) -> EvidenceStrength:
    if score >= config.strong_evidence_threshold:
        return EvidenceStrength.STRONG
    return EvidenceStrength.MODERATE if score >= config.semantic_match_threshold else EvidenceStrength.WEAK


def levels_in(text: str) -> list[str]:
    return [name for name, pattern in LEVEL_PATTERNS if pattern.search(text)]


def month_index(value: str | None) -> int | None:
    """'YYYY-MM' -> months since year 0; anything less precise (e.g. 'YYYY') is not used."""
    if value and len(value) == 7 and value[4] == "-" and value[:4].isdigit() and value[5:].isdigit():
        return int(value[:4]) * 12 + int(value[5:]) - 1
    return None


STATED_YEARS = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*years?\b", re.IGNORECASE)
STATED_YEARS_TOLERANCE_MONTHS = 6


def refs(entities) -> dict:
    return dict(page_numbers=tuple(dict.fromkeys(str(e.page_number) for e in entities)),
                entity_ids=tuple(dict.fromkeys(e.entity_id for e in entities)))


def stated_experience(resume: CanonicalResume | None) -> dict:
    """The resume's own "N+ years" claim from the summary, verbatim, with its provenance. Empty if there is none."""
    if resume is None or not (resume.summary and resume.summary.value):
        return {}
    match = STATED_YEARS.search(resume.summary.value)
    if not match:
        return {}
    provenance = resume.summary.provenance
    return dict(stated_text=match.group(0), stated_months=round(float(match.group(1)) * 12),
                stated_source_path="summary", stated_source_text=resume.summary.value,
                logical_block_ids=tuple(dict.fromkeys(e.logical_block_id for e in provenance.entities)),
                **refs(provenance.entities))


def experience_match(jd: JobDescription, resume: CanonicalResume | None, config: MatchingConfig) -> ExperienceMatch | None:
    requirements = [r for r in jd.experience_requirements if r.min_years]
    if not requirements:
        return None
    requirement = max(requirements, key=lambda r: (r.min_years, r.text))
    base = dict(requirement_text=requirement.text, required_years=requirement.min_years)
    notes = ["overall employment duration (overlapping roles merged), not technology-specific"]
    if requirement.mentions_skills:
        notes.append(f"the JD ties the years to {', '.join(requirement.mentions_skills)}; "
                     "technology-specific duration is not inferred")
    if resume is None:
        return ExperienceMatch(**base, candidate_months=None, meets=None,
                               note="; ".join(notes + ["canonical resume not provided"]))
    intervals, excluded = [], []
    as_of = month_index(config.as_of)
    for i, work in enumerate(resume.work_experience):
        start = month_index(work.start_date)
        end = month_index(work.end_date) if work.end_date else (as_of if work.current else None)
        if start is None or end is None or end < start:
            excluded.append(f"work_experience[{i}]")
        else:
            intervals.append((start, end))
    if excluded:
        notes.append(f"excluded from the duration (no usable month-level dates or no as_of for a current role): "
                     f"{', '.join(excluded)}")
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    months = sum(end - start for start, end in merged)
    needed = requirement.min_years * 12
    if not resume.work_experience:
        meets, months, notes = None, None, notes + ["no work experience records"]
    elif months >= needed:
        meets = True
    else:
        meets = None if excluded else False
    stated = stated_experience(resume)
    extra = {}
    if stated:
        extra = dict(stated, stated_meets=stated["stated_months"] >= needed)
        if months is not None and abs(stated["stated_months"] - months) > STATED_YEARS_TOLERANCE_MONTHS:
            extra.update(discrepancy=True, discrepancy_note=(
                f"resume states \"{stated['stated_text']}\" ({stated['stated_months']} months) but employment dates "
                f"give {months} months; both are reported and neither is treated as authoritative"))
    status = (ExperienceStatus.UNKNOWN if meets is None else ExperienceStatus.MET if meets else ExperienceStatus.NOT_MET)
    if extra.get("discrepancy") and extra["stated_meets"] != meets and meets is not None:
        status = ExperienceStatus.UNKNOWN   # the two figures give opposite verdicts: the requirement is not confirmed
        notes.append("the stated and calculated durations disagree on whether the requirement is met")
    return ExperienceMatch(**base, candidate_months=months, meets=meets, status=status, note="; ".join(notes), **extra)


def education_match(jd: JobDescription, resume: CanonicalResume | None) -> EducationMatch | None:
    if not jd.education_requirements:
        return None
    required = []
    for text in jd.education_requirements:
        required += [level for level in levels_in(text) if level not in required]
    if not required and any(GENERIC_DEGREE.search(t) for t in jd.education_requirements):
        required = ["degree"]
    notes = []
    if any(re.search(r"\bequivalent\b", t, re.IGNORECASE) for t in jd.education_requirements):
        notes.append("the JD allows an equivalent qualification; equivalence is not evaluated")
    base = dict(requirement_texts=tuple(jd.education_requirements), required_levels=tuple(required))
    if resume is None:
        return EducationMatch(**base, candidate_levels=(), status=EducationStatus.UNKNOWN,
                              note="; ".join(notes + ["canonical resume not provided"]))
    found, evidence = [], []
    for i, education in enumerate(resume.education):
        levels = levels_in(education.degree or "")
        found += [level for level in levels if level not in found]
        provenance = education.provenance.get("degree")
        entities = provenance.entities if provenance else ()
        evidence.append(EducationEvidence(
            canonical_path=f"education[{i}]", degree=education.degree, field_of_study=education.field_of_study,
            institution=education.institution, raw_text=education.raw_text, levels=tuple(levels),
            logical_block_ids=(tuple(dict.fromkeys(e.logical_block_id for e in entities)) if entities
                               else education.logical_block_ids), **refs(entities)))
    base.update(candidate_levels=tuple(found), evidence=tuple(evidence))
    if not required:
        return EducationMatch(**base, status=EducationStatus.UNKNOWN,
                              note="; ".join(notes + ["no recognizable degree level in the JD"]))
    ok = any((level == "degree" and DEGREE_LEVELS & set(found)) or level in found for level in required)
    status = (EducationStatus.EXPLICIT_MATCH if ok else
              EducationStatus.NOT_FOUND if resume.education else EducationStatus.UNKNOWN)
    return EducationMatch(**base, status=status, note="; ".join(notes))


def retrieval_stats(evidence: list[MatchEvidence], config: MatchingConfig) -> RetrievalStats:
    best_by_chunk: dict[str, float] = {}
    for item in evidence:
        best_by_chunk[item.chunk_id] = max(item.similarity_score, best_by_chunk.get(item.chunk_id, float("-inf")))
    ranked = sorted(best_by_chunk.values(), reverse=True)
    top = ranked[:config.top_k]
    counts: dict[str, int] = {}
    for item in evidence:
        counts[item.query_type.value] = counts.get(item.query_type.value, 0) + 1
    return RetrievalStats(
        retrieval_count=len(evidence), unique_chunk_count=len(ranked),
        top_score=ranked[0] if ranked else None,
        average_top_k_score=sum(top) / len(top) if top else None,
        relevant_chunk_count=sum(score >= config.semantic_match_threshold for score in ranked),
        per_query_counts=counts)


def relevance(evidence: list[MatchEvidence], config: MatchingConfig, query_type: QueryType | None = None) -> float | None:
    best_by_chunk: dict[str, float] = {}
    for item in evidence:
        if query_type is None or item.query_type == query_type:
            best_by_chunk[item.chunk_id] = max(item.similarity_score, best_by_chunk.get(item.chunk_id, float("-inf")))
    top = sorted(best_by_chunk.values(), reverse=True)[:config.relevance_top_n]
    return min(1.0, max(0.0, sum(top) / len(top))) if top else None


def weighted(components: dict, weights: dict, names: tuple[str, ...], policy: str = "renormalize",
             not_applicable: frozenset | set = frozenset()) -> float | None:
    """sum(w_i * c_i) / sum(w_i) over the named components.

    A component that is not applicable (the JD has no such requirement) is never in the sum or the denominator.
    A component that is unavailable (None although the JD asks for it) follows `policy`:
      "zero"         c_i = 0, w_i stays in the denominator: absence of evidence cannot raise the score
      "renormalize"  w_i is dropped from both sums (the mean of what is known)
    A genuine 0.0 is always counted. None if no applicable component is available (or nothing is applicable)."""
    applicable = [n for n in names if n not in not_applicable]
    available = [n for n in applicable if components.get(n) is not None]
    if not available:
        return None
    counted = applicable if policy == "zero" else available
    total = sum(weights[n] for n in counted)
    return sum(weights[n] * (components.get(n) or 0.0) for n in counted) / total if total > 0 else None


def not_applicable_components(required_coverage, preferred_coverage, experience, education) -> frozenset:
    """Components that are None because the JD never asked (empty skill level, no experience/education requirement)."""
    return frozenset(name for name, absent in (
        ("required_skill_coverage", required_coverage is None), ("preferred_skill_coverage", preferred_coverage is None),
        ("experience_evidence", experience is None), ("education_evidence", education is None)) if absent)


def experience_component(experience: ExperienceMatch | None) -> float | None:
    """MET -> 1, NOT_MET -> 0, UNKNOWN (including conflicting stated/calculated verdicts) -> None."""
    if experience is None or experience.status == ExperienceStatus.UNKNOWN:
        return None
    return float(experience.status == ExperienceStatus.MET)


def build_components(required_coverage, preferred_coverage, evidence, experience, education, config) -> dict:
    return {
        "required_skill_coverage": required_coverage,
        "preferred_skill_coverage": preferred_coverage,
        "semantic_relevance": relevance(evidence, config),
        "responsibility_relevance": relevance(evidence, config, QueryType.RESPONSIBILITIES),
        "experience_evidence": experience_component(experience),
        "education_evidence": (None if education is None or education.status == EducationStatus.UNKNOWN
                               else float(education.status == EducationStatus.EXPLICIT_MATCH)),
    }

"""Informational evidence helpers (experience, education, retrieval statistics).

None of these feeds the candidate ranking: ranking is skills-only (ranking.py). They are kept so a reader of
the result can see experience and education evidence, with both stated and calculated experience preserved.
"""

import re

from ..parsing.canonical.models import CanonicalResume
from .models import (EducationEvidence, EducationMatch, EducationStatus, EvidenceStrength, ExperienceMatch, ExperienceStatus, JobDescription, MatchEvidence, MatchingConfig,
                     RetrievalStats)

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

"""Skill evidence classification, strongest claim first:

EXPLICIT_SKILL           the normalized skill is in a structured canonical field (skills list, project technologies).
EXPLICIT_TEXT_EVIDENCE   the skill's terminology (normalized form or an M8 formatting alias) appears as a whole term in
                         resume text outside those fields: summary, roles, descriptions, projects, or a retrieved chunk.
SEMANTIC_MATCH           a retrieved chunk scoring above the threshold shares part of a multi-word skill's terminology
                         ("SQL" for "SQL Server", "Web APIs" for "REST APIs"). The full term is NOT established.
NOT_FOUND                none of the above.
UNKNOWN                  there is no resume content to judge by.

Absence from canonical.skills is never NOT_FOUND by itself. Similarity alone proves nothing, a weaker class is never
promoted to a stronger one, and no skill is inferred from another (SQL != SQL Server, C# != .NET, Docker != Kubernetes,
ASP.NET Core != Angular, AWS != Azure). Normalization is M8's (`SKILL_FORMATTING`), reused as-is.
"""

import re

from ..parsing.canonical.models import CanonicalResume, Provenance
from ..parsing.normalization.skill_normalizer import SKILL_FORMATTING
from .models import (EvidenceKind, JobDescription, MatchEvidence, MatchingConfig, RequirementLevel, SkillMatch,
                     SkillStatus)

EXPLICIT_STATUSES = (SkillStatus.EXPLICIT_SKILL, SkillStatus.EXPLICIT_TEXT_EVIDENCE)
# Words that do not identify a technology on their own; sharing only these is not a semantic signal.
GENERIC_TOKENS = frozenset({"core", "server", "framework", "services", "service", "development", "programming",
                            "language", "platform", "tools", "tool", "engine"})


def mentions(skill: str, text: str) -> bool:
    """Whole-term, case-insensitive literal mention (".NET" is not found inside "ASP.NET")."""
    return re.search(rf"(?<![\w.#+]){re.escape(skill)}(?![\w#+])", text, re.IGNORECASE) is not None


def standalone_mention(skill: str, text: str, longer_terms=()) -> bool:
    """`mentions`, but an occurrence that is the start of a longer known skill ("SQL" in "SQL Server", ".NET" in
    ".NET Core") does not count: no skill is inferred from another."""
    longer = [t for t in longer_terms if len(t) > len(skill) and t.casefold().startswith(skill.casefold() + " ")]
    for match in re.finditer(rf"(?<![\w.#+]){re.escape(skill)}(?![\w#+])", text, re.IGNORECASE):
        rest = text[match.start():].casefold()
        if not any(rest.startswith(t.casefold()) and not rest[len(t):len(t) + 1].isalnum() for t in longer):
            return True
    return False


def block_ids(provenance: Provenance) -> tuple[str, ...]:
    return tuple(dict.fromkeys(e.logical_block_id for e in provenance.entities))


def canonical_skill_index(resume: CanonicalResume) -> dict[str, tuple]:
    """casefolded normalized value -> (kind, path, value, logical blocks). First occurrence wins; the
    skills list takes precedence over project technologies."""
    index: dict[str, tuple] = {}
    for i, skill in enumerate(resume.skills):
        index.setdefault(skill.value.casefold(), (EvidenceKind.SKILLS_LIST, f"skills[{i}]", skill.value,
                                                  block_ids(skill.provenance)))
    for j, project in enumerate(resume.projects):
        for k, technology in enumerate(project.technologies):
            index.setdefault(technology.value.casefold(),
                             (EvidenceKind.PROJECT_TECHNOLOGY, f"projects[{j}].technologies[{k}]", technology.value,
                              block_ids(technology.provenance)))
    return index


def best(evidence: list[MatchEvidence]) -> MatchEvidence | None:
    return min(evidence, key=lambda e: (-e.similarity_score, e.chunk_id)) if evidence else None


def term_variants(skill: str) -> tuple[str, ...]:
    """The normalized skill, M8's formatting aliases for it, "REST" as "RESTful", and a singular trailing "APIs"."""
    variants = [skill, *(alias for alias, value in SKILL_FORMATTING.items() if value == skill)]
    variants += [f"RESTful{v[4:]}" for v in variants if v.startswith("REST ")]
    variants += [v[:-1] for v in variants if v.endswith(" APIs")]
    return tuple(dict.fromkeys(variants))


def found_term(skill: str, text: str, longer_terms=()) -> str | None:
    return next((v for v in term_variants(skill) if standalone_mention(v, text, longer_terms)), None)


def known_terms(jd: JobDescription, resume: CanonicalResume | None) -> tuple[str, ...]:
    """Skill terms that could extend a shorter one: M8's canonical spellings, the JD's skills, the resume's own."""
    terms = {*SKILL_FORMATTING.values(), *(s.normalized for s in (*jd.required_skills, *jd.preferred_skills))}
    if resume is not None:
        terms |= {skill.value for skill in resume.skills}
        terms |= {t.value for project in resume.projects for t in project.technologies}
    return tuple(sorted(terms))


def text_sources(resume: CanonicalResume):
    """(canonical_path, text, logical blocks, provenance entities) for each narrative field, in a fixed order."""
    def source(path, text, provenance, fallback_blocks):
        return path, text, (block_ids(provenance) if provenance else fallback_blocks), \
            (provenance.entities if provenance else ())

    if resume.summary and resume.summary.value:
        yield source("summary", resume.summary.value, resume.summary.provenance, ())
    for i, work in enumerate(resume.work_experience):
        for name in ("role", "description"):
            if getattr(work, name):
                yield source(f"work_experience[{i}].{name}", getattr(work, name), work.provenance.get(name),
                             work.logical_block_ids)
    for j, project in enumerate(resume.projects):
        for name in ("name", "description"):
            if getattr(project, name):
                yield source(f"projects[{j}].{name}", getattr(project, name), project.provenance.get(name),
                             project.logical_block_ids)
    for k, certification in enumerate(resume.certifications):
        if certification.name:
            yield source(f"certifications[{k}].name", certification.name,
                         certification.provenance.get("certification"), certification.logical_block_ids)


def text_evidence(skill: str, resume: CanonicalResume | None, evidence: list[MatchEvidence],
                  longer_terms=()) -> dict | None:
    """First explicit mention: canonical narrative fields, then retrieved chunks (best score first)."""
    if resume is not None:
        for path, text, blocks, entities in text_sources(resume):
            term = found_term(skill, text, longer_terms)
            if term:
                chunk = best([e for e in evidence if e.matched_text == text])   # same text retrieved as a chunk
                return dict(canonical_path=path, supporting_text=text, matched_term=term, logical_block_ids=blocks,
                            page_numbers=tuple(dict.fromkeys(str(e.page_number) for e in entities)),
                            entity_ids=tuple(dict.fromkeys(e.entity_id for e in entities)),
                            supporting_chunk_id=chunk.chunk_id if chunk else None,
                            similarity_score=chunk.similarity_score if chunk else None)
    for chunk in sorted(evidence, key=lambda e: (-e.similarity_score, e.chunk_id)):
        term = found_term(skill, chunk.matched_text, longer_terms)
        if term:
            return dict(canonical_path=chunk.canonical_path, supporting_text=chunk.matched_text, matched_term=term,
                        logical_block_ids=chunk.logical_block_ids, supporting_chunk_id=chunk.chunk_id,
                        similarity_score=chunk.similarity_score, page_numbers=chunk.page_numbers,
                        entity_ids=chunk.entity_ids)
    return None


def partial_terms(skill: str, text: str) -> list[str]:
    """Distinctive tokens of a multi-word skill that the text mentions (the full term is checked elsewhere)."""
    tokens = skill.split()
    if len(tokens) < 2:
        return []
    return [t for t in tokens if t.casefold() not in GENERIC_TOKENS and mentions(t, text)]


def match_skills(jd: JobDescription, resume: CanonicalResume | None, evidence: list[MatchEvidence],
                 config: MatchingConfig) -> list[SkillMatch]:
    index = canonical_skill_index(resume) if resume is not None else {}
    has_content = resume is not None and bool(resume.skills or resume.projects or resume.work_experience
                                              or resume.summary)
    longer_terms = known_terms(jd, resume)
    results = []
    for skill in (*jd.required_skills, *jd.preferred_skills):
        required = skill.requirement == RequirementLevel.REQUIRED
        threshold = config.required_skill_match_threshold if required else config.preferred_skill_match_threshold
        common = dict(skill=skill.normalized, raw_jd_skill=skill.raw, requirement=skill.requirement)
        explicit = index.get(skill.normalized.casefold())
        if explicit:
            kind, path, value, blocks = explicit
            owner = "skills" if kind == EvidenceKind.SKILLS_LIST else path.split(".")[0]
            supporting = best([e for e in evidence if e.canonical_path == owner])
            results.append(SkillMatch(
                **common, status=SkillStatus.EXPLICIT_SKILL, evidence_kind=kind, canonical_path=path,
                resume_value=value, logical_block_ids=blocks,
                supporting_chunk_id=supporting.chunk_id if supporting else None,
                similarity_score=supporting.similarity_score if supporting else None,
                term_in_text=bool(supporting and mentions(skill.normalized, supporting.matched_text))))
            continue
        found = text_evidence(skill.normalized, resume, evidence, longer_terms)
        if found:
            results.append(SkillMatch(
                **common, status=SkillStatus.EXPLICIT_TEXT_EVIDENCE, evidence_kind=EvidenceKind.TEXT_MENTION,
                term_in_text=True, note="stated in resume text; not listed in the structured skills fields", **found))
            continue
        semantic = best([e for e in evidence if e.similarity_score >= threshold
                         and partial_terms(skill.normalized, e.matched_text)])
        if semantic:
            terms = partial_terms(skill.normalized, semantic.matched_text)
            results.append(SkillMatch(
                **common, status=SkillStatus.SEMANTIC_MATCH, evidence_kind=EvidenceKind.TEXT_MENTION,
                canonical_path=semantic.canonical_path, supporting_chunk_id=semantic.chunk_id,
                similarity_score=semantic.similarity_score, logical_block_ids=semantic.logical_block_ids,
                supporting_text=semantic.matched_text, matched_term=", ".join(terms),
                page_numbers=semantic.page_numbers, entity_ids=semantic.entity_ids,
                note=f"related wording ({', '.join(terms)}) only; {skill.normalized} itself is not stated"))
        elif not has_content:
            results.append(SkillMatch(**common, status=SkillStatus.UNKNOWN,
                                      note="no resume content available and no retrieved chunk mentions it"))
        else:
            results.append(SkillMatch(**common, status=SkillStatus.NOT_FOUND))
    return results


def coverage(matches: list[SkillMatch], level: RequirementLevel) -> float | None:
    """Skills with explicit evidence (structured or stated in text) / skills at that level. Not "fit"."""
    pool = [m for m in matches if m.requirement == level]
    if not pool:
        return None
    return sum(m.status in EXPLICIT_STATUSES for m in pool) / len(pool)

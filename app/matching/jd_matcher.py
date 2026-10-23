"""One JD against many resumes: structured, explainable evidence (no LLM, no hiring decision)."""

import logging

from ..parsing.canonical.models import CanonicalResume
from ..parsing.vector_store import PineconeStore
from .models import (CandidateMatch, EducationStatus, ExperienceStatus, JobDescription, MatchEvidence, MatchResult, MatchingConfig, RequirementLevel,
                     SkillStatus)
from .retrieval import ResumeRetriever, build_query_units
from .ranking import assign_ranks, skill_score
from .scoring import education_match, experience_match, retrieval_stats
from .skill_matching import coverage, match_skills

logger = logging.getLogger("app.matching")


class JDMatcher:
    """Holds one embedder and one store; both are reused for every query and candidate."""

    def __init__(self, store: PineconeStore, embedder, config: MatchingConfig | None = None):
        self.config = config or MatchingConfig()
        self.retriever = ResumeRetriever(store, embedder, self.config)

    def match(self, jd: JobDescription, candidates: dict[str, CanonicalResume] | None = None,
              namespaces: list[str] | None = None) -> MatchResult:
        """`candidates` maps document_id -> canonical resume (source of explicit skill evidence).
        `namespaces` limits the search; by default every resume namespace is searched."""
        candidates = candidates or {}
        units = build_query_units(jd, self.config)
        searched, retrieved = self.retriever.retrieve(units, namespaces)
        matches = [self.candidate_match(jd, document_id, retrieved.get(document_id, []), candidates.get(document_id))
                   for document_id in sorted(set(retrieved) | set(candidates))]
        matches.sort(key=lambda m: (tuple(-v for v in m.skill_score.key), m.document_id))   # skills only
        matches = [m.model_copy(update={"rank": rank})
                   for m, rank in zip(matches, assign_ranks([m.skill_score for m in matches]))]
        logger.info("matched jd=%s queries=%d namespaces=%d candidates=%d", jd.job_id, len(units), len(searched),
                    len(matches))
        return MatchResult(jd_id=jd.job_id, title=jd.title, queries=tuple(units), namespaces=tuple(searched),
                           candidates=tuple(matches), warnings=jd.warnings)

    def candidate_match(self, jd: JobDescription, document_id: str, evidence: list[MatchEvidence],
                        resume: CanonicalResume | None) -> CandidateMatch:
        config = self.config
        skills = match_skills(jd, resume, evidence, config)
        required = tuple(m for m in skills if m.requirement == RequirementLevel.REQUIRED)
        preferred = tuple(m for m in skills if m.requirement == RequirementLevel.PREFERRED)
        required_coverage = coverage(skills, RequirementLevel.REQUIRED)
        preferred_coverage = coverage(skills, RequirementLevel.PREFERRED)
        experience = experience_match(jd, resume, config)
        education = education_match(jd, resume)
        stats = retrieval_stats(evidence, config)

        warnings = []
        if resume is None:
            warnings.append("canonical resume not provided: explicit skill, experience and education evidence is UNKNOWN")
        if not evidence:
            warnings.append("no chunks were retrieved for this candidate")
        for m in skills:
            if m.status == SkillStatus.NOT_FOUND:
                warnings.append(f"{m.skill} ({m.requirement.value.lower()}) has no explicit resume evidence")
            elif m.status == SkillStatus.EXPLICIT_TEXT_EVIDENCE:
                warnings.append(f"{m.skill} ({m.requirement.value.lower()}) is only mentioned in text "
                                f"({m.canonical_path}), not listed as a skill")
            elif m.status == SkillStatus.SEMANTIC_MATCH:
                warnings.append(f"{m.skill} ({m.requirement.value.lower()}) has only semantic evidence "
                                f"({m.canonical_path}); the term itself is not stated")
        if experience and experience.status == ExperienceStatus.UNKNOWN:
            warnings.append(f"experience requirement is not confirmed (UNKNOWN): {experience.note}")
        if experience and experience.discrepancy:
            warnings.append(experience.discrepancy_note)
        if education and education.status != EducationStatus.EXPLICIT_MATCH:
            warnings.append(f"education requirement not explicitly met ({education.status.value}): {education.note}")
        return CandidateMatch(
            document_id=document_id, jd_id=jd.job_id, evidence=tuple(evidence), skill_matches=tuple(skills),
            required_skill_matches=required, preferred_skill_matches=preferred,
            required_skill_coverage=required_coverage, preferred_skill_coverage=preferred_coverage,
            retrieval=stats, experience=experience, education=education, skill_score=skill_score(skills), rank=0,
            retrieval_count=stats.retrieval_count, warnings=tuple(warnings))

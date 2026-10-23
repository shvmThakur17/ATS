"""M13 models, configuration and errors.

M13 produces structured *evidence*. Similarity scores are evidence of semantic relevance, never proof
that a candidate has a skill, and nothing here is a hiring decision.
"""

from dataclasses import dataclass, field
from enum import Enum
import os

from pydantic import BaseModel, ConfigDict


# ---- errors --------------------------------------------------------------------------------------
class MatchingError(Exception):
    """Base class for every M13 failure. No silent fallbacks."""


class InvalidJobDescriptionError(MatchingError):
    pass


class EmptyJobDescriptionError(InvalidJobDescriptionError):
    pass


class MissingJDSectionsError(InvalidJobDescriptionError):
    """The JD has no required skills and no responsibilities, so there is nothing to match on."""


class InvalidConfigError(MatchingError):
    pass


class QueryEmbeddingError(MatchingError):
    """The query embedding could not be produced."""


class EmbeddingDimensionError(MatchingError):
    """The query vector does not fit the index."""


class RetrievalError(MatchingError):
    """Pinecone failed while retrieving."""


class NamespaceDiscoveryError(RetrievalError):
    pass


class MalformedMatchResultError(MatchingError):
    """A Pinecone match is missing the id, score or metadata M13 needs."""


# ---- enums ---------------------------------------------------------------------------------------
class RequirementLevel(str, Enum):
    REQUIRED = "REQUIRED"
    PREFERRED = "PREFERRED"


class SkillStatus(str, Enum):
    EXPLICIT_SKILL = "EXPLICIT_SKILL"                   # normalized skill is in a structured canonical field
    EXPLICIT_TEXT_EVIDENCE = "EXPLICIT_TEXT_EVIDENCE"   # the skill's terminology appears in resume text only
    SEMANTIC_MATCH = "SEMANTIC_MATCH"   # a relevant chunk shares part of the terminology; term not established
    NOT_FOUND = "NOT_FOUND"             # no supporting evidence
    UNKNOWN = "UNKNOWN"                 # not enough information to decide


class ExperienceStatus(str, Enum):
    MET = "MET"             # the dates show the JD years and the resume's own claim does not contradict it
    NOT_MET = "NOT_MET"     # known to fall short
    UNKNOWN = "UNKNOWN"     # not determinable, or the stated and calculated durations disagree on the verdict


class EducationStatus(str, Enum):
    EXPLICIT_MATCH = "EXPLICIT_MATCH"   # a canonical degree has an accepted level
    NOT_FOUND = "NOT_FOUND"             # education is listed but no accepted level
    UNKNOWN = "UNKNOWN"                 # not determinable (no resume, no education records, no JD level)


class QueryType(str, Enum):
    REQUIRED_SKILLS = "REQUIRED_SKILLS"
    PREFERRED_SKILLS = "PREFERRED_SKILLS"
    RESPONSIBILITIES = "RESPONSIBILITIES"
    EXPERIENCE = "EXPERIENCE"
    EDUCATION = "EDUCATION"


class EvidenceStrength(str, Enum):
    STRONG = "STRONG"       # score >= strong_evidence_threshold
    MODERATE = "MODERATE"   # score >= semantic_match_threshold
    WEAK = "WEAK"           # below semantic_match_threshold


class EvidenceKind(str, Enum):
    SKILLS_LIST = "SKILLS_LIST"
    PROJECT_TECHNOLOGY = "PROJECT_TECHNOLOGY"
    TEXT_MENTION = "TEXT_MENTION"


# ---- configuration -------------------------------------------------------------------------------
DEFAULT_QUERY_CHUNK_TYPES = {
    QueryType.REQUIRED_SKILLS.value: (),
    QueryType.PREFERRED_SKILLS.value: (),
    QueryType.RESPONSIBILITIES.value: ("WORK_EXPERIENCE", "PROJECT", "SUMMARY"),
    QueryType.EXPERIENCE.value: ("WORK_EXPERIENCE", "SUMMARY"),
    QueryType.EDUCATION.value: ("EDUCATION", "CERTIFICATION"),
}


@dataclass(frozen=True)
class MatchingConfig:
    """All tunable values in one place. Thresholds are NOT calibrated probabilities. There are no ranking
    weights: candidate ranking is skills-only and lexicographic (see ranking.py)."""
    top_k: int = 10                              # chunks retrieved per JD query per resume
    semantic_match_threshold: float = 0.60       # min cosine for a chunk to count as relevant (MODERATE+)
    strong_evidence_threshold: float = 0.75      # min cosine to label evidence STRONG
    required_skill_match_threshold: float = 0.60   # min cosine for a chunk to support a REQUIRED skill SEMANTIC_MATCH
    preferred_skill_match_threshold: float = 0.60  # same, for PREFERRED skills
    query_chunk_types: dict = field(default_factory=lambda: dict(DEFAULT_QUERY_CHUNK_TYPES))
    as_of: str | None = None                     # "YYYY-MM" end for current roles; None = excluded (no clock is read)

    def __post_init__(self):
        if self.top_k < 1:
            raise InvalidConfigError("top_k must be >= 1")
        for name in ("semantic_match_threshold", "strong_evidence_threshold", "required_skill_match_threshold",
                     "preferred_skill_match_threshold"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise InvalidConfigError(f"{name} must be within [0, 1]")
        if self.strong_evidence_threshold < self.semantic_match_threshold:
            raise InvalidConfigError("strong_evidence_threshold must be >= semantic_match_threshold")
        if self.as_of is not None and not (len(self.as_of) == 7 and self.as_of[4] == "-"
                                          and self.as_of[:4].isdigit() and self.as_of[5:].isdigit()):
            raise InvalidConfigError("as_of must look like YYYY-MM")

    @classmethod
    def from_env(cls) -> "MatchingConfig":
        default = cls()
        return cls(top_k=int(os.environ.get("M13_TOP_K", default.top_k)),
                   semantic_match_threshold=float(os.environ.get("M13_SEMANTIC_MATCH_THRESHOLD",
                                                                 default.semantic_match_threshold)),
                   as_of=os.environ.get("M13_AS_OF") or None)


# ---- models --------------------------------------------------------------------------------------
class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class JDSkill(FrozenModel):
    raw: str                    # as written in the JD
    normalized: str             # via the M8 skill normalizer (same terminology as resumes)
    requirement: RequirementLevel


class ExperienceRequirement(FrozenModel):
    text: str
    min_years: int | None = None
    mentions_skills: tuple[str, ...] = ()   # JD skills named in the item (years may be technology-specific)


class JobDescription(FrozenModel):
    job_id: str
    title: str | None
    raw_text: str                           # always kept verbatim
    required_skills: tuple[JDSkill, ...] = ()
    preferred_skills: tuple[JDSkill, ...] = ()
    responsibilities: tuple[str, ...] = ()
    qualifications: tuple[str, ...] = ()
    experience_requirements: tuple[ExperienceRequirement, ...] = ()
    education_requirements: tuple[str, ...] = ()
    unclassified_text: tuple[str, ...] = ()  # nothing that is not recognized is discarded
    warnings: tuple[str, ...] = ()


class QueryUnit(FrozenModel):
    query_type: QueryType
    text: str                               # exactly what is embedded
    chunk_types: tuple[str, ...] = ()       # Pinecone chunk_type filter; empty = no filter


class MatchEvidence(FrozenModel):
    """JD query -> Pinecone vector -> chunk -> canonical path -> logical blocks, all preserved."""
    document_id: str
    namespace: str
    chunk_id: str
    chunk_type: str
    canonical_path: str
    similarity_score: float
    query_type: QueryType
    matched_text: str                       # the stored M10 chunk text, unchanged
    evidence_strength: EvidenceStrength
    logical_block_ids: tuple[str, ...] = ()
    page_numbers: tuple[str, ...] = ()
    entity_ids: tuple[str, ...] = ()
    metadata: dict = {}


class SkillMatch(FrozenModel):
    skill: str                              # normalized JD skill
    raw_jd_skill: str
    requirement: RequirementLevel
    status: SkillStatus
    evidence_kind: EvidenceKind | None = None
    canonical_path: str | None = None       # e.g. skills[3], projects[1].technologies[0], or the chunk's path
    resume_value: str | None = None         # the resume's own (normalized) value for EXPLICIT_SKILL
    supporting_text: str | None = None      # original resume text containing the term (EXPLICIT_TEXT_EVIDENCE)
    matched_term: str | None = None         # the JD-side spelling that was found in the text
    page_numbers: tuple[str, ...] = ()
    entity_ids: tuple[str, ...] = ()
    supporting_chunk_id: str | None = None
    similarity_score: float | None = None   # best retrieved score of the supporting chunk, if any
    logical_block_ids: tuple[str, ...] = ()
    term_in_text: bool = False              # literal mention inside the supporting chunk text
    note: str | None = None


class ExperienceMatch(FrozenModel):
    """Both figures are kept; neither is authoritative. `meets` follows the employment dates only."""
    requirement_text: str
    required_years: int
    candidate_months: int | None            # calculated from employment dates, overlapping roles merged
    meets: bool | None                      # per the calculated duration; None = cannot be determined
    note: str
    stated_text: str | None = None          # resume's own claim, verbatim ("2.6+ years")
    stated_months: int | None = None
    stated_meets: bool | None = None        # per the stated figure
    stated_source_path: str | None = None
    stated_source_text: str | None = None   # the full text the claim was read from
    logical_block_ids: tuple[str, ...] = ()
    page_numbers: tuple[str, ...] = ()
    entity_ids: tuple[str, ...] = ()
    status: ExperienceStatus = ExperienceStatus.UNKNOWN   # qualified verdict; UNKNOWN on conflicting verdicts
    discrepancy: bool = False               # stated and calculated durations differ beyond the tolerance
    discrepancy_note: str | None = None


class EducationEvidence(FrozenModel):
    """One canonical education record, as written, with its provenance."""
    canonical_path: str
    degree: str | None
    field_of_study: str | None
    institution: str | None
    raw_text: str
    levels: tuple[str, ...] = ()
    logical_block_ids: tuple[str, ...] = ()
    page_numbers: tuple[str, ...] = ()
    entity_ids: tuple[str, ...] = ()


class EducationMatch(FrozenModel):
    requirement_texts: tuple[str, ...]
    required_levels: tuple[str, ...]
    candidate_levels: tuple[str, ...]
    status: EducationStatus
    note: str
    evidence: tuple[EducationEvidence, ...] = ()


class RetrievalStats(FrozenModel):
    retrieval_count: int                    # evidence items across all queries
    unique_chunk_count: int
    top_score: float | None
    average_top_k_score: float | None       # mean best score of the top_k unique chunks
    relevant_chunk_count: int               # unique chunks scoring >= semantic_match_threshold
    per_query_counts: dict[str, int] = {}


class SkillScore(FrozenModel):
    """Skills-only ranking inputs. Experience, education, responsibilities and retrieval similarity are not in it.
    A level with no skills in the JD has coverage None and contributes 0.0 to the key (the same for every candidate)."""
    required_total: int
    required_explicit: int                  # EXPLICIT_SKILL + EXPLICIT_TEXT_EVIDENCE
    required_semantic: int                  # SEMANTIC_MATCH only: never counted as covered
    preferred_total: int
    preferred_explicit: int
    preferred_semantic: int
    required_coverage: float | None
    preferred_coverage: float | None
    required_semantic_share: float | None
    preferred_semantic_share: float | None
    key: tuple[float, float, float, float]  # compared lexicographically, larger is better


class CandidateMatch(FrozenModel):
    """Evidence about one resume (document_id). Not a hiring recommendation."""
    document_id: str
    jd_id: str
    evidence: tuple[MatchEvidence, ...]
    skill_matches: tuple[SkillMatch, ...]
    required_skill_matches: tuple[SkillMatch, ...]
    preferred_skill_matches: tuple[SkillMatch, ...]
    required_skill_coverage: float | None   # explicit required skills / required skills
    preferred_skill_coverage: float | None
    retrieval: RetrievalStats
    experience: ExperienceMatch | None      # informational; never used in ranking
    education: EducationMatch | None        # informational; never used in ranking
    skill_score: SkillScore                 # the ONLY input to ranking (see ranking.py)
    rank: int                               # 1 = best; candidates with an identical skill_score share a rank
    retrieval_count: int
    warnings: tuple[str, ...] = ()

    def grouped_evidence(self) -> dict[str, tuple[MatchEvidence, ...]]:
        """Evidence by chunk type, in a fixed type order; types without evidence are absent."""
        order = ("SUMMARY", "WORK_EXPERIENCE", "PROJECT", "EDUCATION", "SKILLS", "CERTIFICATION", "AWARD",
                 "LANGUAGE", "INTEREST", "CONTACT")
        return {t: tuple(e for e in self.evidence if e.chunk_type == t) for t in order
                if any(e.chunk_type == t for e in self.evidence)}


class MatchResult(FrozenModel):
    jd_id: str
    title: str | None
    queries: tuple[QueryUnit, ...]
    namespaces: tuple[str, ...]
    candidates: tuple[CandidateMatch, ...]  # ordered by skill_score (desc), then document_id; skills only
    warnings: tuple[str, ...] = ()

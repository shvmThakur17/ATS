"""POST /api/v1/jd/rank: rank the whole shared resume pool against a job description, discovered server-side.

MVP SCOPE: single company, one shared pool. The caller sends only the job description; the server lists the resume
namespaces under its own prefix, so HR never handles document ids or access tokens. Authentication is the same API
key as every other endpoint. There is NO per-user, per-role or per-tenant scoping: any holder of an API key can rank
every resume in the pool, so this endpoint must not be exposed to more than one company. See docs/mvp-single-company.md.

Ranking itself is the unchanged N1/M13 skills-only service. Resumes that are not provably fully indexed are skipped
and reported, never ranked from partial data. The pool is capped; a larger pool is flagged `truncated`.
"""

import logging

from fastapi import APIRouter, Depends

from app.api.dependencies import get_caller, get_matching_service, get_readiness, get_runtime
from app.api.errors import ApiError
from app.api.readiness import READY, ReadinessUnavailableError, provider_call
from app.api.runner import JobTimeoutError, ServerBusyError, resolve
from app.api.v1.jd_matching import candidate_result, store_unavailable
from app.matching import InvalidJobDescriptionError, MatchingError, parse_job_description
from app.matching.models import EmbeddingDimensionError, QueryEmbeddingError, RetrievalError
from app.matching.pinecone_canonical_adapter import (METADATA_CHUNK_TYPES, CanonicalReconstructionError,
                                                      MetadataRetrievalError)
from app.matching.pinecone_match_service import (CandidateRef, NamespaceCountUnavailableError,
                                                  RetrievalDepthExceededError)
from app.schemas.pipeline_api import HEX_SHA256, JdSummary, MAX_CANDIDATES, RetrievalSummary
from app.schemas.pool_api import PoolCandidateResult, PoolRankRequest, PoolRankResponse, PoolSummary, SkippedResume
from app.services.m14_scoring import calculate_match_score
from app.schemas.m14_reasoning import CandidateReasoning
from app.services.ai.m14_reasoning_service import M14ReasoningService
import hashlib

from app.matching.jd_parser import parse_job_description
from app.matching.models import (
    ExperienceRequirement,
    JDSkill,
    JobDescription,
    RequirementLevel,
)
from app.parsing.normalization.skill_normalizer import normalize_skill
logger = logging.getLogger("app.api")
router = APIRouter()
RETRY_SECONDS = 5
MAX_POOL_CONSIDERED = MAX_CANDIDATES
MAX_REASONING_CANDIDATES = 5
REASONING_TIMEOUT_SECONDS = 60
def build_structured_jd(structured, raw_text: str) -> JobDescription:
    """Build the existing M13 JD model from HR-approved structured fields."""

    def make_skills(names: list[str], level: RequirementLevel) -> tuple[JDSkill, ...]:
        result = []
        seen = set()

        for name in names:
            normalized = normalize_skill(name).normalized_value
            if not normalized or normalized.casefold() in seen:
                continue

            seen.add(normalized.casefold())
            result.append(
                JDSkill(
                    raw=name,
                    normalized=normalized,
                    requirement=level,
                )
            )

        return tuple(result)

    required = make_skills(
        structured.required_skills,
        RequirementLevel.REQUIRED,
    )
    required_names = {skill.normalized.casefold() for skill in required}

    preferred = tuple(
        skill
        for skill in make_skills(
            structured.preferred_skills,
            RequirementLevel.PREFERRED,
        )
        if skill.normalized.casefold() not in required_names
    )

    if not required and not preferred:
        raise InvalidJobDescriptionError(
            "At least one recognizable required or preferred skill is needed."
        )

    experience = structured.experience
    experience_text = "; ".join(
        f"{label}: {value:g} years"
        for label, value in (
            ("Minimum experience", experience.minimum_years),
            ("Maximum experience", experience.maximum_years),
        )
        if value is not None
    )
    sections = [
        structured.role,
        "Required skills: " + ", ".join(structured.required_skills),
        "Preferred skills: " + ", ".join(structured.preferred_skills),
        experience_text,
        ("Education (required): " if structured.education.required else "Education: ")
        + (structured.education.description or ""),
        "Responsibilities: " + "; ".join(structured.responsibilities),
        "Additional preferences: " + "; ".join(structured.additional_preferences),
        raw_text,
    ]
    description = "\n".join(section for section in sections if section)

    acc = JobDescription(
        job_id="jd-" + hashlib.sha256(description.encode("utf-8")).hexdigest()[:12],
        title=structured.role,
        raw_text=description,
        required_skills=required,
        preferred_skills=preferred,
        responsibilities=tuple(structured.responsibilities),
        qualifications=tuple(structured.additional_preferences),
        experience_requirements=(
            (ExperienceRequirement(
                text=experience_text,
                min_years=(
                    int(experience.minimum_years)
                    if experience.minimum_years is not None
                    and experience.minimum_years.is_integer()
                    else None
                ),
            ),) if experience_text else ()
        ),
        education_requirements=(
            (structured.education.description,)
            if structured.education.description
            else ()
        ),
        unclassified_text=(),
        warnings=(),
    )
    return acc
def discover(store) -> tuple[list[CandidateRef], dict[str, int]]:
    """Every well-formed resume namespace under the store's prefix, sorted. Counts come from one stats call."""
    prefix = store.config.namespace_prefix + ":"
    stats = provider_call(store.index.describe_index_stats)
    refs, counts = [], {}
    for name, entry in stats.namespaces.items():
        document_id = name[len(prefix):] if name.startswith(prefix) else ""
        if HEX_SHA256.match(document_id):                     # anything else under the prefix is not an N2 resume
            refs.append(CandidateRef(document_id, name))
            counts[name] = int(entry.vector_count)
    refs.sort(key=lambda r: r.document_id)
    return refs, counts


def source_file(store, namespace: str) -> str | None:
    """File name from one metadata vector. Cosmetic: any failure just leaves the name out."""
    try:
        probe = [1.0] + [0.0] * ((store.dimension or store.describe_index().dimension) - 1)
        matches = store.query(probe, 1, namespace, filter={"chunk_type": {"$in": list(METADATA_CHUNK_TYPES)}})
        name = (matches[0].metadata or {}).get("source_file") if matches else None
        return name if isinstance(name, str) and name else None
    except Exception:
        return None
    
def explain_top_candidates(job_description: str, candidates: list[PoolCandidateResult]) -> None:
    """Add qualitative reasoning to the top five without changing rank or score."""
    for candidate in candidates[:MAX_REASONING_CANDIDATES]:
        try:
            result = M14ReasoningService.analyze(job_description=job_description,
                required_skills=[skill.model_dump() for skill in candidate.required_skills],
                preferred_skills=[skill.model_dump() for skill in candidate.preferred_skills],
                missing_required=list(candidate.missing_skills.required),
                missing_preferred=list(candidate.missing_skills.preferred),
            )
            candidate.reasoning = result
        except Exception as exc:
            # Reasoning is optional; never fail the candidate search.
            logger.warning(
                "M14 candidate reasoning failed type=%s",
                type(exc).__name__,
            )
            candidate.reasoning = CandidateReasoning(
                status="UNAVAILABLE",
                recommendation="Insufficient Evidence",
            )


@router.post("/rank", response_model=PoolRankResponse)
async def rank_pool(request: PoolRankRequest, caller: str = Depends(get_caller), runtime=Depends(get_runtime),
                    matching=Depends(get_matching_service), gate=Depends(get_readiness)):
    light, heavy = runtime.light(), runtime.heavy()
    if request.structured_jd is not None:
        try:
            jd = build_structured_jd(request.structured_jd,request.job_description,)
        except InvalidJobDescriptionError:
            raise ApiError(422,"INVALID_JOB_DESCRIPTION","Provide at least one recognizable required or preferred skill.",) from None
    else:
        try:
            jd = await light.run(parse_job_description,request.job_description,)
        except InvalidJobDescriptionError:
            raise ApiError(422,"INVALID_JOB_DESCRIPTION","The job description needs recognizable required skills or responsibilities.",) from None
        except (JobTimeoutError, ServerBusyError):
            raise ApiError(422,"INVALID_JOB_DESCRIPTION","The job description could not be analysed.",) from None
        if not (jd.required_skills or jd.preferred_skills):
            raise ApiError(422, "NO_SKILLS_IN_JOB_DESCRIPTION",
                       "The job description lists no required or preferred skills to rank on.")

    def job(service):
        store = service.store
        found, counts = discover(store)
        considered = found[:MAX_POOL_CONSIDERED]
        ready, skipped = [], []
        for ref in considered:
            state = gate.assess(store, ref, counts[ref.namespace]).state
            if state == READY:
                ready.append(ref)
            else:
                skipped.append(SkippedResume(document_id=ref.document_id, reason=state))
        outcome = service.match(jd, ready, {r.namespace: counts[r.namespace] for r in ready}) if ready else None
        names = {r.document_id: source_file(store, r.namespace) for r in ready}
        return found, considered, skipped, outcome, names

    try:
        service = await resolve(matching, heavy)                 # built only after every cheap check passed
        found, considered, skipped, outcome, names = await heavy.run(job, service)
        ranked = []
        for candidate in (outcome.result.candidates if outcome else []):
            base = candidate_result(candidate)

            score = calculate_match_score(base.required_skills,base.preferred_skills)

            ranked.append(PoolCandidateResult(**base.model_dump(),source_file=names.get(candidate.document_id),required_skill_points=score.required_points,
                    preferred_skill_points=score.preferred_points,overall_match_score=score.overall_score))
            
        if ranked and request.include_reasoning:
            try:
                await heavy.run(explain_top_candidates,jd.raw_text,ranked[:MAX_REASONING_CANDIDATES],timeout=REASONING_TIMEOUT_SECONDS,)
            except (JobTimeoutError, ServerBusyError) as exc:
                logger.warning("M14 reasoning stage unavailable type=%s",type(exc).__name__)
                for candidate in ranked[:MAX_REASONING_CANDIDATES]:
                    candidate.reasoning = CandidateReasoning(status="UNAVAILABLE",recommendation="Insufficient Evidence")
        return PoolRankResponse(
            jd=JdSummary(job_id=jd.job_id, title=jd.title, required_skills=[s.normalized for s in jd.required_skills],
                         preferred_skills=[s.normalized for s in jd.preferred_skills]),
            retrieval=RetrievalSummary(effective_top_k=outcome.effective_top_k if outcome else 0,
                                       max_retrieval_depth=service.limits.max_retrieval_depth),
            pool=PoolSummary(discovered=len(found), considered=len(considered), ranked=len(ranked),
                             skipped=len(skipped), truncated=len(found) > len(considered),
                             max_considered=MAX_POOL_CONSIDERED),
            skipped=skipped, candidates=ranked)
    except ServerBusyError:
        raise ApiError(503, "SERVER_BUSY", "The server is busy. Try again shortly.",
                       headers={"Retry-After": RETRY_SECONDS}) from None
    except JobTimeoutError:
        raise ApiError(504, "PROCESSING_TIMEOUT", "Matching took too long. Try again.") from None
    except RetrievalDepthExceededError:
        raise ApiError(422, "CANDIDATE_TOO_LARGE", "A resume has more stored data than can be matched completely.") from None
    except ReadinessUnavailableError as error:
        logger.warning("pool readiness check failed reason=%s", error.reason)
        raise store_unavailable() from None
    except (NamespaceCountUnavailableError, MetadataRetrievalError, RetrievalError):
        raise store_unavailable() from None
    except (QueryEmbeddingError, EmbeddingDimensionError):
        raise ApiError(503, "EMBEDDING_UNAVAILABLE", "The embedding service is not available right now.") from None
    except ApiError:
        raise
    except (CanonicalReconstructionError, MatchingError) as error:
        logger.error("pool ranking failed type=%s", type(error).__name__)
        raise ApiError(500, "MATCHING_FAILED", "The resumes could not be ranked.") from None
    except Exception as error:
        logger.error("pool ranking failed type=%s", type(error).__name__)
        unbuilt = not getattr(matching, "built", True)
        raise (ApiError(503, "SERVICE_UNAVAILABLE", "The service is not available right now.") if unbuilt
               else ApiError(500, "MATCHING_FAILED", "The resumes could not be ranked.")) from None

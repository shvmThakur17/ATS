"""POST /api/v1/jd/match: skills-only ranking through the N1 service. POST /api/v1/jd/search: retired.

The request names candidates by (document_id, access_token). Namespaces are never accepted from the caller: they are
derived here from verified document ids, and the N1 service receives that explicit list, so no namespace is ever
discovered or searched implicitly. Cheap checks (duplicates, tokens, job description) run before the heavy service
is built; candidates whose data is not provably complete in the index are refused, never ranked.
"""

import logging

from fastapi import APIRouter, Depends

from app.api.candidate_access import CandidateTokenSigner
from app.api.dependencies import get_caller, get_matching_service, get_readiness, get_runtime, get_token_signer
from app.api.errors import ApiError
from app.api.evidence import safe_excerpt
from app.api.readiness import TOO_LARGE, CandidatesNotReadyError, ReadinessUnavailableError
from app.api.runner import JobTimeoutError, ServerBusyError, resolve
from app.matching import InvalidJobDescriptionError, MatchingError, parse_job_description
from app.matching.models import EmbeddingDimensionError, QueryEmbeddingError, RetrievalError, SkillStatus
from app.matching.pinecone_canonical_adapter import CanonicalReconstructionError, MetadataRetrievalError
from app.matching.pinecone_match_service import (CandidateRef, NamespaceCountUnavailableError, PineconeMatchOutcome,
                                                  RetrievalDepthExceededError)
from app.schemas.pipeline_api import (CandidateResult, JdMatchRequest, JdMatchResponse, JdSummary, MissingSkills,
                                      RetrievalSummary, SkillEvidence, SkillResult)

logger = logging.getLogger("app.api")
router = APIRouter()
RETRY_SECONDS = 5
COVERED = (SkillStatus.EXPLICIT_SKILL, SkillStatus.EXPLICIT_TEXT_EVIDENCE)     # M13's own definition of "covered"


def skill_result(match) -> SkillResult:
    # Step 1: Check whether the skill is confirmed.
    covered = match.status in COVERED

    # Step 2: Evidence is optional.
    evidence = None

    # Include evidence for confirmed or semantic matches.
    if match.status in COVERED or match.status == SkillStatus.SEMANTIC_MATCH:
        evidence = SkillEvidence(
            kind=match.evidence_kind.value if match.evidence_kind else None,
            canonical_path=match.canonical_path,
            matched_term=match.matched_term,
            excerpt=safe_excerpt(
                match.supporting_text,
                match.matched_term,
            ),
            page_numbers=list(match.page_numbers),
            chunk_id=match.supporting_chunk_id,
        )

    # Step 3: Return the skill result.
    add =  SkillResult(
        skill=match.skill,
        requirement=match.requirement.value,
        status=match.status.value,
        covered=covered,
        evidence=evidence,
    )
    return add


def candidate_result(candidate) -> CandidateResult:
    # Step 1: Convert required and preferred skill matches.
    required_skills = []

    for match in candidate.required_skill_matches:
        required_skills.append(skill_result(match))

    preferred_skills = []

    for match in candidate.preferred_skill_matches:
        preferred_skills.append(skill_result(match))

    # Step 2: Find skills that are not confirmed.
    missing_required = [
        skill.skill
        for skill in required_skills
        if not skill.covered
    ]

    missing_preferred = [
        skill.skill
        for skill in preferred_skills
        if not skill.covered
    ]

    # Step 3: Add a warning if no resume vectors were retrieved.
    warnings = []

    if candidate.retrieval_count == 0:
        warnings.append("NO_RESUME_VECTORS")

    # Step 4: Return the complete candidate result.
    abc = CandidateResult(
        rank=candidate.rank,
        document_id=candidate.document_id,
        required_skill_coverage=candidate.required_skill_coverage,
        preferred_skill_coverage=candidate.preferred_skill_coverage,
        ranking_key=list(candidate.skill_score.key),
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        missing_skills=MissingSkills(
            required=missing_required,
            preferred=missing_preferred,
        ),
        warnings=warnings,
    )
    return abc


def to_response(jd, outcome: PineconeMatchOutcome) -> JdMatchResponse:
    return JdMatchResponse(
        jd=JdSummary(job_id=jd.job_id, title=jd.title, required_skills=[s.normalized for s in jd.required_skills],
                     preferred_skills=[s.normalized for s in jd.preferred_skills]),
        retrieval=RetrievalSummary(effective_top_k=outcome.effective_top_k,
                                   max_retrieval_depth=outcome.max_retrieval_depth),
        candidates=[candidate_result(c) for c in outcome.result.candidates])


def indexes(values) -> dict:
    return {"candidate_indexes": list(values)}


def store_unavailable() -> ApiError:
    """Retryable provider failure. One contract for every way the vector store can be unreachable while matching."""
    return ApiError(502, "RETRIEVAL_UNAVAILABLE", "The vector store is not available right now.",
                    {"retryable": True}, {"Retry-After": RETRY_SECONDS})


def not_ready(error: CandidatesNotReadyError) -> ApiError:
    if error.indexes(TOO_LARGE):
        return ApiError(422, "CANDIDATE_TOO_LARGE", "A candidate has more stored data than can be matched completely.",
                        indexes(error.indexes(TOO_LARGE)))
    return ApiError(409, "CANDIDATE_NOT_READY",
                    "One or more candidates are not fully indexed yet. Nothing was ranked; try again shortly.",
                    {**indexes(error.indexes()), "retryable": True}, {"Retry-After": RETRY_SECONDS})


@router.post("/match", response_model=JdMatchResponse)
async def match_candidates(request: JdMatchRequest, caller: str = Depends(get_caller),
                           signer: CandidateTokenSigner = Depends(get_token_signer), runtime=Depends(get_runtime),
                           matching=Depends(get_matching_service), gate=Depends(get_readiness)):
    seen, duplicates = {}, []
    for index, candidate in enumerate(request.candidates):
        if candidate.document_id in seen:
            duplicates.append(index)
        seen.setdefault(candidate.document_id, index)
    if duplicates:
        raise ApiError(422, "DUPLICATE_CANDIDATE", "The same candidate was listed more than once.", indexes(duplicates))
    unauthorized = [i for i, c in enumerate(request.candidates) if not signer.verify(c.document_id, c.access_token)]
    if unauthorized:
        raise ApiError(403, "UNAUTHORIZED_CANDIDATE", "One or more candidate references are not authorized.",
                       indexes(unauthorized))
    light, heavy = runtime.light(), runtime.heavy()
    try:
        jd = await light.run(parse_job_description, request.job_description)      # off the event loop
    except InvalidJobDescriptionError:
        raise ApiError(422, "INVALID_JOB_DESCRIPTION",
                       "The job description needs recognizable required skills or responsibilities.") from None
    except (JobTimeoutError, ServerBusyError):
        raise ApiError(422, "INVALID_JOB_DESCRIPTION", "The job description could not be analysed.") from None
    if not (jd.required_skills or jd.preferred_skills):
        raise ApiError(422, "NO_SKILLS_IN_JOB_DESCRIPTION",
                       "The job description lists no required or preferred skills to rank on.")
    try:
        service = await resolve(matching, heavy)                    # built only now, after every cheap check passed
        refs = [CandidateRef(c.document_id, service.store.namespace_for(c.document_id)) for c in request.candidates]

        def job():
            counts = gate.verify(service.store, refs)               # one stats call; raises if anything is not ready
            return service.match(jd, refs, counts) if counts else service.match(jd, refs)

        outcome = await heavy.run(job)
        return await light.run(to_response, jd, outcome)            # bounded, off the event loop
    except CandidatesNotReadyError as error:
        raise not_ready(error) from None
    except ServerBusyError:
        raise ApiError(503, "SERVER_BUSY", "The server is busy. Try again shortly.",
                       headers={"Retry-After": RETRY_SECONDS}) from None
    except JobTimeoutError:
        raise ApiError(504, "PROCESSING_TIMEOUT", "Matching took too long. Try again.") from None
    except RetrievalDepthExceededError:
        raise ApiError(422, "CANDIDATE_TOO_LARGE", "A candidate has more stored data than can be matched completely.") from None
    except ReadinessUnavailableError as error:                  # readiness could not be verified: nothing was ranked
        logger.warning("readiness check failed reason=%s", error.reason)
        raise store_unavailable() from None
    except (NamespaceCountUnavailableError, MetadataRetrievalError, RetrievalError):
        raise store_unavailable() from None
    except (QueryEmbeddingError, EmbeddingDimensionError):
        raise ApiError(503, "EMBEDDING_UNAVAILABLE", "The embedding service is not available right now.") from None
    except ApiError:
        raise
    except (CanonicalReconstructionError, MatchingError) as error:
        logger.error("matching failed type=%s", type(error).__name__)
        raise ApiError(500, "MATCHING_FAILED", "The candidates could not be matched.") from None
    except Exception as error:
        logger.error("matching failed type=%s", type(error).__name__)
        unbuilt = not getattr(matching, "built", True)
        raise (ApiError(503, "SERVICE_UNAVAILABLE", "The service is not available right now.") if unbuilt
               else ApiError(500, "MATCHING_FAILED", "The candidates could not be matched.")) from None


@router.post("/search", deprecated=True, status_code=410)
async def retired_search():
    """The earlier similarity-ranked, Ollama-assisted search is retired; ranking is skills-only via /match."""
    raise ApiError(410, "ENDPOINT_RETIRED",
                   "This endpoint has been retired. Upload resumes to /api/v1/resume/upload and rank them with "
                   "POST /api/v1/jd/match.")

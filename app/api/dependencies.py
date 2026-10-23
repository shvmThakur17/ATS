"""FastAPI dependencies. All of them are cheap: none builds a model, a Pinecone client or reads a file.

Expensive services are exposed as `LazyService` objects and are built by the route only after the request has passed
authentication, rate limiting and input validation. Tests replace any of these with `app.dependency_overrides`.
"""

from dataclasses import dataclass

from fastapi import Request

from ..ingestion.pdf_resume_ingestion_service import IngestionLimits
from ..schemas.pipeline_api import MAX_CANDIDATES, MAX_JD_CHARS
from .candidate_access import CandidateTokenSigner, TokenConfigurationError
from .errors import ApiError
from .runtime import ApiRuntime
from .settings import SettingsError

MULTIPART_OVERHEAD_BYTES = 64 * 1024
JSON_BYTES_PER_CHARACTER = 12            # worst case: one non-BMP character written as a \uXXXX\uXXXX escape pair
JSON_BYTES_PER_CANDIDATE = 256
JSON_SLACK_BYTES = 4096


@dataclass(frozen=True)
class ApiLimits:
    max_upload_bytes: int = IngestionLimits().max_pdf_bytes
    max_jd_chars: int = MAX_JD_CHARS
    max_candidates: int = MAX_CANDIDATES

    @property
    def max_upload_body_bytes(self) -> int:
        return self.max_upload_bytes + MULTIPART_OVERHEAD_BYTES

    @property
    def max_json_body_bytes(self) -> int:
        """Hard memory bound for any JSON body, sized so every VALID request fits whatever escaping the client used."""
        return (JSON_BYTES_PER_CHARACTER * self.max_jd_chars + JSON_BYTES_PER_CANDIDATE * self.max_candidates
                + JSON_SLACK_BYTES)


def get_runtime(request: Request) -> ApiRuntime:
    return request.app.state.runtime


def get_api_limits(request: Request) -> ApiLimits:
    return request.app.state.limits_provider()


def get_ingestion_service(request: Request):
    return request.app.state.runtime.ingestion


def get_matching_service(request: Request):
    return request.app.state.runtime.matching


def get_readiness(request: Request):
    return request.app.state.runtime.gate()


def get_caller(request: Request) -> str:
    caller = getattr(request.state, "caller", None)         # set by the auth gate from the matched API key, nowhere else
    if not caller:
        raise ApiError(401, "UNAUTHENTICATED", "A valid API key is required.", headers={"WWW-Authenticate": "Bearer"})
    return caller


def get_token_signer(request: Request) -> CandidateTokenSigner:
    try:
        signer = request.app.state.runtime.settings().signer
    except (SettingsError, TokenConfigurationError):
        signer = None
    if signer is None:
        raise ApiError(503, "SERVICE_UNAVAILABLE", "The service is not available right now.")   # fail closed
    return signer

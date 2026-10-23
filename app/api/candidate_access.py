"""Candidate access tokens: the API's authorization boundary for resume references.

The repository has no user model, so a caller-supplied document id or namespace cannot be treated as authorized.
Instead the server issues an unforgeable capability when it ingests a resume:
    access_token = HMAC-SHA256(server secret, "candidate-access:v1:" + document_id)
Matching accepts (document_id, access_token) pairs only; the namespace is always derived server-side from a
verified document id and is never read from the request. This is a bearer capability, NOT user authentication:
whoever holds the token can match that resume. It is layered under API-key authentication (see auth.py), which
identifies the caller; real identity-provider integration is still required for a production deployment.
"""

import hashlib
import hmac
import os

MIN_SECRET_LENGTH = 32
SECRET_ENV = "CANDIDATE_TOKEN_SECRET"
CONTEXT = b"candidate-access:v1:"


class TokenConfigurationError(Exception):
    """The signing secret is missing or too short. The API fails closed rather than issuing weak tokens."""


class CandidateTokenSigner:
    def __init__(self, secret: str | bytes):
        raw = secret.encode("utf-8") if isinstance(secret, str) else bytes(secret)
        if len(raw) < MIN_SECRET_LENGTH:
            raise TokenConfigurationError(f"{SECRET_ENV} must be at least {MIN_SECRET_LENGTH} characters")
        self._secret = raw

    @classmethod
    def from_env(cls) -> "CandidateTokenSigner":
        secret = os.environ.get(SECRET_ENV)
        if not secret:
            raise TokenConfigurationError(f"{SECRET_ENV} is not set")
        return cls(secret)

    def issue(self, document_id: str) -> str:
        return hmac.new(self._secret, CONTEXT + document_id.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify(self, document_id, token) -> bool:
        """False for anything malformed (non-string, non-ASCII, wrong length); never raises."""
        if not isinstance(document_id, str) or not isinstance(token, str):
            return False
        try:
            presented = token.encode("ascii")
        except UnicodeEncodeError:
            return False
        return hmac.compare_digest(self.issue(document_id).encode("ascii"), presented)      # constant time, bytes

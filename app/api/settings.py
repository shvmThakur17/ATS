"""All API configuration, read once.

`.env` is loaded a single time (never overriding the process environment) the first time settings are needed, not at
import and not per request. Secrets live in fields excluded from repr so they cannot reach logs by accident.
A configuration problem raises SettingsError, which the API reports as a sanitized 503 (fail closed).
"""

from dataclasses import dataclass, field
import hmac
from hashlib import sha256
import os
from pathlib import Path
import re
import threading
from typing import Mapping
from urllib.parse import urlsplit

from ..ingestion.pdf_resume_ingestion_service import DEFAULT_MAX_PDF_BYTES, IngestionLimits
from ..matching.pinecone_match_service import PineconeMatchLimits
from .candidate_access import CandidateTokenSigner, TokenConfigurationError

ROOT = Path(__file__).resolve().parents[2]
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
MAX_KEY_LENGTH = 256


class SettingsError(Exception):
    """Configuration is missing or invalid. Messages name the variable, never its value."""


def parse_int(environ: Mapping, name: str, default: int, low: int, high: int) -> int:
    raw = environ.get(name)
    if raw in (None, ""):
        return default
    try:
        value = int(raw)
    except ValueError:
        raise SettingsError(f"{name} must be an integer") from None
    if not low <= value <= high:
        raise SettingsError(f"{name} must be between {low} and {high}")
    return value


def parse_api_keys(raw: str | None) -> dict[str, str]:
    """`caller=sha256hex,caller2=sha256hex`. Only hashes are configured, never the keys themselves."""
    keys: dict[str, str] = {}
    for entry in (raw or "").split(","):
        if not entry.strip():
            continue
        caller, _, digest = entry.strip().partition("=")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", caller) or not SHA256_HEX.match(digest):
            raise SettingsError("API_KEY_HASHES entries must look like name=<64 lowercase hex sha256>")
        if caller in keys:
            raise SettingsError("API_KEY_HASHES lists the same caller twice")
        keys[caller] = digest
    return keys


def parse_origins(raw: str | None) -> tuple[str, ...]:
    origins = []
    for entry in (raw or "").split(","):
        origin = entry.strip()
        if not origin:
            continue
        parts = urlsplit(origin)
        if origin == "*" or parts.scheme not in ("http", "https") or not parts.netloc or parts.path not in ("", "/") \
                or parts.query or parts.fragment or "*" in origin:
            raise SettingsError("API_CORS_ORIGINS must list exact http(s) origins; wildcards are not allowed")
        origins.append(f"{parts.scheme}://{parts.netloc}")
    return tuple(dict.fromkeys(origins))


class ApiKeyAuthenticator:
    """Bearer API keys, compared by SHA-256 digest. The caller identity comes only from the matching key."""

    def __init__(self, hashes: Mapping[str, str]):
        self.hashes = dict(hashes)

    def authenticate(self, header: str | None) -> str | None:
        if not isinstance(header, str):
            return None
        scheme, _, key = header.strip().partition(" ")
        key = key.strip()
        if scheme.lower() != "bearer" or not key or len(key) > MAX_KEY_LENGTH:
            return None
        digest = sha256(key.encode("utf-8", "replace")).hexdigest()
        caller = None
        for name, expected in self.hashes.items():          # no early exit: time does not depend on which key matched
            if hmac.compare_digest(digest.encode(), expected.encode()):
                caller = name
        return caller


@dataclass(frozen=True)
class ApiSettings:
    api_key_hashes: Mapping[str, str] = field(default_factory=dict, repr=False)
    token_secret: str | None = field(default=None, repr=False)
    rate_limit_per_minute: int = 60
    upload_quota_per_day: int = 100
    cors_origins: tuple[str, ...] = ()
    auth_disabled: bool = False             # local/internal use only: no API key required (API_AUTH_DISABLED=true)
    max_upload_bytes: int = DEFAULT_MAX_PDF_BYTES
    max_chunks: int = 200
    max_retrieval_depth: int = 200
    parse_timeout_seconds: int = 30
    parse_memory_mb: int = 1024
    job_timeout_seconds: int = 180
    max_pending_jobs: int = 8
    readiness_timeout_seconds: int = 10
    readiness_poll_seconds: float = 0.5

    def __post_init__(self):
        if self.max_chunks > self.max_retrieval_depth:
            raise SettingsError("INGEST_MAX_CHUNKS must not exceed M13_MAX_RETRIEVAL_DEPTH: an accepted resume "
                                "could otherwise never be matched")

    @property
    def authenticator(self) -> ApiKeyAuthenticator | None:
        return ApiKeyAuthenticator(self.api_key_hashes) if self.api_key_hashes else None

    @property
    def signer(self) -> CandidateTokenSigner | None:
        return CandidateTokenSigner(self.token_secret) if self.token_secret else None

    @classmethod
    def from_env(cls, environ: Mapping | None = None) -> "ApiSettings":
        env = os.environ if environ is None else environ
        try:
            ingestion = IngestionLimits.from_env() if environ is None else IngestionLimits(
                parse_int(env, "INGEST_MAX_PDF_BYTES", DEFAULT_MAX_PDF_BYTES, 1, 1 << 30),
                parse_int(env, "INGEST_MAX_CHUNKS", IngestionLimits().max_chunks, 1, 1000))
            depth = PineconeMatchLimits.from_env().max_retrieval_depth if environ is None else parse_int(
                env, "M13_MAX_RETRIEVAL_DEPTH", PineconeMatchLimits().max_retrieval_depth, 1, 1000)
        except SettingsError:
            raise
        except Exception:
            raise SettingsError("an ingestion or matching limit is invalid") from None
        secret = env.get("CANDIDATE_TOKEN_SECRET") or None
        if secret is not None:
            try:
                CandidateTokenSigner(secret)                  # validates the length without exposing the value
            except TokenConfigurationError:
                raise SettingsError("CANDIDATE_TOKEN_SECRET is too short") from None
        return cls(
            api_key_hashes=parse_api_keys(env.get("API_KEY_HASHES")), token_secret=secret,
            rate_limit_per_minute=parse_int(env, "API_RATE_LIMIT_PER_MINUTE", 60, 1, 100_000),
            upload_quota_per_day=parse_int(env, "API_UPLOAD_QUOTA_PER_DAY", 100, 1, 1_000_000),
            cors_origins=parse_origins(env.get("API_CORS_ORIGINS")),
            auth_disabled=(env.get("API_AUTH_DISABLED") or "").strip().lower() in ("1", "true", "yes", "on"),
            max_upload_bytes=ingestion.max_pdf_bytes,
            max_chunks=ingestion.max_chunks, max_retrieval_depth=depth,
            parse_timeout_seconds=parse_int(env, "API_PARSE_TIMEOUT_SECONDS", 30, 1, 3600),
            parse_memory_mb=parse_int(env, "API_PARSE_MEMORY_MB", 1024, 64, 65_536),
            job_timeout_seconds=parse_int(env, "API_JOB_TIMEOUT_SECONDS", 180, 1, 7200),
            max_pending_jobs=parse_int(env, "API_MAX_PENDING_JOBS", 8, 1, 1000),
            readiness_timeout_seconds=parse_int(env, "API_READINESS_TIMEOUT_SECONDS", 10, 0, 300))


_lock = threading.Lock()
_loaded: ApiSettings | None = None


def load_settings() -> ApiSettings:
    """Load .env once, build the settings once, reuse them for every request. A failure is not cached."""
    global _loaded
    with _lock:
        if _loaded is None:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env", override=False)
            try:
                _loaded = ApiSettings.from_env()
            except SettingsError:
                raise
            except Exception:
                raise SettingsError("configuration could not be read") from None
        return _loaded

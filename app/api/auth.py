"""API-key authentication and per-caller rate limiting, enforced before the request body is read.

DEVELOPMENT-SAFE BOUNDARY, NOT AN IDENTITY PROVIDER. Callers present `Authorization: Bearer <api key>`; the server
holds only SHA-256 hashes of the keys (`API_KEY_HASHES=name=<hash>,...`) and derives the caller name from the key
that matched. No identity header sent by the client (X-User, X-Forwarded-For, ...) is ever trusted. With no keys
configured the API fails closed (503). Keys are static and shared per caller: rotate them by changing the hashes,
and put a real identity provider (OIDC/JWT) in front before treating this API as production-secure.
"""

import math

from .errors import asgi_error

PROTECTED_SUFFIXES = ("/resume/upload", "/jd/match", "/jd/rank")      # suffix match: independent of any mount prefix


def is_protected(method: str, path: str) -> bool:
    return method != "OPTIONS" and path.rstrip("/").endswith(PROTECTED_SUFFIXES)


class AuthGateMiddleware:
    def __init__(self, app, settings_provider, limiter_provider):
        self.app, self.settings_provider, self.limiter_provider = app, settings_provider, limiter_provider

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not is_protected(scope["method"], scope["path"]):
            return await self.app(scope, receive, send)
        try:
            settings = self.settings_provider()
            authenticator, disabled = settings.authenticator, settings.auth_disabled
        except Exception:
            authenticator, disabled = None, False
        if disabled:                              # explicit opt-in (API_AUTH_DISABLED): one shared anonymous caller,
            caller = "anonymous"                  # still rate limited, so the limits below keep working
            wait = self.limiter_provider().acquire(caller)
            if wait > 0:
                return await asgi_error(send, 429, "RATE_LIMITED", "Too many requests. Try again later.",
                                        {"Retry-After": max(1, math.ceil(wait))})
            scope.setdefault("state", {})["caller"] = caller
            return await self.app(scope, receive, send)
        if authenticator is None:                                  # fail closed: never open when unconfigured
            return await asgi_error(send, 503, "SERVICE_UNAVAILABLE", "The service is not available right now.")
        header = dict(scope["headers"]).get(b"authorization")
        caller = authenticator.authenticate(header.decode("latin-1") if header else None)
        if caller is None:
            return await asgi_error(send, 401, "UNAUTHENTICATED", "A valid API key is required.",
                                    {"WWW-Authenticate": "Bearer"})
        wait = self.limiter_provider().acquire(caller)
        if wait > 0:
            return await asgi_error(send, 429, "RATE_LIMITED", "Too many requests. Try again later.",
                                    {"Retry-After": max(1, math.ceil(wait))})
        scope.setdefault("state", {})["caller"] = caller
        await self.app(scope, receive, send)

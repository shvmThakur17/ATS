"""Registers the routers, error handlers, middleware and runtime on a FastAPI app.

`configure_app` takes the app as an argument so tests can build isolated apps; `app.main` calls it for the real one.
Middleware order, outermost first: CORS allow-list, authentication + rate limit, request-body cap.
"""

from typing import Callable

from fastapi import FastAPI

from .auth import AuthGateMiddleware
from .body_limit import BodySizeLimitMiddleware
from .cors import AllowlistCorsMiddleware
from .dependencies import ApiLimits
from .errors import register_error_handlers
from .runtime import ApiRuntime
from .settings import SettingsError
from .v1 import jd_matching, jd_pool, resume_upload

UPLOAD_SUFFIX = "/resume/upload"
BODY_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def configure_app(app: FastAPI, limits_provider: Callable[[], ApiLimits] | None = None,
                  runtime: ApiRuntime | None = None) -> FastAPI:
    runtime = runtime or ApiRuntime()
    app.state.runtime = runtime

    def default_limits() -> ApiLimits:
        try:
            return ApiLimits(max_upload_bytes=runtime.settings().max_upload_bytes)
        except SettingsError:
            return ApiLimits()                      # a bad configuration must not remove the cap

    def provider() -> ApiLimits:
        try:
            return (limits_provider or default_limits)()
        except Exception:
            return ApiLimits()

    app.state.limits_provider = provider
    register_error_handlers(app)
    app.include_router(resume_upload.router, prefix="/api/v1/resume", tags=["Resume"])
    app.include_router(jd_matching.router, prefix="/api/v1/jd", tags=["Job Description"])
    app.include_router(jd_pool.router, prefix="/api/v1/jd", tags=["Job Description"])

    def cap(method: str, path: str) -> int | None:
        """Every body-carrying request is capped; only the upload path gets the larger multipart allowance. Matching
        the path by suffix keeps the cap in force when the app is mounted under a prefix."""
        if method not in BODY_METHODS:
            return None
        if path.rstrip("/").endswith(UPLOAD_SUFFIX):
            return provider().max_upload_body_bytes
        return provider().max_json_body_bytes

    def settings_or_none():
        try:
            return runtime.settings()
        except Exception:
            return None

    def origins() -> tuple:
        settings = settings_or_none()
        return settings.cors_origins if settings else ()

    app.add_middleware(BodySizeLimitMiddleware, limit_for=cap)
    app.add_middleware(AuthGateMiddleware, settings_provider=runtime.settings, limiter_provider=runtime.rate_limiter)
    app.add_middleware(AllowlistCorsMiddleware, origins_provider=origins)
    return app

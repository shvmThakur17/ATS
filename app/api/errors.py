"""One sanitized error envelope for the whole API: {"error": {"code", "message", "details"?}}.

Messages are fixed strings. Exception text, stack traces, request bodies, file names and validation inputs are
never echoed. Framework validation errors normally repeat the offending input and the caller's field names, so
they are rewritten here: only field names the API itself defines are reported.
"""

import json
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("app.api")

STATUS_CODES = {400: "BAD_REQUEST", 401: "UNAUTHENTICATED", 404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED",
                413: "PAYLOAD_TOO_LARGE", 415: "UNSUPPORTED_MEDIA_TYPE", 422: "INVALID_REQUEST", 429: "RATE_LIMITED"}
# The only request field names this API defines. Anything else a caller invents is reported as "(unknown)".
KNOWN_FIELDS = frozenset({"file", "job_description", "candidates", "document_id", "access_token"})
SAFE_HEADERS = frozenset({"allow", "retry-after", "www-authenticate"})


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details: dict | None = None,
                 headers: dict | None = None):
        super().__init__(code)
        self.status, self.code, self.message, self.details, self.headers = status, code, message, details, headers


def envelope(status: int, code: str, message: str, details: dict | None = None,
             headers: dict | None = None) -> JSONResponse:
    body = {"code": code, "message": message}
    if details:
        body["details"] = details
    return JSONResponse(status_code=status, content={"error": body},
                        headers={k: str(v) for k, v in headers.items()} if headers else None)


async def asgi_error(send, status: int, code: str, message: str, headers: dict | None = None) -> None:
    """The same envelope for pure-ASGI middleware, which has no Response object to return."""
    body = json.dumps({"error": {"code": code, "message": message}}).encode()
    sent = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
            (b"cache-control", b"no-store")]
    sent += [(name.lower().encode(), str(value).encode()) for name, value in (headers or {}).items()]
    await send({"type": "http.response.start", "status": status, "headers": sent})
    await send({"type": "http.response.body", "body": body})


def field_path(location) -> str:
    parts = []
    for part in location:
        if part == "body":
            continue
        parts.append(str(part) if isinstance(part, int) or part in KNOWN_FIELDS else "(unknown)")
    return ".".join(parts)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error(_: Request, error: ApiError):
        return envelope(error.status, error.code, error.message, error.details, error.headers)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, error: RequestValidationError):
        fields = [{"field": field_path(item["loc"]), "problem": item["type"]} for item in error.errors()][:20]
        return envelope(422, "INVALID_REQUEST", "The request is missing required data or contains invalid data.",
                        {"fields": fields})

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, error: StarletteHTTPException):
        code = STATUS_CODES.get(error.status_code, "HTTP_ERROR")
        headers = {k: v for k, v in (error.headers or {}).items() if k.lower() in SAFE_HEADERS}   # keeps Allow
        return envelope(error.status_code, code, "The request could not be processed.", headers=headers or None)

    @app.exception_handler(Exception)
    async def unexpected(_: Request, error: Exception):
        logger.error("unhandled error type=%s", type(error).__name__)     # type only: messages can hold resume text
        return envelope(500, "INTERNAL_ERROR", "An internal error occurred.")

"""Request-body size cap enforced while the body streams in, before it is parsed or buffered.

FastAPI parses multipart uploads (spooling them to memory/disk) before a route runs, so a size check inside the
route is too late. This pure-ASGI middleware rejects a declared oversize Content-Length immediately and aborts a
streamed or chunked body once it passes the cap. The abort is a BaseException so FastAPI's "error parsing the body"
handling (which catches Exception) cannot turn it into a different status.
"""

import json
from typing import Callable

CAP_FOR_PATH = Callable[[str, str], int | None]


class BodyTooLarge(BaseException):
    pass


class BodySizeLimitMiddleware:
    def __init__(self, app, limit_for: CAP_FOR_PATH):
        self.app, self.limit_for = app, limit_for

    async def __call__(self, scope, receive, send):
        limit = self.limit_for(scope["method"], scope["path"]) if scope["type"] == "http" else None
        if limit is None:
            return await self.app(scope, receive, send)
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None:
            try:
                if int(declared) > limit:
                    return await self.reject(send)
            except ValueError:
                return await self.reject(send, 400)
        received, started = 0, False

        async def counted_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise BodyTooLarge()
            return message

        async def tracked_send(message):
            nonlocal started
            started = started or message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, counted_receive, tracked_send)
        except BodyTooLarge:
            if started:
                raise
            await self.reject(send)

    @staticmethod
    async def reject(send, status: int = 413):
        code, message = (("PAYLOAD_TOO_LARGE", "The request body exceeds the allowed size.") if status == 413
                         else ("BAD_REQUEST", "The request could not be processed."))
        body = json.dumps({"error": {"code": code, "message": message}}).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
                                (b"connection", b"close")]})
        await send({"type": "http.response.body", "body": body})

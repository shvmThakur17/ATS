"""CORS for exactly the configured frontend origins (API_CORS_ORIGINS). Never a wildcard, never credentials.

Authentication is a bearer header, not a cookie, so `Access-Control-Allow-Credentials` is never sent. An origin that
is not on the list gets no CORS headers at all, which makes browsers refuse the response.
"""

ALLOWED_METHODS = b"GET, POST, OPTIONS"
ALLOWED_HEADERS = b"Authorization, Content-Type"


class AllowlistCorsMiddleware:
    def __init__(self, app, origins_provider):
        self.app, self.origins_provider = app, origins_provider

    def allowed(self, origin: str) -> bool:
        try:
            return origin in self.origins_provider()
        except Exception:
            return False

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        raw = headers.get(b"origin")
        origin = raw.decode("latin-1") if raw else None
        if not origin or not self.allowed(origin):
            return await self.app(scope, receive, send)
        grant = [(b"access-control-allow-origin", origin.encode("latin-1")), (b"vary", b"Origin")]
        if scope["method"] == "OPTIONS" and b"access-control-request-method" in headers:
            await send({"type": "http.response.start", "status": 204, "headers": grant + [
                (b"access-control-allow-methods", ALLOWED_METHODS), (b"access-control-allow-headers", ALLOWED_HEADERS),
                (b"access-control-max-age", b"600"), (b"content-length", b"0")]})
            return await send({"type": "http.response.body", "body": b""})

        async def with_cors(message):
            if message["type"] == "http.response.start":
                message = {**message, "headers": [*message.get("headers", []), *grant,
                                                  (b"access-control-expose-headers", b"Retry-After")]}
            await send(message)

        await self.app(scope, receive, with_cors)

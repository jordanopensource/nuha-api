"""Tests for app.common.http: shared ASGI middleware + exception handlers.

Covers the 413 body-size cap, the security-headers middleware, the 422
input-stripping handler, and the generic 500. All pure-ASGI / pure-pydantic;
nothing here touches the ML stack or the app's routes.
"""

import asyncio


# =============================================================================
# BodySizeLimitMiddleware (413 body cap, no buffering)
# =============================================================================


class TestBodySizeLimit:
    """The ASGI body-size middleware rejects oversized bodies with 413 without
    buffering them, so a directly-exposed backend can't be OOM'd by a huge POST."""

    def _mini_app(self, max_bytes):
        """Wrap a trivial ASGI app that would happily read any body."""
        from app.common.http import BodySizeLimitMiddleware

        async def echo_app(scope, receive, send):
            body = b""
            while True:
                message = await receive()
                body += message.get("body", b"")
                if not message.get("more_body"):
                    break
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"text/plain")],
                }
            )
            await send({"type": "http.response.body", "body": b"ok:%d" % len(body)})

        return BodySizeLimitMiddleware(echo_app, max_bytes=max_bytes)

    def _request(self, app, body: bytes, content_length: int | None):
        """Drive the ASGI app with one request; return (status, sent messages)."""
        headers = []
        if content_length is not None:
            headers.append((b"content-length", str(content_length).encode()))
        scope = {"type": "http", "method": "POST", "path": "/", "headers": headers}
        messages = [
            {"type": "http.request", "body": body[i : i + 10], "more_body": i + 10 < len(body)}
            for i in range(0, max(len(body), 1), 10)
        ]
        sent = []

        async def receive():
            return messages.pop(0)

        async def send(message):
            sent.append(message)

        asyncio.run(app(scope, receive, send))
        status = next(m["status"] for m in sent if m["type"] == "http.response.start")
        return status, sent

    def test_declared_content_length_over_cap_rejected(self):
        """An oversized Content-Length is rejected up front, body unread."""
        app = self._mini_app(max_bytes=50)
        status, _ = self._request(app, b"x" * 100, content_length=100)
        assert status == 413

    def test_streamed_body_over_cap_rejected(self):
        """A chunked body (no Content-Length) is cut off once it passes the cap;
        this drives the middleware directly so it exercises its own 413 backstop
        for a non-FastAPI mount. Either way the body is never fully read, the
        memory-safety guarantee."""
        app = self._mini_app(max_bytes=50)
        status, _ = self._request(app, b"x" * 100, content_length=None)
        assert status == 413

    def test_streamed_body_over_cap_under_fastapi_is_413(self):
        """Under FastAPI the mid-read abort must ALSO surface as the contract
        413, not FastAPI's generic 400 body-parse error: BodyTooLargeError
        subclasses HTTPException(413) precisely so the body-parse guard
        re-raises it instead of swallowing it."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from app.common.http import (
            DETAIL_BODY_TOO_LARGE,
            BodySizeLimitMiddleware,
            install_common_handlers,
        )

        app = FastAPI()
        install_common_handlers(app)
        app.add_middleware(BodySizeLimitMiddleware, max_bytes=50)

        @app.post("/echo")
        async def echo(body: dict):
            return body

        client = TestClient(app)
        resp = client.post(
            "/echo",
            content=iter([b'{"k": "', b"x" * 100, b'"}']),  # chunked: no Content-Length
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 413
        assert resp.json() == {"detail": DETAIL_BODY_TOO_LARGE}

    def test_body_under_cap_passes(self):
        """A body under the cap reaches the app untouched."""
        app = self._mini_app(max_bytes=50)
        status, sent = self._request(app, b"x" * 30, content_length=30)
        assert status == 200
        assert any(b"ok:30" in m.get("body", b"") for m in sent)

    def test_non_http_scope_passes_through(self):
        """A non-HTTP scope (e.g. lifespan) is forwarded untouched."""
        from app.common.http import BodySizeLimitMiddleware

        seen = {}

        async def app(scope, receive, send):
            seen["type"] = scope["type"]

        mw = BodySizeLimitMiddleware(app, max_bytes=10)
        asyncio.run(mw({"type": "lifespan"}, None, None))
        assert seen["type"] == "lifespan"


# =============================================================================
# SecurityHeadersMiddleware
# =============================================================================


class TestSecurityHeaders:
    """The middleware sets the response security headers (nosniff, DENY)
    without clobbering one the app already set."""

    async def _headers_for(self, app_headers):
        from app.common.http import SecurityHeadersMiddleware

        async def inner(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": list(app_headers)})
            await send({"type": "http.response.body", "body": b"ok"})

        mw = SecurityHeadersMiddleware(inner)
        sent = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        await mw({"type": "http", "headers": []}, receive, send)
        start = next(m for m in sent if m["type"] == "http.response.start")
        return {k.lower(): v for k, v in start["headers"]}

    def test_adds_headers(self):
        headers = asyncio.run(self._headers_for([(b"content-type", b"application/json")]))
        assert headers[b"x-content-type-options"] == b"nosniff"
        assert headers[b"x-frame-options"] == b"DENY"

    def test_does_not_duplicate_existing(self):
        headers = asyncio.run(self._headers_for([(b"x-frame-options", b"SAMEORIGIN")]))
        # The app's own value wins; the middleware does not add a second one.
        assert headers[b"x-frame-options"] == b"SAMEORIGIN"

    def test_passes_non_http_scope(self):
        from app.common.http import SecurityHeadersMiddleware

        seen = []

        async def inner(scope, receive, send):
            seen.append(scope["type"])

        async def receive():
            return {}

        async def send(message):
            pass

        asyncio.run(SecurityHeadersMiddleware(inner)({"type": "lifespan"}, receive, send))
        assert seen == ["lifespan"]


# =============================================================================
# Shared exception handlers: 413 / 422-strip / 500
# =============================================================================


def _handlers_app():
    """A tiny FastAPI app wired with install_common_handlers + routes that trigger
    each handler, so the handlers are tested without either real service."""
    from fastapi import FastAPI
    from pydantic import BaseModel

    from app.common.http import install_common_handlers

    app = FastAPI()
    install_common_handlers(app)

    class Body(BaseModel):
        value: int

    @app.post("/validate")
    async def validate(body: Body):
        return {"ok": body.value}

    @app.get("/boom")
    async def boom():
        raise ValueError("secret internal detail")

    return app


class TestCommonHandlers:
    def _client(self):
        from fastapi.testclient import TestClient

        return TestClient(_handlers_app(), raise_server_exceptions=False)

    def test_422_strips_input_and_ctx(self):
        """The 422 body keeps loc/msg/type but never echoes the rejected input."""
        client = self._client()
        marker = "SENTINEL-DO-NOT-ECHO"
        resp = client.post("/validate", json={"value": marker})
        assert resp.status_code == 422
        assert marker not in resp.text
        errors = resp.json()["detail"]
        assert isinstance(errors, list) and errors
        for err in errors:
            assert set(err) == {"loc", "msg", "type"}

    def test_500_is_generic_no_leak(self):
        client = self._client()
        resp = client.get("/boom")
        assert resp.status_code == 500
        assert resp.json() == {"detail": "Internal server error"}
        assert "secret internal detail" not in resp.text
        assert "Traceback" not in resp.text

    def test_body_too_large_handler_returns_413(self):
        """The BodyTooLargeError handler maps a mid-read overflow to a 413 body."""
        from app.common.http import BodyTooLargeError, body_too_large_handler

        resp = asyncio.run(body_too_large_handler(None, BodyTooLargeError()))
        assert resp.status_code == 413

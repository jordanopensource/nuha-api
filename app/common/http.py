"""Shared ASGI middleware and exception handlers.

- ``BodySizeLimitMiddleware``: pure-ASGI 413 body cap.
- ``install_common_handlers``: the 413 / 422-input-stripping / 500 handlers.
  The app adds its own 503/504 handlers on top.
- ``DETAIL_*``: the frozen public error strings, defined once so handlers and
  tests can never drift apart.
- ``HEADER_REQUEST_ID``: the correlation-id header name, minted or echoed by
  the app and safe to log.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


logger = logging.getLogger(__name__)

# The frozen public error bodies ({"detail": <string>}). Single source of truth:
# the handlers below, the app's own 503/504 handlers, and the tests import these.
DETAIL_OVERLOADED = "Service temporarily overloaded, try again shortly"
DETAIL_TIMEOUT = "Inference timed out, try again shortly"
DETAIL_INTERNAL = "Internal server error"
DETAIL_BODY_TOO_LARGE = "Request body too large"

# Correlation id: accepted from the client when it matches the app's strict
# token pattern, minted otherwise, echoed on responses that reach a route.
HEADER_REQUEST_ID = "X-Request-Id"


class BodyTooLargeError(StarletteHTTPException):
    """Raised mid-read when a request body exceeds the cap.

    Subclasses HTTPException(413) because FastAPI's body-parse guard re-raises
    HTTPExceptions untouched but collapses any other exception into a generic
    400; this keeps a chunked (or lying-Content-Length) oversize body on the
    contract 413, same as the up-front Content-Length rejection.
    """

    def __init__(self) -> None:
        super().__init__(status_code=413, detail=DETAIL_BODY_TOO_LARGE)


_TOO_LARGE_RESPONSE = {
    "type": "http.response.start",
    "status": 413,
    "headers": [(b"content-type", b"application/json")],
}
_TOO_LARGE_BODY = {
    "type": "http.response.body",
    "body": f'{{"detail":"{DETAIL_BODY_TOO_LARGE}"}}'.encode(),
}


class BodySizeLimitMiddleware:
    """Reject request bodies over ``max_bytes`` with a 413, without buffering.

    Pure ASGI (no BaseHTTPMiddleware), so the overhead is one header scan per
    request plus an integer add per body chunk; nothing is copied or buffered.
    Two layers: a declared Content-Length over the cap is refused up front;
    chunked bodies (or a lying Content-Length) are counted as they stream and the
    read raises ``BodyTooLargeError`` the moment the total passes the cap, so the
    oversized body is never fully read. Under FastAPI the registered handler
    turns that abort into the contract 413; this middleware's own 413 send is
    the backstop for a non-FastAPI mount.
    """

    def __init__(self, app, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        for name, value in scope["headers"]:
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    break  # malformed; the server/app will reject it
                if declared > self.max_bytes:
                    await send(_TOO_LARGE_RESPONSE)
                    await send(_TOO_LARGE_BODY)
                    return
                break

        received = 0
        response_started = False

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise BodyTooLargeError()
            return message

        async def tracking_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except BodyTooLargeError:
            if response_started:
                raise
            await tracking_send(_TOO_LARGE_RESPONSE)
            await tracking_send(_TOO_LARGE_BODY)


class SecurityHeadersMiddleware:
    """Set response security headers on the public API.

    Sets ``X-Content-Type-Options: nosniff`` and ``X-Frame-Options: DENY``
    (they matter most for the HTML docs pages). Pure ASGI: appends the headers
    on ``http.response.start`` unless the app already set them.
    """

    _HEADERS = ((b"x-content-type-options", b"nosniff"), (b"x-frame-options", b"DENY"))

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                present = {name.lower() for name, _ in headers}
                for name, value in self._HEADERS:
                    if name not in present:
                        headers.append((name, value))
            await send(message)

        await self.app(scope, receive, send_wrapper)


async def body_too_large_handler(request: Request, exc: BodyTooLargeError):
    """Return 413 when a streamed body passes the cap mid-read."""
    return JSONResponse(status_code=413, content={"detail": DETAIL_BODY_TOO_LARGE})


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return 422 without echoing the rejected input.

    FastAPI's default validation response includes the offending value (its
    ``input``/``ctx`` fields). Inputs here are abuse text and can be ~10 MB
    batches, so reflect only the machine-readable location and reason.
    """
    errors = [
        {"loc": err.get("loc", ()), "msg": err.get("msg", ""), "type": err.get("type", "")}
        for err in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


async def generic_exception_handler(request: Request, exc: Exception):
    """Prevent unhandled exceptions from leaking internal details to clients."""
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": DETAIL_INTERNAL})


def install_common_handlers(app: FastAPI) -> None:
    """Register the shared 413 / 422-strip / 500 handlers."""
    app.add_exception_handler(BodyTooLargeError, body_too_large_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

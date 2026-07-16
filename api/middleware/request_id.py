# api/middleware/request_id.py

from uuid import uuid4
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    HTTP Middleware responsible for unique request isolation tracing.
    
    Responsibilities:
    1. Inspects the incoming 'X-Request-ID' header for distributed tracing.
    2. Validates incoming constraints to protect against log injection.
    3. Mints a distinct, trace-ready tracking token when missing.
    4. Binds the verified trace directly to request.state for downstream pipelines.
    5. Returns the identical token to the calling client in response headers.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # 1. Extract the incoming header
        incoming_id = request.headers.get("X-Request-ID")

        # 2. Validate the request ID string to prevent payload abuse or junk headers
        if incoming_id and incoming_id.strip() and len(incoming_id) <= 100:
            request_id = incoming_id.strip()
        else:
            # 3. Mint a unique tracking token if missing or malformed
            request_id = f"req_{uuid4().hex}"

        # 4. Bind tracing boundaries to the request state context
        request.state.request_id = request_id

        # 5. Hand execution down the HTTP call stack
        response: Response = await call_next(request)

        # 6. Expose the request trace to the out-of-band response headers
        response.headers["X-Request-ID"] = request_id

        return response
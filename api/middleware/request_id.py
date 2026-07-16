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
        
        incoming_id = request.headers.get("X-Request-ID")

       
        if incoming_id and incoming_id.strip() and len(incoming_id) <= 100:
            request_id = incoming_id.strip()
        else:
            
            request_id = f"req_{uuid4().hex}"

       
        request.state.request_id = request_id

        
        response: Response = await call_next(request)

        
        response.headers["X-Request-ID"] = request_id

        return response
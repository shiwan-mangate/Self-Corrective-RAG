## api/middleware/logging.py
import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    HTTP Middleware responsible for capturing boundary-level request telemetry.
    
    Responsibilities:
    1. Extracts the distributed request_id from the state.
    2. Measures exact wall-clock latency via high-resolution performance counters.
    3. Safely executes the downstream router/graph.
    4. Captures the final HTTP status code (defaulting to 500 on application crashes).
    5. Emits a sanitized, predictable, one-line log summary.
    6. Strictly avoids logging PII, secrets, document contents, or query strings.
    """

    async def dispatch(self, request: Request, call_next):
        
        request_id = getattr(request.state, "request_id", "unknown")
        
        
        method = request.method
        path = request.url.path
        
    
        start_time = time.perf_counter()
        
        
        status_code = 500 

        try:
           
            response = await call_next(request)
            status_code = response.status_code
            return response

        except Exception:
           
            status_code = 500
        
            raise

        finally:
           
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            
            log_message = (
                f"request_id={request_id} | "
                f"method={method} | "
                f"path={path} | "
                f"status_code={status_code} | "
                f"latency_ms={latency_ms:.2f}"
            )

           
            if status_code >= 500:
                logger.error(log_message)
            else:
                logger.info(log_message)
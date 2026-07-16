## api/middleware/exception_handler.py
import logging
from typing import Any, Dict, Optional

from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError


from shared.exceptions import (
    SelfHealingRAGError,
    SessionNotFoundError,
    GenerationExecutionError,
    EvaluationExecutionError,
)
from api.exceptions import EvaluationNotFoundError


logger = logging.getLogger(__name__)


EXCEPTION_MAPPINGS = {
    SessionNotFoundError: {
        "status_code": 404,
        "error_code": "SESSION_NOT_FOUND"
    },
    EvaluationNotFoundError: {
        "status_code": 404,
        "error_code": "EVALUATION_NOT_FOUND"
    },
    GenerationExecutionError: {
        "status_code": 503,
        "error_code": "GENERATION_SERVICE_UNAVAILABLE"
    },
    EvaluationExecutionError: {
        "status_code": 503,
        "error_code": "EVALUATION_SERVICE_UNAVAILABLE"
    },
    SelfHealingRAGError: {
        "status_code": 500,
        "error_code": "APPLICATION_ERROR"
    }
}


def _get_request_id(request: Request) -> str:
    """Safely extracts the tracing ID injected by RequestIDMiddleware."""
    return getattr(request.state, "request_id", "unknown")


def _build_error_response(
    status_code: int, 
    error_code: str, 
    message: str, 
    request_id: str, 
    details: Optional[Any] = None
) -> JSONResponse:
    """Standardizes the JSON error payload across the entire API."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error_code": error_code,
            "message": message,
            "request_id": request_id,
            "details": details
        }
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Catches FastAPI/Pydantic schema validation errors.
    Returns HTTP 422 with unpacked error details for the frontend.
    """
    request_id = _get_request_id(request)

    logger.warning(f"Request validation failed | RequestID={request_id} | Path={request.url.path}")

    return _build_error_response(
        status_code=422,
        error_code="VALIDATION_ERROR",
        message="The request payload failed schema validation.",
        request_id=request_id,
        details=exc.errors() 
    )


async def application_exception_handler(request: Request, exc: SelfHealingRAGError) -> JSONResponse:
    """
    Catches known, custom domain exceptions from shared/exceptions.py.
    Maps them to semantic HTTP status codes using the mapping registry.
    """
    request_id = _get_request_id(request)
    
    mapping = EXCEPTION_MAPPINGS.get(type(exc), EXCEPTION_MAPPINGS[SelfHealingRAGError])
    
    logger.error(
        f"Domain Exception Caught | RequestID={request_id} | "
        f"Error={exc.__class__.__name__} | Message={str(exc)}"
    )

    return _build_error_response(
        status_code=mapping["status_code"],
        error_code=mapping["error_code"],
        message=str(exc),
        request_id=request_id
    )


async def unexpected_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    The Ultimate Safety Net.
    Catches unmapped exceptions (like KeyError, ValueError, RuntimeError).
    Prevents tracebacks from leaking to the client while logging them securely.
    """
    request_id = _get_request_id(request)
    
    logger.exception(f"Unhandled Server Crash | RequestID={request_id} | Path={request.url.path}")

    return _build_error_response(
        status_code=500,
        error_code="INTERNAL_ERROR",
        message="An unexpected application error occurred. The engineering team has been notified.",
        request_id=request_id
    )


def register_exception_handlers(app: FastAPI) -> None:
    """
    Called exactly once in api/main.py to wire the handlers into the FastAPI lifecycle.
    """
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(SelfHealingRAGError, application_exception_handler)
    
    app.add_exception_handler(Exception, unexpected_exception_handler)
import uuid
import pytest
from fastapi.testclient import TestClient

# Import the real, fully assembled FastAPI application
from api.main import app as real_app

from core.container import container
from self_healing.models import RecoveryDecision, RecoveryActionType


@pytest.fixture
def app():
    """
    Provides the real FastAPI application instance.
    No routers or containers are manually registered here because 
    api/main.py already handles the complete assembly.
    """
    return real_app


@pytest.fixture
def api_client(app):
    """
    Provides a reusable HTTP client for API tests.
    
    Using `with TestClient(app) as client:` is critical. It forces FastAPI to 
    execute the `lifespan` context manager defined in `api/main.py`.
    
    Startup: Initializes DB, Container, and LangGraph.
    Yield: Tests run.
    Teardown: Disposes DB connection pool.
    """
    with TestClient(app) as client:
        yield client



@pytest.fixture
def test_session_id() -> str:
    """
    Provides a universally unique session ID for each test.
    This ensures that memory tests do not contaminate each other's 
    conversational history in the PostgreSQL database.
    """
    return f"test-session-{uuid.uuid4().hex}"


@pytest.fixture
def test_user_id() -> str:
    """
    Provides a standard, static user ID.
    Since user identity doesn't require strict isolation in the current 
    architecture, a static identifier is sufficient.
    """
    return "pytest-user"


@pytest.fixture
def force_query_rewrite():
    """
    Forces the Self-Healing policy boundary to select the query rewrite path.

    The HTTP endpoint, FastAPI dependencies, LangGraph, query rewriter,
    retrieval retry, generation, and response formatting remain real.
    """

    original_validator = container.policy_validator

    class ForcedQueryRewriteValidator:
        def validate(self, request):
            return RecoveryDecision(
                requires_recovery=True,
                reason="Forced poor retrieval condition for query rewrite E2E test.",
                suggested_actions=[
                    RecoveryActionType.REWRITE_QUERY,
                    RecoveryActionType.RETRY_RETRIEVAL,
                ],
            )

    container.__dict__["policy_validator"] = ForcedQueryRewriteValidator()

    yield

    container.__dict__["policy_validator"] = original_validator
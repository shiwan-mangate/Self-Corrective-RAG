# core/health.py

import time
import logging
from typing import Dict
from pydantic import BaseModel, Field

from database.connection import DatabaseManager

logger = logging.getLogger(__name__)


class LivenessReport(BaseModel):
    """
    Structured process liveness state.
    """
    status: str = Field(..., description="Will always be 'alive' if the process is running.")


class ReadinessReport(BaseModel):
    """
    Structured application readiness state.
    """
    status: str = Field(..., description="'ready' or 'not_ready'")
    ready: bool = Field(..., description="Boolean flag for safe operational routing.")
    checks: Dict[str, bool] = Field(
        default_factory=dict, 
        description="Granular pass/fail statuses for individual dependencies."
    )
    latency_ms: float = Field(..., description="Time taken to perform all checks.")


class ApplicationHealth:
    """
    Provides runtime liveness and readiness inspection for the application.

    Architecture Rules:
    1. READ-ONLY: Health checks must never mutate application or domain state.
    2. LIGHTWEIGHT: Do not call LLMs, embeddings, retrieval, or LangGraph workflows.
    3. NO EXCEPTION LEAKAGE: Dependency failures are converted into structured
       health reports rather than crashing the API health endpoint.
    4. SEPARATE LIVENESS FROM READINESS:
       - Liveness answers: "Is the application process running?"
       - Readiness answers: "Can the application safely serve requests?"
    """

    def __init__(self, db_manager: DatabaseManager):
        # Strict Dependency Injection
        self.db_manager = db_manager

    def check_liveness(self) -> LivenessReport:
        """
        Returns the lightweight process liveness state.

        This check intentionally avoids all external dependencies.
        If Python can execute this method, the application process is alive.
        """
        return LivenessReport(
            status="alive"
        )

    def check_readiness(self) -> ReadinessReport:
        """
        Validates whether critical runtime infrastructure is available.

        Readiness currently depends on PostgreSQL because request-scoped
        workflows require database-backed memory and persistence services.
        """
        start_time = time.perf_counter()

        # 1. Check Dependencies
        database_healthy = self._check_database()

        # 2. Aggregate Status
        ready = database_healthy

        # 3. Calculate Latency
        latency_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2
        )

        status = "ready" if ready else "not_ready"

        report = ReadinessReport(
            status=status,
            ready=ready,
            checks={
                "database": database_healthy
            },
            latency_ms=latency_ms
        )

        # 4. Telemetry Logging
        if ready:
            logger.debug(
                f"Application readiness check passed | "
                f"Latency={latency_ms}ms"
            )
        else:
            logger.warning(
                f"Application readiness check failed | "
                f"Checks={report.checks} | "
                f"Latency={latency_ms}ms"
            )

        return report

    def _check_database(self) -> bool:
        """
        Performs a lightweight PostgreSQL connectivity check.

        Any failure is converted into False so runtime health inspection
        never leaks infrastructure exceptions into the API layer.
        """
        try:
            return self.db_manager.check_connection()
        except Exception:
            logger.exception(
                "Database health check failed unexpectedly."
            )
            return False
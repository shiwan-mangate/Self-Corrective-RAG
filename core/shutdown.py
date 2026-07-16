# core/shutdown.py

import time
import logging

from database.connection import DatabaseManager

logger = logging.getLogger(__name__)


class ApplicationShutdown:
    """
    Coordinates the controlled teardown of application resources.
    
    Architecture Rules:
    1. BEST-EFFORT: Do not crash the shutdown sequence if a single resource fails. 
       Catch the error, record it, and proceed to the next cleanup step.
    2. REVERSE DEPENDENCY ORDER: (Future) Stop accepting requests -> stop background 
       workers -> dispose databases.
    3. NO BUSINESS LOGIC: Do not trigger final summaries or save conversations.
    4. NO FAKE CLEANUP: Do not manually attempt to destroy stateless pipelines,
       LLM clients, or compiled LangGraph instances. The process termination 
       handles them naturally.
    """

    def __init__(self, db_manager: DatabaseManager):
        # Strict Dependency Injection ensures testability and prevents global lookups
        self.db_manager = db_manager

    def shutdown(self) -> None:
        """
        The main orchestration method for application teardown.
        Called exactly once during the FastAPI lifespan shutdown.
        """
        start_time = time.perf_counter()
        logger.info("Application Shutdown Sequence Started...")
        
        # Accumulate errors to prevent a single failure from blocking the entire cleanup
        errors: list[Exception] = []

        # 1. (Future Implementation) 
        # self._stop_background_workers(errors)
        
        # 2. Dispose database infrastructure
        self._dispose_database(errors)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # 3. Final Observability Telemetry
        if errors:
            logger.error(
                f"Application Shutdown Complete | Status=STOPPED_WITH_ERRORS | "
                f"Errors={len(errors)} | Latency={latency_ms}ms"
            )
        else:
            logger.info(
                f"Application Shutdown Complete | Status=STOPPED | "
                f"Latency={latency_ms}ms"
            )

    def _dispose_database(self, errors: list[Exception]) -> None:
        """
        Safely tells the DatabaseManager to release the SQLAlchemy engine pool.
        """
        logger.info("Disposing PostgreSQL connection pool...")
        try:
            # Delegate to the resource owner's explicit lifecycle contract
            self.db_manager.dispose()
            logger.info("Database infrastructure successfully disposed.")
        except Exception as e:
            # Note: This catches explicit exceptions raised during disposal. 
            # If the network socket hangs indefinitely, the OS or deployment 
            # environment will eventually SIGKILL the process.
            logger.exception("Failed to dispose database infrastructure cleanly.")
            errors.append(e)
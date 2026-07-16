# core/startup.py

import time
import logging

from core.container import ApplicationContainer
from database.connection import DatabaseManager

logger = logging.getLogger(__name__)


class ApplicationStartup:
    """
    Coordinates the one-time application boot and readiness validation sequence.
    
    Architecture Rules:
    1. FAIL FAST: If any critical dependency is missing or broken, the application 
       must crash immediately. Do NOT swallow exceptions.
    2. NO BUSINESS LOGIC: Do not execute vector searches, do not send fake 
       prompts to Groq, and do not mutate domain state.
    3. WARMING: Trigger lazy-loaded singletons so the first user request 
       doesn't suffer a massive latency penalty.
    """

    def __init__(self, container: ApplicationContainer, db_manager: DatabaseManager):
        # Strict Dependency Injection ensures testability
        self.container = container
        self.db_manager = db_manager

    def initialize(self) -> None:
        """
        The main orchestration method for application boot.
        Called exactly once during the FastAPI lifespan startup.
        """
        start_time = time.perf_counter()
        logger.info("Initializing Self-Healing RAG Application...")

        try:
            self._validate_database()
            self._warm_dependencies()
            self._validate_graph_assembly()
            
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.info(f"Application Startup Complete | Status=READY | BootTime={elapsed_ms}ms")
            
        except Exception as e:
            # exc_info=True ensures we print the full traceback for a critical boot crash
            logger.critical("Application Startup FAILED. CRITICAL HALT.", exc_info=True)
            raise RuntimeError("Application failed to start due to critical dependency failure.") from e

    def _validate_database(self) -> None:
        """
        Ensures PostgreSQL (Neon) is initialized and reachable.
        Does NOT execute migrations. Only verifies network connectivity and auth.
        """
        logger.info("Startup Phase 1/3: Validating database connectivity...")
        
        # Explicitly guarantee the Engine and SessionFactory are initialized
        # The database manager is responsible for its own idempotency.
        self.db_manager.initialize()
        
        # Executes a lightweight 'SELECT 1' 
        is_connected = self.db_manager.check_connection()
        
        if not is_connected:
            raise ConnectionError("PostgreSQL database is unreachable. Check DATABASE_URL and network settings.")
            
        logger.info("Database validation successful.")

    def _warm_dependencies(self) -> None:
        """
        Touches the @cached_property methods on the ApplicationContainer.
        This forces Python to instantiate the heavy resources (like LLM clients, 
        embedding models, and stateless pipelines) during boot rather than penalizing 
        the latency of the very first user request.
        """
        logger.info("Startup Phase 2/3: Warming global container dependencies...")
        
        try:
            # 1. Warm Heavy Infrastructure (Using the public interface abstractions)
            _ = self.container.text_generation_service
            _ = self.container.embedding_service
            _ = self.container.judge_service
            
            # 2. Warm ML/RAGAS adapters (these can be notoriously slow to boot)
            _ = self.container.ragas_llm
            _ = self.container.ragas_embeddings
            
            # 3. Warm Stateless Pipelines
            _ = self.container.generation_pipeline
            _ = self.container.evaluation_pipeline
            
            logger.info("Global dependency warming successful.")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize global dependencies: {str(e)}") from e

    def _validate_graph_assembly(self) -> None:
        """
        Performs a dry-run assembly of the LangGraph workflow.
        This verifies that all request-scoped pipelines can be instantiated 
        and the LangGraph node, edge, and conditional routing topology 
        can be compiled successfully.
        """
        logger.info("Startup Phase 3/3: Validating LangGraph compilation...")
        
        # We need a transient, scoped DB session just to satisfy the 
        # constructor requirements of the request-scoped repositories.
        db_session = self.db_manager.SessionLocal()
        
        try:
            # Tell the container to build the workflow. 
            # This triggers GraphBuilder.build() -> builder.compile()
            _ = self.container.create_workflow(db_session)
            
            logger.info("LangGraph compilation successful.")
            
        except Exception as e:
            raise RuntimeError(f"LangGraph failed to compile: {str(e)}") from e
            
        finally:
            # Immediately destroy the test session so it doesn't leak into the connection pool
            db_session.close()
from fastapi import Request, Depends
from sqlalchemy.orm import Session
from database.repositories.evaluation_repository import EvaluationRepository
# Core Configuration & State
from core.container import ApplicationContainer
from core.health import ApplicationHealth
from database.connection import get_db, db_manager
from database.repositories.memory_repository import MemoryRepository
# Domain Pipelines & Workflows
from graph.workflow import GraphWorkflow
from ingestion.pipeline import IngestionPipeline
from evaluation.pipeline import EvaluationPipeline
from memory.pipeline import MemoryPipeline


def get_container(request: Request) -> ApplicationContainer:
    """
    Retrieves the globally initialized ApplicationContainer from the FastAPI app state.
    Fails fast and clearly if the container is missing, preventing cryptic 'NoneType' errors.
    """
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise RuntimeError(
            "ApplicationContainer is not initialized. Ensure 'core.startup' runs in the FastAPI lifespan."
        )
    return container


def get_graph_workflow(
    container: ApplicationContainer = Depends(get_container),
    db_session: Session = Depends(get_db)
) -> GraphWorkflow:
    """
    Retrieves a request-scoped LangGraph workflow.
    Ties the internal repositories and pipelines to the specific HTTP request's DB session.
    """
    return container.create_workflow(db_session)


def get_ingestion_pipeline(
    container: ApplicationContainer = Depends(get_container),
    db_session: Session = Depends(get_db)
) -> IngestionPipeline:
    """
    Retrieves a request-scoped IngestionPipeline for document processing.
    """
    return container.create_ingestion_pipeline(db_session)


def get_evaluation_pipeline(
    container: ApplicationContainer = Depends(get_container)
) -> EvaluationPipeline:
    """
    Retrieves the stateless EvaluationPipeline.
    Since it doesn't require a scoped DB transaction for read/write, we access the cached property directly.
    """
    # Matches exactly to the @cached_property in your core/container.py
    return container.evaluation_pipeline


def get_memory_pipeline(
    container: ApplicationContainer = Depends(get_container),
    db_session: Session = Depends(get_db)
) -> MemoryPipeline:
    """
    Retrieves a request-scoped MemoryPipeline.
    """
    if not hasattr(container, "create_memory_pipeline"):
        raise NotImplementedError(
            "ApplicationContainer is missing 'create_memory_pipeline'. "
            "Extract the MemoryPipeline wiring out of 'create_workflow' into its own factory method."
        )
    return container.create_memory_pipeline(db_session)


def get_application_health() -> ApplicationHealth:
    """
    Retrieves the Core Health Service.
    Does not require the global container since it binds directly to the singleton db_manager.
    """
    # Matches your core/health.py ApplicationHealth(db_manager) signature perfectly
    return ApplicationHealth(db_manager)



def get_evaluation_repository(
    db_session: Session = Depends(get_db)
) -> EvaluationRepository:
    """
    Retrieves a request-scoped EvaluationRepository.
    """
    return EvaluationRepository(db_session)

def get_memory_repository(
    db_session: Session = Depends(get_db)
) -> MemoryRepository:
    """
    Retrieves a request-scoped MemoryRepository.
    Safe for transient HTTP READ operations of chat histories.
    """
    return MemoryRepository(db_session)
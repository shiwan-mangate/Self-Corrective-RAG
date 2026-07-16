import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ==========================================
# Core Infrastructure & Lifecycle
# ==========================================
from core.container import container
from database.connection import db_manager
from core.startup import ApplicationStartup
from core.shutdown import ApplicationShutdown

# ==========================================
# Middleware & Exceptions
# ==========================================
from api.middleware.request_id import RequestIDMiddleware
from api.middleware.logging import RequestLoggingMiddleware
from api.middleware.exception_handler import register_exception_handlers

# ==========================================
# Routers
# ==========================================
from api.routers import health
from api.routers import documents
from api.routers import chat
from api.routers import evaluation
from api.routers import memory

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """
    Manages the global application lifecycle.
    """
   
    logger.info("Initiating Application Lifespan Startup...")
    startup_coordinator = ApplicationStartup(container, db_manager)
    startup_coordinator.initialize()
    
    fastapi_app.state.container = container
    
    logger.info("Application is ready to receive traffic.")
    
    yield  

  
    logger.info("Initiating Application Lifespan Shutdown...")
    shutdown_coordinator = ApplicationShutdown(db_manager)
    shutdown_coordinator.shutdown()
    logger.info("Application safely terminated.")


def create_app() -> FastAPI:
    """
    Factory function to assemble the FastAPI application.
    """
    app = FastAPI(
        title="Self-Healing RAG Platform",
        description="Enterprise-grade RAG system with autonomous hallucination detection and self-healing.",
        version="1.0.0",
        lifespan=lifespan,
        servers=[
            {
                "url": "https://self-corrective-rag-ciwm.onrender.com",
                "description": "Production Server"
            }
        ]
    )

  
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)

   
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(chat.router)
    app.include_router(evaluation.router)
    app.include_router(memory.router)

    return app


app = create_app()
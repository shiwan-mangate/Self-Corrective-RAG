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
async def lifespan(app: FastAPI):
    """
    Manages the global application lifecycle.
    Bootstraps all heavy singletons before accepting HTTP traffic and 
    cleans up database connection pools during shutdown.
    """
    # 1. Start Up
    logger.info("Initiating Application Lifespan Startup...")
    startup_coordinator = ApplicationStartup(container, db_manager)
    startup_coordinator.initialize()
    
    app.state.container = container
    
    logger.info("Application is ready to receive traffic.")
    
    yield  # Yield control back to FastAPI to run the application

    # 2. Shut Down
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
        # ---------------------------------------------------------
        # THIS IS THE MAGIC FIX: Force Swagger UI to use HTTPS
        # ---------------------------------------------------------
        servers=[
            {
                "url": "https://self-corrective-rag-ciwm.onrender.com",
                "description": "Production Server"
            }
        ]
    )

    # ---------------------------------------------------------
    # 1. Register CORS Middleware (Must be first!)
    # ---------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------------------------------------------------------
    # 2. Register Custom Middleware
    # ---------------------------------------------------------
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # ---------------------------------------------------------
    # 3. Register Exception Handlers
    # ---------------------------------------------------------
    register_exception_handlers(app)

    # ---------------------------------------------------------
    # 4. Include Routers
    # ---------------------------------------------------------
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(chat.router)
    app.include_router(evaluation.router)
    app.include_router(memory.router)

    return app


# Create the global FastAPI application instance that Uvicorn will serve
app = create_app()
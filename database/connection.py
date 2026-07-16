import logging
from urllib.parse import urlparse
from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session


from config.settings import settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manages the SQLAlchemy engine and session lifecycle.
    Uses lazy initialization to ensure it plays nicely with testing environments 
    and application startup/shutdown events.
    """
    def __init__(self):
        self.engine = None
        self.SessionLocal = None

    def initialize(self) -> None:
        """Creates the engine and session factory. Safe to call multiple times."""
        if self.engine is not None:
            return

        database_url = settings.NEON_VECTOR_DATABASE_URL
        if not database_url:
            raise ValueError("NEON_DATABASE_URL is missing from configuration.")

        
        parsed_url = urlparse(database_url)
        logger.info(
            f"Initializing Database Engine | Host: {parsed_url.hostname} | "
            f"DB: {parsed_url.path.lstrip('/')} | Pool Size: 5"
        )

        try:
            self.engine = create_engine(
                database_url,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=1800,
                pool_pre_ping=True,     
            )

            self.SessionLocal = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False   
            )
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")
            raise

    def get_session(self) -> Generator[Session, None, None]:
        """Creates a new database session and ensures it is closed safely."""
        if self.SessionLocal is None:
            self.initialize()
            
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def check_connection(self) -> bool:
        """
        Executes a lightweight query to verify database health.
        Designed for use in core/health.py or Kubernetes readiness probes.
        """
        if self.engine is None:
            self.initialize()
            
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
        
    def dispose(self) -> None:
        """
        Gracefully closes the connection pool and resets database infrastructure state.
        Safe to call even if the database was never initialized.
        """
        try:
            if self.engine is not None:
                self.engine.dispose()
        finally:
            self.engine = None
            self.SessionLocal = None


db_manager = DatabaseManager()

def get_db() -> Generator[Session, None, None]:
    """FastAPI/Dependency-Injection friendly wrapper."""
    yield from db_manager.get_session()
import logging
from abc import ABC, abstractmethod
from typing import Optional
from sqlalchemy.orm import Session as DBSession
from sqlalchemy.exc import SQLAlchemyError

from memory.models import Session
from database.models.session import SessionModel

logger = logging.getLogger(__name__)


class BaseSessionStorage(ABC):
    """
    Abstract contract for persisting Session lifecycles.
    Ensures the Memory subsystem relies on abstractions, not databases.
    """
    
    @abstractmethod
    def load(self, session_id: str) -> Optional[Session]:
        pass
        
    @abstractmethod
    def save(self, session: Session) -> Session:
        pass


class PostgresSessionStorage(BaseSessionStorage):
    """
    SQLAlchemy implementation of session persistence.
    Acts as the Anti-Corruption Layer translating SQL rows into pure Pydantic Domain models.
    """
    
    def __init__(self, db: DBSession):
        self.db = db

    def load(self, session_id: str) -> Optional[Session]:
        """
        Fetches the active connection state for a user.
        Strictly filters for active=True to ignore expired/archived sessions.
        """
        try:
            record = self.db.query(SessionModel).filter_by(
                session_id=session_id,
                active=True
            ).first()
            
            if not record:
                return None
                
          
            return Session(
                session_id=record.session_id,
                conversation_id=None,  
                created_at=record.created_at,
                last_activity=record.last_activity,
                message_count=record.message_count,
                active=record.active
            )
            
        except SQLAlchemyError as e:
            logger.error(f"Database error loading session {session_id}: {str(e)}")
            raise

    def save(self, session: Session) -> Session:
        """
        Upserts the session state into the database.
        Splits internal logic into create vs. update, commits the single-table transaction, 
        and refreshes server-side defaults (like onupdate=func.now()).
        """
        try:
            record = self.db.query(SessionModel).filter_by(session_id=session.session_id).first()
            
            if not record:
                record = SessionModel(
                    session_id=session.session_id,
                    user_id=getattr(session, 'user_id', None),
                    created_at=session.created_at,
                    last_activity=session.last_activity,
                    message_count=session.message_count,
                    active=session.active
                )
                self.db.add(record)
                logger.debug(f"Creating new database record for Session {session.session_id}")
            else:
                
                record.last_activity = session.last_activity
                record.message_count = session.message_count
                record.active = session.active
                logger.debug(f"Updating existing database record for Session {session.session_id}")
                
            self.db.commit()
            self.db.refresh(record)
            
            return Session(
                session_id=record.session_id,
                created_at=record.created_at,
                last_activity=record.last_activity,
                message_count=record.message_count,
                active=record.active
            )
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error saving session {session.session_id}: {str(e)}")
            raise
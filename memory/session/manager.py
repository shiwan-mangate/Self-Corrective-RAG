# memory/session/manager.py

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from memory.models import Session
from memory.session.storage import BaseSessionStorage
from memory.constants import SESSION_TIMEOUT_HOURS, AUTO_CREATE_SESSION
from shared.exceptions import SessionNotFoundError 

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages the lifecycle, timeouts, and state transitions of chat sessions.
    Strictly coordinates policies and delegates database ops to the Storage layer.
    """

    def __init__(self, storage: BaseSessionStorage):
        self.storage = storage

    def load_or_create(
        self, 
        session_id: str, 
        user_id: Optional[str] = None, 
        query_id: Optional[str] = None
    ) -> Session:
        """
        Public API: Retrieves an active session or mints a new one.
        """
        session = self.storage.load(session_id)

        if session and self._is_expired(session):
            logger.info(
                f"Session {session_id} expired due to inactivity "
                f"| QueryID={query_id}"
            )
            self._expire_session(session)
            session = None  

        if not session:
            if not AUTO_CREATE_SESSION:
                logger.error(f"Session {session_id} not found and AUTO_CREATE_SESSION is disabled.")
                raise SessionNotFoundError(f"Active session '{session_id}' is required but missing.")
                
            session = self._create_session(session_id, user_id, query_id)

        return session

    def update_activity(
        self, 
        session_id: str, 
        message_delta: int, 
        query_id: Optional[str] = None
    ) -> Session:
        """
        Public API: Bumps timestamps and total interaction telemetry.
        """
        session = self.storage.load(session_id)
        
        if not session:
            logger.warning(
                f"Attempted to update missing session {session_id}. Re-creating safely. "
                f"| QueryID={query_id}"
            )
            session = self._create_session(session_id=session_id, user_id=None, query_id=query_id)

        session.last_activity = datetime.now(timezone.utc)
        session.message_count += message_delta
        
        return self.storage.save(session)

    def touch(self, session_id: str, query_id: Optional[str] = None) -> Session:
        """
        Public API: Lightweight wrapper to bump the last_activity timestamp without adding messages.
        """
        return self.update_activity(session_id=session_id, message_delta=0, query_id=query_id)

    def close_session(self, session_id: str, query_id: Optional[str] = None) -> None:
        """
        Public API: Manually archives a session (e.g., user logged out).
        """
        session = self.storage.load(session_id)
        if session and session.active:
            self._expire_session(session)
            logger.info(f"Session {session_id} manually closed | QueryID={query_id}")



    def _is_expired(self, session: Session) -> bool:
        """Policy check: Has the session exceeded the max idle time?"""
        now = datetime.now(timezone.utc)
        expiration_time = session.last_activity + timedelta(hours=SESSION_TIMEOUT_HOURS)
        return now > expiration_time

    def _create_session(
        self, 
        session_id: str, 
        user_id: Optional[str], 
        query_id: Optional[str]
    ) -> Session:
        """Factory method to ensure all new sessions are minted with identical structures."""
        now = datetime.now(timezone.utc)
        logger.debug(f"Creating new session lifecycle for {session_id} | QueryID={query_id}")
        
        new_session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_activity=now,
            message_count=0,
            active=True
        )
        return self.storage.save(new_session)

    def _expire_session(self, session: Session) -> None:
        """Helper to mark a session as inactive while keeping timestamps accurate."""
        session.active = False
        session.last_activity = datetime.now(timezone.utc)
        self.storage.save(session)
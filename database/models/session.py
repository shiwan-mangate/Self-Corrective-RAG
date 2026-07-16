# database/models/session.py

from sqlalchemy import Column, String, Integer, Boolean, DateTime, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from database.models.base import Base


class SessionModel(Base):
    """
    SQLAlchemy ORM model mapping to the `chat_sessions` table.
    Tracks the lifecycle and telemetry of an active chat session.

    Architecture Note:
    A session owns many conversation messages and at most one active
    conversation summary. Conversation content is persisted separately.
    """
    __tablename__ = "chat_sessions"

    # Identity
    session_id = Column(
        String(100), 
        primary_key=True,
        comment="Unique identifier for the active connection/session."
    )
    user_id = Column(
        String(100), 
        nullable=True, 
        index=True,
        comment="Optional identifier to link sessions to a specific user profile."
    )
    
    # Lifecycle
    created_at = Column(
        DateTime(timezone=True), 
        nullable=False, 
        server_default=func.now()
    )
    last_activity = Column(
        DateTime(timezone=True), 
        nullable=False, 
        server_default=func.now(), 
        onupdate=func.now(),
        comment="Automatically updates via SQL whenever the row is modified."
    )
    
    # Telemetry & State
    message_count = Column(
        Integer, 
        nullable=False, 
        default=0,
        comment="Telemetry metric tracking the volume of activity in this session."
    )
    active = Column(
        Boolean, 
        nullable=False, 
        default=True,
        comment="Lifecycle flag. False indicates the session is expired or archived."
    )
    
    # Extensibility
    session_metadata = Column(
        "metadata", # Mapped safely to 'metadata' in postgres
        JSONB, 
        nullable=False, 
        server_default='{}',
        comment="Flexible storage for browser info, client version, locale, etc."
    )

    __table_args__ = (
        Index("ix_chat_sessions_last_activity", "last_activity"),
        Index("ix_chat_sessions_active", "active"),
    )

    def __repr__(self):
        return f"<SessionModel(session_id='{self.session_id}', user_id='{self.user_id}', active={self.active})>"
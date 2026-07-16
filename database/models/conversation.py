# database/models/conversation.py

from sqlalchemy import Column, String, Text, Integer, DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from database.models.base import Base


class ConversationMessageModel(Base):
    """
    SQLAlchemy model for individual conversation messages.
    A chat session may contain many ordered messages.
    """
    __tablename__ = "conversation_messages"

    message_id = Column(
        String(100),
        primary_key=True
    )

    session_id = Column(
        String(100),
        ForeignKey(
            "chat_sessions.session_id",
            ondelete="CASCADE"
        ),
        nullable=False,
        index=True
    )

    role = Column(
        String(20),
        nullable=False
    )

    content = Column(
        Text,
        nullable=False,
        comment="Conversation message content."
    )

    tokens = Column(
        Integer,
        nullable=False,
        default=0
    )

    message_metadata = Column(
        "metadata",
        JSONB,
        server_default='{}',
        nullable=False
    )

    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now()
    )

    __table_args__ = (
        Index(
            "ix_conversation_messages_session_time",
            "session_id",
            "timestamp"
        ),
    )

    def __repr__(self):
        return (
            f"<MessageModel("
            f"id='{self.message_id}', "
            f"role='{self.role}', "
            f"session='{self.session_id}'"
            f")>"
        )


class ConversationSummaryModel(Base):
    """
    SQLAlchemy model for the active compressed conversation summary.

    Exactly one active summary is maintained per session.
    """
    __tablename__ = "conversation_summaries"

    session_id = Column(
        String(100),
        ForeignKey(
            "chat_sessions.session_id",
            ondelete="CASCADE"
        ),
        primary_key=True
    )

    summary_text = Column(
        Text,
        nullable=False
    )

    # FIXED: Renamed from covered_message_count to match Memory domain contract
    covered_messages = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Running total of messages compressed into the active summary."
    )

    summary_version = Column(
        Integer,
        nullable=False,
        default=1
    )

    model_name = Column(
        String(100),
        nullable=False
    )

    # Kept for future distributed tracing capabilities
    last_query_id = Column(
        String(100),
        nullable=True,
        comment="Optional distributed trace identifier for summary generation."
    )

    generated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now()
    )

    def __repr__(self):
        return (
            f"<SummaryModel("
            f"session='{self.session_id}', "
            f"version={self.summary_version}"
            f")>"
        )
## database/repositories/memory_repository.py
from typing import Optional, List
from sqlalchemy.orm import Session

from database.models.session import SessionModel
from database.models.conversation import ConversationMessageModel


class MemoryRepository:
    """
    The exclusive Read-Side gateway for the Conversation Memory.
    Retrieves persisted chat history without triggering the LangGraph state machine.
    """
    def __init__(self, session: Session):
        self.session = session

    def get_session(self, session_id: str) -> Optional[SessionModel]:
        """Fetches the metadata and lifecycle state for a given session."""
        return (
            self.session.query(SessionModel)
            .filter(SessionModel.session_id == session_id)
            .first()
        )

    def get_messages(self, session_id: str) -> List[ConversationMessageModel]:
        """
        Fetches all messages for a session.
        Orders by timestamp ASCENDING, with a fallback message_id sort to 
        guarantee deterministic ordering under heavy concurrent loads.
        """
        return (
            self.session.query(ConversationMessageModel)
            .filter(ConversationMessageModel.session_id == session_id)
            .order_by(
                ConversationMessageModel.timestamp.asc(),
                ConversationMessageModel.message_id.asc()
            )
            .all()
        )
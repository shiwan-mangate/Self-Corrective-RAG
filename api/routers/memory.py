# api/routers/memory.py
import logging
from typing import List
from fastapi import APIRouter, Depends

# Public Schemas
from schemas.memory import (
    ConversationHistoryResponse,
    SessionResponse,
    ConversationMessageResponse,
    MessageRole
)


from api.dependencies import get_memory_repository
from database.repositories.memory_repository import MemoryRepository


from shared.exceptions import SessionNotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/memory",
    tags=["Memory"]
)


@router.get("/{session_id}", response_model=ConversationHistoryResponse)
def get_session_history(
    session_id: str,
    repository: MemoryRepository = Depends(get_memory_repository)
) -> ConversationHistoryResponse:
    """
    Retrieves the complete, chronological public conversation history for a session.
    Automatically filters out internal system and tool messages.
    """
    logger.info(f"Conversation History requested | SessionID={session_id}")
    
   
    session = repository.get_session(session_id)
    
    if not session:
        logger.warning(f"Memory read failed: Session '{session_id}' not found.")
        raise SessionNotFoundError(f"The requested session '{session_id}' does not exist.")

   
    raw_messages = repository.get_messages(session_id)
    
   
    session_response = SessionResponse(
        session_id=session.session_id,
        created_at=session.created_at,
        last_activity=session.last_activity,
        message_count=session.message_count,
        active=session.active
    )
    
   
    message_responses: List[ConversationMessageResponse] = []
    
    for msg in raw_messages:
       
        role = str(msg.role).lower()
        
        
        if role in {"user", "assistant"}:
            message_responses.append(
                ConversationMessageResponse(
                    message_id=msg.message_id,
                    role=MessageRole(role),
                    content=msg.content,
                    timestamp=msg.timestamp
                )
            )

    logger.info(
        f"Memory payload mapped | SessionID={session_id} | "
        f"Public Messages={len(message_responses)} (Raw={len(raw_messages)})"
    )

   
    return ConversationHistoryResponse(
        session=session_response,
        messages=message_responses
    )
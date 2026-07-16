# schemas/memory.py

from datetime import datetime
from enum import Enum
from typing import List
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """
    Public enumeration of message roles.
    Restricts API output to user-facing roles only, shielding internal system prompts.
    """
    USER = "user"
    ASSISTANT = "assistant"


class SessionResponse(BaseModel):
    """
    Public representation of a chat session's metadata.
    """
    session_id: str = Field(
        ..., 
        description="The unique identifier for the conversation session."
    )
    created_at: datetime = Field(
        ..., 
        description="The exact UTC timestamp when the session was originally created."
    )
    last_activity: datetime = Field(
        ..., 
        description="The UTC timestamp of the most recent interaction in this session."
    )
    message_count: int = Field(
        ..., 
        ge=0,
        description="The total number of messages exchanged in this session across its entire lifespan."
    )
    active: bool = Field(
        ...,
        description="Whether the conversation session is currently active."
    )


class ConversationMessageResponse(BaseModel):
    """
    Public representation of a single conversation turn.
    Excludes internal metadata, token counts, and evaluator annotations.
    """
    message_id: str = Field(
        ..., 
        description="The unique identifier for the message."
    )
    role: MessageRole = Field(
        ..., 
        description="The actor who generated the message ('user' or 'assistant')."
    )
    content: str = Field(
        ..., 
        min_length=1,
        description="The text content of the message."
    )
    timestamp: datetime = Field(
        ..., 
        description="When the message was recorded."
    )


class ConversationHistoryResponse(BaseModel):
    """
    Public representation of a complete conversation thread.
    Combines session metadata with the ordered list of messages.
    """
    session: SessionResponse = Field(
        ..., 
        description="The metadata and lifecycle state of the session."
    )
    messages: List[ConversationMessageResponse] = Field(
        default_factory=list,
        description="The chronological list of user and assistant messages."
    )
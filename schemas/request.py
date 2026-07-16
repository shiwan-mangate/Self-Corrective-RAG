# schemas/request.py

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """
    The top-level external contract for interacting with the Self-Healing RAG API.
    Acts as the security and validation boundary before data enters the LangGraph orchestrator.
    """
    
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=10000,
        description="The user's input question or prompt.",
        json_schema_extra={"example": "How does the Self-Healing RAG system recover from hallucinations?"}
    )
    

    session_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="A unique identifier for the conversation thread. Used to retrieve chat history.",
        json_schema_extra={"example": "session-abc-12345"}
    )
    
    user_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Optional identifier for the user. Useful for analytics and long-term preferences.",
        json_schema_extra={"example": "user-9981"}
    )
    
    query_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Optional trace ID. If not provided, the graph's Input Preparation Node will generate one.",
        json_schema_extra={"example": "req-trace-5542"}
    )

    @field_validator('query')
    @classmethod
    def validate_query_not_empty(cls, v: str) -> str:
        """
        Ensures the query isn't just empty spaces, tabs, or newlines.
        """
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query cannot be empty or solely whitespace.")
        return cleaned
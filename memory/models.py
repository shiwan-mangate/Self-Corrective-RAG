import uuid
from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"




class ConversationMessage(BaseModel):
    """A single, immutable turn in the conversation."""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole = Field(...)
    content: str = Field(...)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tokens: int = Field(default=0)
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Flexible storage for citations, UI telemetry, tool calls, or multi-modal links."
    )


class ConversationHistory(BaseModel):
    """The localized literal history, isolated from how it's stored."""
    messages: List[ConversationMessage] = Field(default_factory=list)
    total_messages: int = Field(default=0)
    total_tokens: int = Field(default=0)


class ConversationSummary(BaseModel):
    """The compressed knowledge representation of older turns."""
    summary: str = Field(..., description="The LLM-generated narrative of past context.")
    covered_messages: int = Field(..., description="How many past messages were compressed into this summary.")
    summary_version: int = Field(default=1, description="Tracks revisions to the summary for debugging and rollback.")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_name: str = Field(..., description="The model used to generate this summary (e.g., llama3-8b).")



class Session(BaseModel):
    """The lifecycle state of the chat itself."""
    session_id: str = Field(...)
    conversation_id: Optional[str] = Field(default=None, description="Supports future branching/threading.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    message_count: int = Field(default=0)
    active: bool = Field(default=True)

class SummaryTrigger(str, Enum):
    """Strict enumeration of all possible policy evaluation outcomes."""
    FORCE_REFRESH = "force_refresh"
    TOKEN_THRESHOLD = "token_threshold"
    MESSAGE_THRESHOLD = "message_threshold"
    WITHIN_LIMITS = "within_limits"
    INSUFFICIENT_MESSAGES = "insufficient_messages"

class SummaryDecision(BaseModel):
    """
    Structured telemetry object resulting from the Summary Policy evaluation.
    """
    should_summarize: bool = Field(..., description="The final boolean decision.")
    reason: str = Field(..., description="Human-readable explanation of the decision.")
    triggered_by: SummaryTrigger = Field(..., description="Strict Enum indicating the exact trigger.")


class MemoryRequest(BaseModel):
    """
    The strict input boundary for building context.
    Contains ONLY what the Orchestrator knows at the start of a turn.
    """
    query_id: str = Field(..., description="Unique trace ID for distributed logging.")
    session_id: str = Field(...)
    current_query: str = Field(..., description="The raw question asked by the user.")
    user_id: Optional[str] = Field(default=None)
    conversation_id: Optional[str] = Field(default=None)
    force_refresh_summary: bool = Field(
        default=False, 
        description="Bypasses token thresholds to force a regeneration of the summary."
    )


class SaveConversationRequest(BaseModel):
    """
    The strict input boundary for persisting a completed turn.
    """
    query_id: str = Field(...)
    session_id: str = Field(...)
    user_message: ConversationMessage = Field(...)
    assistant_message: ConversationMessage = Field(...)


class SummaryGenerationRequest(BaseModel):
    """Internal subsystem request sent to the Summarizer."""
    query_id: str = Field(...)
    session_id: str = Field(...)
    messages_to_summarize: List[ConversationMessage] = Field(...)
    previous_summary: Optional[ConversationSummary] = Field(default=None)
    force_refresh: bool = Field(default=False)



class SummaryGenerationResult(BaseModel):
    """The complete result payload from the Summarizer."""
    success: bool = Field(...)
    summary: Optional[ConversationSummary] = Field(default=None)
    latency_ms: float = Field(...)
    total_tokens: int = Field(...)
    model_name: str = Field(...)
    error_message: Optional[str] = Field(default=None)




class MemoryMetadata(BaseModel):
    """Observability and execution telemetry for the Memory subsystem."""
    latency_ms: float = Field(...)
    messages_loaded: int = Field(...)
    history_messages_used: int = Field(..., description="How many raw messages were actively included in the context.")
    summary_used: bool = Field(...)
    summary_generated: bool = Field(..., description="True if a new summary was triggered during this request.")
    token_budget: int = Field(...)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))





# --- ADD THESE MISSING MODELS ---
class PruningPlan(BaseModel):
    message_ids_to_remove: set[str] = Field(default_factory=set)
    remaining_history: ConversationHistory = Field(...)

class ConversationSaveResult(BaseModel):
    session_id: str
    query_id: Optional[str]
    messages_saved: int
    user_message_id: str
    assistant_message_id: str

class Verbosity(str, Enum):
    CONCISE = "concise"
    BALANCED = "balanced"
    DETAILED = "detailed"

class UserProfile(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user.")
    timezone: Optional[str] = Field(default="UTC")
    locale: Optional[str] = Field(default="en-US")
    display_name: Optional[str] = Field(default=None)

class UserPreferences(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user.")
    preferred_language: str = Field(default="English")
    verbosity: Verbosity = Field(default=Verbosity.BALANCED)
    coding_language_preference: Optional[str] = Field(default=None)
    custom_instructions: List[str] = Field(default_factory=list)

# --- REPLACE THE DUPLICATE MemoryContext CLASSES WITH THIS ONE ---
class MemoryContext(BaseModel):
    """
    The unified contextual package. 
    This is what Retrieval and Generation will actually consume.
    """
    session: Session = Field(...)
    active_history: ConversationHistory = Field(..., description="Only the recent, unsummarized window.")
    summary: Optional[ConversationSummary] = Field(default=None)
    
    # FIXED: Strict Pydantic typing to prevent the dictionary crash in the formatter
    user_preferences: Optional[UserPreferences] = Field(default=None) 
    
    formatted_context_string: str = Field(
        ..., 
        description="The finalized, read-only string ready for LLM prompt injection."
    )

class MemoryResponse(BaseModel):
    """
    The final, public output of the Memory Subsystem's build_context operation.
    Perfectly mirrors GenerationResponse and EvaluationReport.
    """
    context: MemoryContext = Field(...)
    metadata: MemoryMetadata = Field(...)
from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from evaluation.models import EvaluationReport
from retrieval.models import RankedChunk,RetrievalMetadata




class RecoveryActionType(str, Enum):
    """Strictly defined actions to prevent string-matching bugs."""
    REWRITE_QUERY = "rewrite_query"
    RETRY_RETRIEVAL = "retry_retrieval"
    WEB_SEARCH = "web_search"
    MERGE_CONTEXT = "merge_context"
    STRICT_GROUNDING = "strict_grounding"
    ASK_CLARIFICATION = "ask_clarification"
    LOG_KNOWLEDGE_GAP = "log_knowledge_gap"
    STOP = "stop"


class RecoveryStatus(str, Enum):
    """The execution lifecycle of a RecoveryPlan."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    ABORTED = "aborted"
    SKIPPED = "skipped"


class FeedbackRating(str, Enum):
    """Standardized user feedback signals."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"




class RetryState(BaseModel):
    """
    Tracks the history of a request to prevent infinite loops and duplicate work.
    """
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=3)
    visited_queries: List[str] = Field(default_factory=list)
    visited_context_hashes: List[str] = Field(default_factory=list)
    visited_action_sequences: List[str] = Field(
        default_factory=list, 
        description="Tracks combinations like 'rewrite->retrieve' to prevent executing the exact same recovery loop twice."
    )
    web_search_used: bool = Field(default=False)
    last_failure_reason: Optional[str] = Field(default=None)




class SelfHealingRequest(BaseModel):
    """
    The strict input boundary for the Self-Healing Subsystem.
    Maintains perfect architectural symmetry with all other subsystems.
    """
    query_id: str = Field(...)
    evaluation_report: EvaluationReport = Field(...)
    retry_state: RetryState = Field(default_factory=RetryState)
    retrieval_metadata: RetrievalMetadata



class RecoveryContext(BaseModel):
    """
    The intermediate state object passed between recovery actors during execution.
    Progressively enriched as the pipeline runs.
    """
    original_query: str = Field(...)
    rewritten_query: Optional[str] = Field(default=None)
    internal_chunks: List[RankedChunk] = Field(default_factory=list)
    web_chunks: List[RankedChunk] = Field(default_factory=list)
    merged_context: Optional[str] = Field(default=None, description="The final text to pass back to Generation.")




class RecoveryDecision(BaseModel):
    """
    The 'Diagnosis' from the Validator. 
    """
    requires_recovery: bool = Field(...)
    reason: str = Field(...)
    suggested_actions: List[RecoveryActionType] = Field(default_factory=list)


class RecoveryAction(BaseModel):
    """
    A specific, executable 'Treatment' step that tracks its own execution status.
    """
    action_type: RecoveryActionType = Field(...)
    execution_order: int = Field(...)
    description: str = Field(...)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Execution Tracking
    completed: bool = Field(default=False)
    success: bool = Field(default=False)
    error_message: Optional[str] = Field(default=None)




class KnowledgeGap(BaseModel):
    """
    Represents a discovered weakness in the system's internal knowledge base.
    Used by the long-term learning loop to trigger automated ingestion.
    """
    missing_topic: str = Field(..., description="The core subject that was missing (e.g., 'Model Context Protocol').")
    failed_queries: List[str] = Field(default_factory=list, description="The specific user questions that triggered this gap.")
    frequency: int = Field(default=1, description="How many times users have asked about this topic.")
    
    # Lifecycle
    resolved: bool = Field(default=False, description="True if the system successfully ingested documents for this topic.")
    
    # Telemetry & Tracing
    first_detected: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_detected: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_query_id: Optional[str] = Field(default=None, description="The specific trace ID of the last request that hit this gap.")
    resolved_at: Optional[datetime] = Field(default=None, description="When the ingestion pipeline successfully resolved this gap.")


class FeedbackEvent(BaseModel):
    """User-driven system correction."""
    query_id: str = Field(...)
    rating: FeedbackRating = Field(...)
    reason: Optional[str] = Field(default=None)
    recovery_used: bool = Field(default=False)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))




class RecoveryMetadata(BaseModel):
    """Observability for the Self-Healing layer."""
    latency_ms: float = Field(...)
    total_recovery_time_ms: float = Field(default=0.0, description="Cumulative time across all retries.")
    actions_executed: int = Field(...)
    web_search_triggered: bool = Field(default=False)
    knowledge_gap_logged: bool = Field(default=False)
    pipeline_version: str = Field(default="1.0")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RecoveryPlan(BaseModel):
    """
    The final, public output of the Self-Healing Subsystem.
    """
    query_id: str = Field(...)
    status: RecoveryStatus = Field(default=RecoveryStatus.PENDING)
    continue_pipeline: bool = Field(
        ..., 
        description="Explicit flag telling the Master Orchestrator whether to loop again or abort/return."
    )
    
    decision: RecoveryDecision = Field(...)
    actions: List[RecoveryAction] = Field(default_factory=list)
    recovery_context: RecoveryContext = Field(..., description="The enriched context ready for the next generation attempt.")
    retry_state: RetryState = Field(...)
    
    knowledge_gap: Optional[KnowledgeGap] = Field(default=None)
    metadata: RecoveryMetadata = Field(...)
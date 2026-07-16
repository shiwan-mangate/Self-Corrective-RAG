# graph/models.py

from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class ExecutionStrategy(str, Enum):
    """
    The pure routing strategies understood by the Graph Router.
    This separates the 'What to do next' (Orchestration) from 
    the 'How to fix it' (Self-Healing RecoveryActions).
    """
    RETURN_RESPONSE = "return_response"
    RESTART_RETRIEVAL = "restart_retrieval"
    RESTART_GENERATION = "restart_generation"
    TERMINATE = "terminate"


class NodeType(str, Enum):
    """
    Strict enumeration of all graph nodes. 
    Prevents string-matching bugs and enables type-safe routing.
    """
    INPUT_PREPARATION = "input_preparation"
    MEMORY = "memory"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    EVALUATION = "evaluation"
    SELF_HEALING = "self_healing"
    RETRY_GUARD = "retry_guard"
    RESPONSE = "response"
    PERSIST = "persist"


class NodeStatus(str, Enum):
    """
    The strict lifecycle states of a single node's execution.
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"

class ResponseStatus(str, Enum):
    """
    The strict contract for the public API response status.
    Allows the frontend to handle clarifications differently than final answers.
    """
    SUCCESS = "success"
    PARTIAL = "partial"  
    FAILED = "failed"


class NodeExecution(BaseModel):
    """
    Telemetry for a single node's execution within the graph.
    Strictly orchestration metadata, containing no business data.
    """
    node_type: NodeType = Field(..., description="The strict enum of the executed node.")
    status: NodeStatus = Field(default=NodeStatus.PENDING)
    attempt: int = Field(default=1, description="Which iteration of the recovery loop this execution belongs to.")
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = Field(default=None)
    latency_ms: float = Field(default=0.0)
    error_message: Optional[str] = Field(default=None)
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Flexible storage for future tracing (e.g., token usage, model versions)."
    )


class ExecutionTrace(BaseModel):
    """
    The ordered chronological history of the workflow.
    Crucial for debugging cyclic loops and understanding the exact path taken.
    """
    executions: List[NodeExecution] = Field(default_factory=list)
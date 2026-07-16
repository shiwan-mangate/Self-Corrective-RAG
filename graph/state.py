# graph/state.py

import uuid
from typing import Optional, List, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from schemas.request import ChatRequest
# Graph Orchestration Models
from graph.models import ExecutionTrace, ResponseStatus

# Subsystem Contracts (The immutable domain boundaries)
from memory.models import MemoryResponse
from retrieval.models import RetrievalContext
from generation.models import GenerationResponse
from evaluation.models import EvaluationReport
from self_healing.models import RecoveryPlan
from schemas.response import ChatResponse

from typing import List, Optional, Any
from pydantic import BaseModel, Field
from graph.models import ResponseStatus

from graph.models import ExecutionTrace, ResponseStatus, ExecutionStrategy





# ==========================================
# 2. Graph-Specific Runtime State Models
# ==========================================
class ExecutionState(BaseModel):
    """
    Pure orchestration state. 
    Changes every request and belongs ONLY to LangGraph.
    """
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    current_node: str = Field(default="START")
    
    # Retry Guard State
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=3)
    retry_allowed: bool = Field(default=True)
    termination_reason: Optional[str] = Field(default=None)

    visited_nodes: List[str] = Field(default_factory=list)
    execution_strategy: ExecutionStrategy = Field(default=ExecutionStrategy.RETURN_RESPONSE)
    
    # Telemetry
    total_latency_ms: float = Field(default=0.0)
    execution_trace: ExecutionTrace = Field(default_factory=ExecutionTrace)


class GraphError(BaseModel):
    """
    Catches catastrophic pipeline failures before they crash the orchestrator.
    """
    node_name: str = Field(..., description="The node where the failure occurred.")
    error_type: str = Field(..., description="The class name of the exception.")
    message: str = Field(..., description="The string representation of the exception.")
    traceback: Optional[str] = Field(default=None, description="The stack trace if available.")


class ResponseState(BaseModel):
    """Internal graph state for the final assembled response."""
    status: ResponseStatus
    answer: str
    citations: List[Any] = Field(default_factory=list) # Accepts internal GenerationCitations
    confidence: float = 0.0
    recovery_used: bool = False

# ==========================================
# 3. The Master Graph State
# ==========================================
class GraphState(BaseModel):
    """
    The master state object for the LangGraph orchestrator.
    Acts as the shared whiteboard for the workflow.
    Each node updates only the part of the snapshot it owns.
    """
    
    # Input
    request: ChatRequest

    # Runtime
    execution: ExecutionState = Field(default_factory=ExecutionState)

    # Subsystem Outputs (Populated progressively)
    memory: Optional[MemoryResponse] = None
    retrieval: Optional[RetrievalContext] = None
    generation: Optional[GenerationResponse] = None
    evaluation: Optional[EvaluationReport] = None
    recovery: Optional[RecoveryPlan] = None

    # Final Output
    response: Optional[ResponseState] = None

    # Failure State
    error: Optional[GraphError] = None





# graph/nodes/input_preparation.py

import uuid
import logging
from datetime import datetime, timezone

from graph.state import GraphState, ExecutionState
from graph.models import NodeType

logger = logging.getLogger(__name__)

def input_preparation_node(state: GraphState) -> GraphState:
    """
    The Bootstrap Node.
    Validates the minimal required input and initializes the execution context.
    Does NOT use the GraphTracer because the execution has not formally begun.
    """
    request = state.request


    if not request:
        raise ValueError("Workflow aborted: ChatRequest is missing.")
        
    if not request.query or not request.query.strip():
        raise ValueError("Workflow aborted: The search query cannot be empty.")


    # We do not mutate the immutable ChatRequest. 
    # We derive the IDs and store them strictly in ExecutionState.
    query_id = request.query_id or str(uuid.uuid4())
    request_id = str(uuid.uuid4())


    execution = ExecutionState(
        query_id=query_id,
        request_id=request_id,
        start_time=datetime.now(timezone.utc),
        current_node=NodeType.INPUT_PREPARATION,
        retry_count=0,
        total_latency_ms=0.0
    )

    logger.info(
        f"Bootstrap Complete | QueryID={query_id} | "
        f"RequestID={request_id} | Session={request.session_id}"
    )


    state.execution = execution
    
    return state
import logging
from typing import List
from fastapi import APIRouter, Depends

# API Schemas (Public Contracts)
from schemas.request import ChatRequest
from schemas.response import ChatResponse
from schemas.retrieval import CitationResponse

# Core Dependencies & Domain Models
from api.dependencies import get_graph_workflow
from graph.workflow import GraphWorkflow
from graph.state import GraphState

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)

@router.post("/query", response_model=ChatResponse)
def query(
    request: ChatRequest,
    workflow: GraphWorkflow = Depends(get_graph_workflow)
) -> ChatResponse:
    """
    The main execution endpoint for the Self-Healing RAG system.
    
    Accepts a validated user query, invokes the full LangGraph orchestration, 
    and translates the internal GraphState into a clean, public-facing ChatResponse.
    """
    logger.info(f"Received API chat request | Session={request.session_id}")

    # 1. Execute the LangGraph Workflow
    # We pass the validated ChatRequest directly to the workflow's public method.
    final_state: GraphState = workflow.run(request)

    # 2. Extract State Folders
    # Isolate the specific namespaces from the master GraphState
    execution = final_state.execution
    response_stub = final_state.response
    recovery = final_state.recovery
    evaluation = final_state.evaluation

    # 3. Map Citations
    # Translate internal GenerationCitation models into public CitationResponse schemas.
    mapped_citations: List[CitationResponse] = []
    if response_stub and response_stub.citations:
        for cit in response_stub.citations:
            mapped_citations.append(
                CitationResponse(
                    chunk_id=cit.chunk_id,
                    document_id=cit.document_id,
                    document_name=cit.title,
                    source_type=cit.source,
                    page_number=cit.page,
                    relevance_score=cit.score,
                    inline_reference=getattr(cit, "inline_reference", "")
                )
            )

    # 4. Map Self-Healing Telemetry (THE FIX)
    correction_path: List[str] = []
    # Pull from the historical RetryState, because recovery.actions gets cleared on success!
    if recovery and recovery.retry_state and recovery.retry_state.visited_action_sequences:
        raw_sequences = recovery.retry_state.visited_action_sequences
        # Flatten the historical strings (e.g., "rewrite_query->retry_retrieval") back into a list
        for seq in raw_sequences:
            correction_path.extend(seq.split("->"))

    # 5. Map Non-Fatal Warnings
    warnings: List[str] = []
    if evaluation and getattr(evaluation, "warnings", None):
        warnings.extend(evaluation.warnings)

    # 6. Assemble the Final Public API Response
    # The router acts as the ultimate Anti-Corruption Layer, ensuring no internal
    # prompts, raw context blocks, or judge reasoning leak to the frontend.
    return ChatResponse(
        query_id=execution.query_id,
        session_id=request.session_id,
        status=response_stub.status if response_stub else "FAILED",
        answer=response_stub.answer if response_stub else "An orchestration error occurred.",
        citations=mapped_citations,
        confidence=response_stub.confidence if response_stub else 0.0,
        correction_path=correction_path,
        retry_count=execution.retry_count,
        latency_ms=execution.total_latency_ms,
        recovery_used=response_stub.recovery_used if response_stub else False,
        warnings=warnings
    )
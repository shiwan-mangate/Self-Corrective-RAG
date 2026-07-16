## api/routers/chat.py
import logging
from typing import List
from fastapi import APIRouter, Depends


from schemas.request import ChatRequest
from schemas.response import ChatResponse
from schemas.retrieval import CitationResponse


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

   
    final_state: GraphState = workflow.run(request)


    execution = final_state.execution
    response_stub = final_state.response
    recovery = final_state.recovery
    evaluation = final_state.evaluation

    
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


    correction_path: List[str] = []
   
    if recovery and recovery.retry_state and recovery.retry_state.visited_action_sequences:
        raw_sequences = recovery.retry_state.visited_action_sequences
       
        for seq in raw_sequences:
            correction_path.extend(seq.split("->"))


    warnings: List[str] = []
    if evaluation and getattr(evaluation, "warnings", None):
        warnings.extend(evaluation.warnings)

    
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
# api/routers/evaluation.py

import logging
from fastapi import APIRouter, Depends

# Public Schemas
from schemas.evaluation import (
    EvaluationResponse,
    GroundingResponse,
    HallucinationResponse,
    ConfidenceResponse,
    RagasResponse
)

# Core Dependencies
from api.dependencies import get_evaluation_repository
from database.repositories.evaluation_repository import EvaluationRepository

# API Exceptions (Clean Architecture boundary)
from api.exceptions import EvaluationNotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/evaluation",
    tags=["Evaluation"]
)


@router.get("/{query_id}", response_model=EvaluationResponse)
def get_evaluation(
    query_id: str,
    evaluation_repository: EvaluationRepository = Depends(get_evaluation_repository)
) -> EvaluationResponse:
    """
    Retrieves the finalized quality metrics for a specific query execution.
    
    Acts as a strict READ API over the evaluation_runs table to prevent 
    triggering expensive LLM Judges or benchmarking routines unnecessarily.
    """
    logger.info(f"Evaluation Read requested | QueryID={query_id}")
 
    run = evaluation_repository.get_latest_by_query_id(query_id)
    
    if not run:
        logger.warning(f"Evaluation read failed: query_id '{query_id}' not found.")

        raise EvaluationNotFoundError(f"No evaluation result was found for query_id '{query_id}'.")


    return EvaluationResponse(
        grounding=GroundingResponse(
            is_grounded=run.is_grounded,
            confidence=run.grounding_confidence
        ),
        hallucination=HallucinationResponse(
            detected=run.has_hallucination,
            risk=run.hallucination_risk
        ),
        confidence=ConfidenceResponse(
            score=run.overall_confidence,
            retrieval_confidence=run.retrieval_confidence,
            grounding_confidence=run.grounding_confidence,
            hallucination_risk=run.hallucination_risk
        ),
        ragas=RagasResponse(
            faithfulness=run.faithfulness,
            answer_relevancy=run.answer_relevancy,
            context_recall=run.context_recall,
            context_precision=run.context_precision
        ),
       
        retry_recommended=run.retry_recommendation is not None
    )
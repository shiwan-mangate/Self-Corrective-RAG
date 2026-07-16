import logging
from sqlalchemy.orm import Session
from typing import Callable

from evaluation.models import EvaluationRequest, EvaluationReport
from database.models.evaluation_run import EvaluationRun

logger = logging.getLogger(__name__)

class EvaluationLogger:
    """
    Persistence layer for the Evaluation Subsystem.
    
    Responsibility:
    Translate the in-memory EvaluationRequest and EvaluationReport domain models 
    into a flattened SQLAlchemy ORM record and persist it to PostgreSQL.
    """

    def __init__(self, session_factory: Callable[[], Session]):
        self.session_factory = session_factory

    def log(self, request: EvaluationRequest, report: EvaluationReport) -> None:
        """
        Extracts fields from the domain models, constructs an EvaluationRun, 
        and safely commits it to the database with explicit transaction safety.
        """
        try:
          
            eval_mode = report.evaluation_metadata.evaluation_mode
            eval_mode_str = eval_mode.value if hasattr(eval_mode, "value") else str(eval_mode)
            
           
            run_record = EvaluationRun(
                
                query_id=request.query_id,
                evaluation_timestamp=report.evaluation_metadata.timestamp,

               
                original_query=request.original_query,
                optimized_query=request.optimized_query,
                generated_answer=request.answer,

                
                is_grounded=report.grounding.is_grounded,
                supported_claims=report.grounding.supported_claims,
                unsupported_claims=report.grounding.unsupported_claims,
                grounding_reason=report.grounding.explanation,

               
                has_hallucination=report.hallucination.has_hallucination,
                hallucinated_claims=report.hallucination.hallucinated_claims,
                hallucination_reason=report.hallucination.reasoning,

               
                overall_confidence=report.confidence.overall_score,
                retrieval_confidence=report.confidence.retrieval_confidence,
                grounding_confidence=report.confidence.grounding_confidence,
                hallucination_risk=report.confidence.hallucination_risk.value,
                confidence_reason=report.confidence.explanation,

           
                faithfulness=report.ragas.faithfulness_score,
                answer_relevancy=report.ragas.answer_relevancy_score,
                context_precision=report.ragas.context_precision_score,
                context_recall=report.ragas.context_recall_score,
                ragas_latency_ms=report.ragas.latency_ms,

                
                decision=report.decision.value,
                retry_recommendation=(
                    report.retry_recommendation.value if report.retry_recommendation else None
                ),
                warnings=report.warnings,

              
                retrieval_latency_ms=request.retrieval_metadata.latency_ms,
                generation_latency_ms=request.generation_metadata.latency_ms,
                evaluation_latency_ms=report.evaluation_metadata.latency_ms,
                total_tokens=request.generation_metadata.token_usage.total_tokens,
                generation_model_name=request.generation_metadata.model_name,
                judge_model_name=report.evaluation_metadata.judge_model_name,
                evaluation_mode=eval_mode_str,
                evaluation_version=report.evaluation_metadata.version
            )
        except Exception as e:
           
            logger.exception(f"Failed to map Evaluation models to ORM record: {e}")
            return

       
        session = self.session_factory()
        try:
            session.add(run_record)
            session.commit()
            
          
            logger.info(f"EvaluationTelemetry successfully persisted | QueryID={request.query_id}")

        except Exception as e:
            session.rollback()
            logger.exception(f"Failed to persist Evaluation Telemetry to PostgreSQL: {e}")
            
        finally:
            session.close()
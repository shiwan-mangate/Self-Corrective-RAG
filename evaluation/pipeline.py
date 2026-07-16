## evaluation/pipeline.py
import time
import logging
from typing import Optional

from evaluation.models import (
    EvaluationRequest, 
    EvaluationReport, 
    EvaluationDecision, 
    RetryRecommendation,
    EvaluationMetadata,
    GroundingResult,
    HallucinationResult,
    RagasResult,
    ConfidenceResult
)
from evaluation.constants import (
    MIN_OVERALL_CONFIDENCE_SCORE,
    MIN_REVIEW_CONFIDENCE_SCORE,
    DEFAULT_JUDGE_MODEL,
    MIN_GOOD_RETRIEVAL_SIMILARITY
)
from evaluation.ragas.metrics import RagasEvaluationMode
from retrieval.models import RetrievalMetadata
from evaluation.logger import EvaluationLogger
# Import Component Contracts
from evaluation.grounding.base import BaseGroundingVerifier
from evaluation.hallucination.base import BaseHallucinationDetector
from evaluation.confidence.base import BaseConfidenceScorer
from evaluation.ragas.dataset_builder import RagasDatasetBuilder
from evaluation.ragas.evaluator import RagasEvaluator

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """
    The Master Orchestrator for the Evaluation Subsystem.
    """

    def __init__(
        self,
        grounding_verifier: BaseGroundingVerifier,
        hallucination_detector: BaseHallucinationDetector,
        confidence_scorer: BaseConfidenceScorer,
        ragas_builder: RagasDatasetBuilder,
        ragas_evaluator: RagasEvaluator,
        evaluation_logger: EvaluationLogger,
        default_mode: RagasEvaluationMode = RagasEvaluationMode.LIVE
    ):
        self.grounding = grounding_verifier
        self.hallucination = hallucination_detector
        self.confidence = confidence_scorer
        self.ragas_builder = ragas_builder
        self.ragas_evaluator = ragas_evaluator
        self.default_mode = default_mode
        self.logger = evaluation_logger
        

    def evaluate(
        self, 
        request: EvaluationRequest, 
        mode: Optional[RagasEvaluationMode] = None,
        ground_truth: Optional[str] = None
    ) -> EvaluationReport:
        
        start_time = time.perf_counter()
        exec_mode = mode or self.default_mode

        logger.info(
            f"Evaluation Pipeline Started | QueryID={request.query_id} | "
            f"Mode={exec_mode.value.upper()} | "
            f"Query={request.original_query[:80]}"
        )

        
        grounding_res = self.grounding.verify(request)
        hallucination_res = self.hallucination.detect(request)
        ragas_res = self._evaluate_ragas(request, exec_mode, ground_truth)
        
        confidence_res = self.confidence.score(
            request=request, 
            grounding=grounding_res, 
            hallucination=hallucination_res,
            ragas=ragas_res
        )

       
        decision = self._determine_decision(grounding_res, hallucination_res, confidence_res)
        
        recommendation = self._determine_retry(
            decision=decision, 
            retrieval_metadata=request.retrieval_metadata, 
            grounding=grounding_res, 
            hallucination=hallucination_res
        )

        
        metadata = EvaluationMetadata(
            latency_ms=round((time.perf_counter() - start_time) * 1000, 2),
            evaluation_mode=exec_mode.value,
            judge_model_name=DEFAULT_JUDGE_MODEL 
        )

        report = EvaluationReport(
            decision=decision,
            grounding=grounding_res,
            hallucination=hallucination_res,
            confidence=confidence_res,
            ragas=ragas_res,
            evaluation_metadata=metadata,
            retry_recommendation=recommendation
        )

      
        logger.info(
            f"Evaluation Pipeline Completed | QueryID={request.query_id} | "
            f"Decision={decision.value.upper()} | "
            f"Confidence={confidence_res.overall_score:.2f} | "
            f"Retry={recommendation.value if recommendation else 'NONE'} | "
            f"Latency={metadata.latency_ms}ms"
        )
        self.logger.log(request, report)

        return report

    def _evaluate_ragas(self, request: EvaluationRequest, mode: RagasEvaluationMode, ground_truth: Optional[str]) -> RagasResult:
        """Safely build and execute RAGAS."""
        try:
            dataset = self.ragas_builder.build(request, ground_truth)
            return self.ragas_evaluator.evaluate(dataset, mode)
        except Exception:
            logger.exception("RAGAS execution failed. Falling back to default scores.")
            return RagasResult(
                faithfulness_score=0.0, 
                answer_relevancy_score=0.0,
                context_precision_score=None,
                context_recall_score=None,
                latency_ms=0.0
            )

    def _determine_decision(
        self, 
        grounding: GroundingResult, 
        hallucination: HallucinationResult, 
        confidence: ConfidenceResult
    ) -> EvaluationDecision:
        """Policy Engine: Synthesizes all signals."""
        if hallucination.has_hallucination or not grounding.is_grounded:
            return EvaluationDecision.FAIL
            
        if confidence.overall_score >= MIN_OVERALL_CONFIDENCE_SCORE:
            return EvaluationDecision.PASS
            
        if confidence.overall_score >= MIN_REVIEW_CONFIDENCE_SCORE:
            return EvaluationDecision.NEEDS_REVIEW
            
        return EvaluationDecision.FAIL

    def _determine_retry(
        self, 
        decision: EvaluationDecision, 
        retrieval_metadata: RetrievalMetadata,
        grounding: GroundingResult, 
        hallucination: HallucinationResult
    ) -> Optional[RetryRecommendation]:
        """The Self-Healing Router: Routes failure to specific corrective actions."""
        if decision == EvaluationDecision.PASS:
            return None

        if hallucination.has_hallucination:
            return RetryRecommendation.STRICT_GROUNDING
        
        if not grounding.is_grounded:
            retrieval_sim = retrieval_metadata.statistics.average_similarity
            
            if retrieval_sim >= MIN_GOOD_RETRIEVAL_SIMILARITY:
                return RetryRecommendation.WEB_SEARCH
            return RetryRecommendation.REWRITE_QUERY
            
        return RetryRecommendation.ASK_CLARIFICATION
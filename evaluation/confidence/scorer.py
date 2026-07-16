import logging
from evaluation.models import (
    EvaluationRequest, 
    GroundingResult, 
    HallucinationResult, 
    ConfidenceResult,
    HallucinationRisk,
    RagasResult
)
from evaluation.confidence.base import BaseConfidenceScorer
from evaluation.constants import (
    WEIGHT_RETRIEVAL,
    WEIGHT_GROUNDING,
    MIN_OVERALL_CONFIDENCE_SCORE,
    HIGH_HALLUCINATION_PENALTY,
    MEDIUM_HALLUCINATION_PENALTY
)

logger = logging.getLogger(__name__)

class DeterministicConfidenceScorer(BaseConfidenceScorer):
    """
    Calculates system confidence using deterministic mathematics.
    Fast, zero-cost, strictly separated from offline RAGAS metrics.
    """

    def score(
        self, 
        request: EvaluationRequest, 
        grounding: GroundingResult, 
        ragas: RagasResult,
        hallucination: HallucinationResult
    ) -> ConfidenceResult:
        
        
        retrieval_conf = request.retrieval_metadata.statistics.average_similarity
        grounding_conf = grounding.confidence
        
        
        hal_risk = hallucination.risk

        
        total_weight = WEIGHT_RETRIEVAL + WEIGHT_GROUNDING
        norm_retrieval = WEIGHT_RETRIEVAL / total_weight
        norm_grounding = WEIGHT_GROUNDING / total_weight 
        base_score = (retrieval_conf * norm_retrieval) + (grounding_conf * norm_grounding)
        
     
        if hal_risk == HallucinationRisk.HIGH:
            base_score *= HIGH_HALLUCINATION_PENALTY
        elif hal_risk == HallucinationRisk.MEDIUM:
            base_score *= MEDIUM_HALLUCINATION_PENALTY
            
        overall_score = max(0.0, min(1.0, base_score))
        
      
        explanation = self._build_explanation(overall_score, hal_risk, retrieval_conf, grounding_conf)
        
      
        logger.info(
            f"Confidence | "
            f"Overall={overall_score:.2f} | "
            f"Retrieval={retrieval_conf:.2f} | "
            f"Grounding={grounding_conf:.2f} | "
            f"Hallucination={hal_risk.value.upper()} | "
            f"Search={request.retrieval_metadata.search_strategy.value}"
        )
        
        return ConfidenceResult(
            overall_score=round(overall_score, 4),
            retrieval_confidence=round(retrieval_conf, 4),
            grounding_confidence=round(grounding_conf, 4),
            hallucination_risk=hal_risk,
            explanation=explanation
        )

    def _build_explanation(self, overall: float, hal_risk: HallucinationRisk, ret: float, grnd: float) -> str:
        """Generates a clean reasoning string devoid of hardcoded numbers."""
        if hal_risk == HallucinationRisk.HIGH:
            return "CRITICAL: Active hallucination detected. Score severely penalized."
        if hal_risk == HallucinationRisk.MEDIUM:
            return "WARNING: Medium hallucination risk due to unsupported claims. Score penalized."
            
        if overall >= MIN_OVERALL_CONFIDENCE_SCORE:
            return "PASS: High confidence across all active signals."
            
        if grnd < 0.5:
            return "WARNING: Poor grounding in retrieved context compromised confidence."
        if ret < 0.6:
            return "WARNING: Weak initial retrieval evidence compromised confidence."
            
        return "NEEDS REVIEW: Ambiguous confidence signals."
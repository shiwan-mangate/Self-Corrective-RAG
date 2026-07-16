from abc import ABC, abstractmethod
from evaluation.models import EvaluationRequest, GroundingResult, HallucinationResult, ConfidenceResult, RagasResult

class BaseConfidenceScorer(ABC):
    """
    Abstract contract for the Confidence Scorer.
    
    Responsibility:
    Deterministically combine retrieval telemetry and live evaluation 
    results into a final, explainable ConfidenceResult. 
    """

    @abstractmethod
    def score(
        self, 
        request: EvaluationRequest, 
        grounding: GroundingResult, 
        ragas: RagasResult,
        hallucination: HallucinationResult
    ) -> ConfidenceResult:
        raise NotImplementedError
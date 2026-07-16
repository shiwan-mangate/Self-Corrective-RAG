from abc import ABC, abstractmethod
from evaluation.models import EvaluationRequest, HallucinationResult

class BaseHallucinationDetector(ABC):
    """
    Abstract contract for the Hallucination Detector.
    
    Responsibility:
    Accepts an EvaluationRequest (containing question, context, and answer) 
    and returns a strict HallucinationResult indicating if the model actively 
    invented or contradicted facts.
    """

    @abstractmethod
    def detect(self, request: EvaluationRequest) -> HallucinationResult:
        """
        Evaluates the answer for fabrications and contradictions.
        """
        raise NotImplementedError
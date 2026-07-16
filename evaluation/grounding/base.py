from abc import ABC, abstractmethod
from evaluation.models import EvaluationRequest, GroundingResult

class BaseGroundingVerifier(ABC):
    """
    Abstract contract for the Grounding Verifier.
    
    Responsibility:
    Accepts an EvaluationRequest containing the context and the generated answer, 
    and returns a strictly typed GroundingResult indicating whether the claims 
    in the answer are supported by the provided evidence.
    """

    @abstractmethod
    def verify(self, request: EvaluationRequest) -> GroundingResult:
        """
        Evaluates the answer against the context and returns the grounding metrics.
        """
        raise NotImplementedError
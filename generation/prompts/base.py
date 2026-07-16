from abc import ABC, abstractmethod
from generation.models import GenerationRequest, PromptPackage

class BasePromptBuilder(ABC):
    """
    Abstract contract for all Generation Prompt Builders.
    """
    @abstractmethod
    def build(self, request: GenerationRequest, strict_grounding: bool) -> PromptPackage:
        raise NotImplementedError
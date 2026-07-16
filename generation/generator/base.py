from abc import ABC, abstractmethod
from generation.models import PromptPackage, GeneratedAnswer

class BaseAnswerGenerator(ABC):
    """
    Abstract contract for the core LLM execution engine.
    
    Responsibility:
    Accepts a finalized PromptPackage and returns a strictly typed GeneratedAnswer.
    """

    @abstractmethod
    def generate(self, prompt_package: PromptPackage) -> GeneratedAnswer:
        raise NotImplementedError
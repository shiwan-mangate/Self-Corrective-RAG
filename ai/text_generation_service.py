from abc import ABC, abstractmethod


class AITextGenerationService(ABC):
    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        query_id: str, 
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        pass
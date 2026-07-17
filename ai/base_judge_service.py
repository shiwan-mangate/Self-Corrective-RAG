# ai/base_judge_service.py
from abc import ABC, abstractmethod
from typing import TypeVar, Type
from pydantic import BaseModel


T = TypeVar('T', bound=BaseModel)

class BaseAIJudgeService(ABC):
    """
    Abstract contract for high-capability Judge models (e.g., openai/gpt-oss-120b).
    Sits inside the infrastructure boundary (ai/) to shield domain layers from
    raw text cleaning, network calls, transport encoding, and API timeouts.
    """

    @abstractmethod
    def evaluate(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        timeout_ms: int,
        query_id: str,
        response_model: Type[T]
    ) -> T:
        """
        Executes an evaluation prompt, enforces the schema, and returns a validated Pydantic object.
        
        Implementation Responsibilities:
            - Generate JSON schema from the provided response_model.
            - Execute the network call with the specified timeout_ms limit.
            - Parse raw text strings and clean markdown fences.
            - Return response_model.model_validate(parsed_json).
            - Raise explicit infrastructure exceptions on failure.
        """
        raise NotImplementedError
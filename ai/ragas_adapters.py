import os
import sys
import asyncio
import numpy as np
from typing import List, Optional, Any
from unittest.mock import MagicMock

# --- START BUGFIX FOR RAGAS ---
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()
sys.modules['langchain_community.llms.vertexai'] = MagicMock()
# --- END BUGFIX ---

from langchain_core.outputs import ChatResult
from langchain_groq import ChatGroq

from ai.base_ragas_service import BaseRagasLLM, BaseRagasEmbeddings
from ai.embedding_service import AIEmbeddingService
from config.settings import settings  # Clear, uniform config mapping


class NativeRagasEmbeddings(BaseRagasEmbeddings):
    """
    Adapter Pattern: Converts our fast numpy AIEmbeddingService into the 
    strict List[float] format required by the RAGAS library.
    """
    def __init__(self, embedding_service: AIEmbeddingService):
        self.service = embedding_service

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors: np.ndarray = self.service.embed_texts(texts)
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        vectors: np.ndarray = self.service.embed_texts([text])
        return vectors[0].tolist()


class GroqRagasLLM(ChatGroq, BaseRagasLLM):
    """
    Interceptor Pattern: Groq strictly limits the generation parameter 'n' to 1.
    RAGAS frequently asks for n > 1 for AnswerRelevancy metrics. 
    This adapter intercepts 'n' at the lowest Langchain hook (_agenerate), 
    strip it from the payload, spawns concurrent requests, and merges the results.
    """
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.0, **kwargs):
        api_key = kwargs.get("api_key") or settings.GROQ_API_KEY
        if not api_key:
            raise ValueError("GROQ_API_KEY not found. Please set it in your Render Environment Variables.")
            
        super().__init__(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            **kwargs
        )

    def _generate(
        self,
        messages: List[Any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        n = kwargs.pop("n", getattr(self, "n", None)) or 1
        original_n = getattr(self, "n", None)
        
        self.n = None
        kwargs.pop("n", None)

        try:
            if n <= 1:
                return super()._generate(messages, stop, run_manager, **kwargs)

            results = []
            for _ in range(n):
                res = super()._generate(messages, stop, run_manager, **kwargs)
                results.append(res)

            combined_generations = []
            for res in results:
                combined_generations.extend(res.generations)
                
            return ChatResult(generations=combined_generations, llm_output=results[0].llm_output)
        finally:
            self.n = original_n

    async def _agenerate(
        self,
        messages: List[Any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        n = kwargs.pop("n", getattr(self, "n", None)) or 1
        original_n = getattr(self, "n", None)

        self.n = None
        kwargs.pop("n", None)

        try:
            if n <= 1:
                return await super()._agenerate(messages, stop, run_manager, **kwargs)

            tasks = [
                super(GroqRagasLLM, self)._agenerate(messages, stop, run_manager, **kwargs) 
                for _ in range(n)
            ]
            results = await asyncio.gather(*tasks)

            combined_generations = []
            for res in results:
                combined_generations.extend(res.generations)
                
            return ChatResult(generations=combined_generations, llm_output=results[0].llm_output)
        finally:
            self.n = original_n
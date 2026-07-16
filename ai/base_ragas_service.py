import numpy as np
from typing import List, Any
from abc import ABC


from ai.embedding_service import AIEmbeddingService
from ai.base_judge_service import BaseAIJudgeService

class BaseRagasEmbeddings(ABC):
    """
    Wrapper to make our AIEmbeddingService compatible with RAGAS/LangChain.
    RAGAS expects an object with 'embed_documents' and 'embed_query' methods 
    that return Lists of floats, NOT numpy arrays.
    """
    def __init__(self, embedding_service: AIEmbeddingService):
        self.service = embedding_service

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
      
        vectors: np.ndarray = self.service.embed_texts(texts)
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        vectors: np.ndarray = self.service.embed_texts([text])
        return vectors[0].tolist()


class BaseRagasLLM(ABC):
    """
    Wrapper to make our Judge or Generation service compatible with RAGAS.
    (Note: Depending on the specific RAGAS version, you may actually want to use 
    Langchain's ChatGroq directly here, as RAGAS expects an LLM that can handle 
    its internal prompt templates and asynchronous batching.)
    """
    pass
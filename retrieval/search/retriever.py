import logging
import time
from typing import List
from pydantic import BaseModel, Field
from retrieval.models import AnalyzedQuery, RetrievedChunk
from ai.embedding_service import AIEmbeddingService
from database.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)


class RetrievalResult(BaseModel):
    analyzed_query: AnalyzedQuery
    chunks: List[RetrievedChunk] = Field(default_factory=list)


class Retriever:
    """
    The vector execution boundary for the Retrieval Subsystem.
    
    Responsibility:
    Accepts a fully analyzed (and potentially rewritten) query, converts it into 
    dense vector space, and fetches raw chunks from the read-side repository.
    """

    def __init__(
        self,
        embedding_service: AIEmbeddingService,
        retrieval_service: RetrievalService
    ):
        self.embedding_service = embedding_service
        self.retrieval_service = retrieval_service

    def retrieve(self, analyzed_query: AnalyzedQuery) -> RetrievalResult:
        """
        Executes the vector search pipeline: Embed -> Fetch.
        """
        start_time = time.time()
        
       
        if not analyzed_query.normalized_query:
            logger.warning(
                f"Analyzed query is empty. Skipping engine lookup. "
                f"Original input: '{analyzed_query.original_query}'"
            )
            return RetrievalResult(analyzed_query=analyzed_query, chunks=[])
        query_text = analyzed_query.rewritten_query or analyzed_query.normalized_query
        query_vector = self._embed_query(query_text)
        chunks = self._execute_search(query_vector, analyzed_query)
        elapsed_time = time.time() - start_time
        
        logger.info(
            f"Retriever Performance Success | Intent: {analyzed_query.intent.value.upper()} | "
            f"Strategy: {analyzed_query.search_type.value.upper()} | Top K: {analyzed_query.top_k} | "
            f"Model: {self.embedding_service.model_name} | Count Fetched: {len(chunks)} | "
            f"Pipeline Time: {elapsed_time:.4f}s"
        )
        
        return RetrievalResult(analyzed_query=analyzed_query, chunks=chunks)

    def _embed_query(self, text_to_embed: str) -> List[float]:
        """Converts plain text query elements into multi-dimensional dense vector arrays."""
        query_embeddings = self.embedding_service.embed_texts([text_to_embed], batch_size=1)
        
        # FIX: Use 'is None' instead of 'not' to prevent NumPy truth value crashes
        if query_embeddings is None or len(query_embeddings) == 0:
            raise RuntimeError("The shared AI embedding engine returned an empty or corrupt vector payload.")
            
        target_vector = query_embeddings[0]
        return target_vector.tolist() if hasattr(target_vector, "tolist") else target_vector

    def _execute_search(
        self, 
        query_vector: List[float], 
        analyzed_query: AnalyzedQuery
    ) -> List[RetrievedChunk]:
        """Isolates execution strategy decisions from the main transaction flow."""
        return self.retrieval_service.search_by_similarity(
            query_embedding=query_vector,
            top_k=analyzed_query.top_k
        )
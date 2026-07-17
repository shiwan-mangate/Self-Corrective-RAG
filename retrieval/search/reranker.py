# retrieval/search/reranker.py
import logging
import time
from typing import List, Tuple
from retrieval.models import RetrievedChunk, RankedChunk, AnalyzedQuery

logger = logging.getLogger(__name__)


class Reranker:
    """
    The cognitive sorting boundary of the Retrieval Subsystem.
    """

    def rerank(
        self, 
        chunks: List[RetrievedChunk], 
        analyzed_query: AnalyzedQuery
    ) -> List[RankedChunk]:
        
        if not chunks:
            return []

        start_time = time.time()
        normalized_query = analyzed_query.normalized_query
        
        scored_chunks: List[Tuple[RetrievedChunk, float]] = [
            (chunk, self._compute_final_score(chunk, normalized_query)) 
            for chunk in chunks
        ]
      
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        ranked_chunks: List[RankedChunk] = [
            self._build_ranked_chunk(chunk, score, rank=index + 1)
            for index, (chunk, score) in enumerate(scored_chunks)
        ]
        
        top_score = ranked_chunks[0].final_score if ranked_chunks else 0.0
        elapsed_time = time.time() - start_time
        
        logger.info(
            f"Reranker Metrics | Intent: {analyzed_query.intent.value.upper()} | "
            f"Strategy: {analyzed_query.search_type.value.upper()} | "
            f"Input: {len(chunks)} | Output: {len(ranked_chunks)} | "
            f"Top Final Score: {top_score:.4f} | Time: {elapsed_time:.4f}s"
        )
        
        return ranked_chunks

    def _compute_final_score(self, chunk: RetrievedChunk, query: str) -> float:
        """
        Version 1: Identity mapping (returns the base similarity score).
        # FUTURE: CrossEncoder inference will replace this method directly.
        """
        return chunk.similarity_score

    def _build_ranked_chunk(self, chunk: RetrievedChunk, score: float, rank: int) -> RankedChunk:
        chunk_data = chunk.model_dump()
        
        return RankedChunk(
            **chunk_data,
            retrieval_score=chunk.similarity_score,
            rerank_score=score,
            final_score=score,
            rank=rank
        )
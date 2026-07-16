import logging
import time
from typing import List, Tuple

from retrieval.models import (
    RankedChunk, 
    RetrievalContext, 
    AnalyzedQuery, 
    SearchQuery, 
    Citation, 
    RetrievalMetadata, 
    RetrievalStatistics
)
from config.constants import DEFAULT_TOKEN_BUDGET

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Transforms RankedChunks into a strict, token-budgeted RetrievalContext for the LLM.
    """

    def __init__(self, token_budget: int = DEFAULT_TOKEN_BUDGET):
        self.token_budget = token_budget

    def build(
        self, 
        ranked_chunks: List[RankedChunk], 
        analyzed_query: AnalyzedQuery,
        original_query: SearchQuery,
        pipeline_statistics: RetrievalStatistics 
    ) -> RetrievalContext:
        
        start_time = time.time()
        
        selected_chunks, used_tokens = self._select_chunks(ranked_chunks)
        
        context_string = self._build_context_text(selected_chunks)
        
        citations = self._collect_citations(selected_chunks)
        
        pipeline_statistics.context_tokens = used_tokens
        if selected_chunks:
            avg_sim = sum(c.final_score for c in selected_chunks) / len(selected_chunks)
            pipeline_statistics.average_similarity = round(avg_sim, 4)
        
        metadata = RetrievalMetadata(
            search_strategy=analyzed_query.search_type,
            latency_ms=round((time.time() - start_time) * 1000, 2),
            statistics=pipeline_statistics
        )
        
        return RetrievalContext(
            question=original_query.query,
            rewritten_question=(analyzed_query.rewritten_query if analyzed_query.rewrite_performed else None), # ✅ FIXED
            context=context_string,
            chunks=selected_chunks,
            citations=citations,
            metadata=metadata
        )

    def _select_chunks(self, chunks: List[RankedChunk]) -> Tuple[List[RankedChunk], int]:
        selected: List[RankedChunk] = []
        current_tokens = 0
        
        for chunk in chunks:
            if current_tokens + chunk.token_count <= self.token_budget:
                selected.append(chunk)
                current_tokens += chunk.token_count
            else:
                continue
                
        return selected, current_tokens

    def _build_context_text(self, chunks: List[RankedChunk]) -> str:
        if not chunks:
            return "No relevant context found."
            
        context_blocks = []
        for chunk in chunks:
            source = chunk.source or "Unknown Source"
            title = chunk.document_title or "Untitled Document"
            
            
            block = (
                f"--- Evidence {chunk.rank} ---\n"
                f"Rank: {chunk.rank} | Source: {source} | Title: {title}\n"
                f"Content:\n{chunk.text.strip()}\n"
            )
            context_blocks.append(block)
            
        return "\n\n".join(context_blocks)

    def _collect_citations(self, chunks: List[RankedChunk]) -> List[Citation]:
        citations: List[Citation] = []
        
        for chunk in chunks:
            start_loc = chunk.metadata.get("start_location", {})
            page_num = start_loc.get("page")
            
            citations.append(
                Citation(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    title=chunk.document_title,
                    source=chunk.source,
                    page=page_num,
                    score=round(chunk.final_score, 4)
                )
            )
            
        return citations
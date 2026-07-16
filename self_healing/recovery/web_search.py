import time
import logging
from typing import List
from datetime import datetime, timezone


from tavily import TavilyClient

from self_healing.recovery.retry_manager import RetryManager
from self_healing.constants import (
    MAX_WEB_RESULTS, 
    WEB_SEARCH_TIMEOUT_SEC
)

from retrieval.models import RankedChunk
from config.settings import settings

logger = logging.getLogger(__name__)


class WebSearchService:
    """
    Acts as an External Retriever.
    Queries the live internet (via Tavily) to supplement missing internal knowledge.
    Returns normalized `RankedChunk` objects to maintain architectural symmetry with pgvector.
    """

    def __init__(self, retry_manager: RetryManager):
        self.retry_manager = retry_manager

        api_key = settings.TAVILY_API_KEY
        if not api_key:
            raise ValueError("TAVILY_API_KEY is missing from environment variables.")
            
        self.client = TavilyClient(api_key=api_key)

    def search(self, query: str) -> List[RankedChunk]:
        """
        Executes the web search, handles timeouts, and normalizes the results.
        Returns an empty list if the search is aborted or fails.
        """
        logger.info(f"WebSearchService invoked for query: '{query}'")

        if not self.retry_manager.can_use_web_search():
            logger.warning("Web search aborted: Tavily was already called in this recovery loop.")
            return []

        start_time = time.time()
        try:

            raw_response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=MAX_WEB_RESULTS,
                include_answer=False, 
                include_raw_content=False,
            )
            
            elapsed_time = time.time() - start_time
            results = raw_response.get("results", [])
            
            logger.info(f"Web Search Complete | Hits: {len(results)} | Latency: {elapsed_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Web Search failed: {str(e)}")
            self.retry_manager.increment_retry(failure_reason=f"tavily_api_failure: {str(e)}")
            return []

   
        self.retry_manager.mark_web_search_used()

        # 4. Normalization
        normalized_chunks = self._normalize_results(results, query)
        
        logger.info(f"Successfully normalized {len(normalized_chunks)} Web Chunks.")
        return normalized_chunks

    def _normalize_results(self, tavily_results: List[dict], original_query: str) -> List[RankedChunk]:
        """
        Converts raw Tavily dictionaries into standardized RankedChunk objects.
        """
        normalized = []
        for index, result in enumerate(tavily_results):
            # Fallbacks for safety
            url = result.get("url", "unknown_url")
            title = result.get("title", "Web Source")
            content = result.get("content", "")
            

            score = result.get("score", 0.0) 

            if not content:
                continue

            # TODO: Tech Debt - Refactoring Pass Required Later
            # 1. Token count is an estimate. Needs tiktoken utility.
            # 2. Checksum generation is reusing RetryManager's private method. 
            # Both should be moved to a `shared/utils.py` once the Self-Healing subsystem is complete.
            estimated_tokens = len(content.split())
            checksum = self.retry_manager._hash_text(content)

            chunk = RankedChunk(
                chunk_id=f"web_chunk_{index}",
                document_id=url,  
                source="web",
                document_title=title,
                text=content,
                metadata={
                    "url": url, 
                    "provider": "tavily",
                    "search_query": original_query,
                    "retrieved_at": datetime.now(timezone.utc).isoformat()
                },
                similarity_score=score,
                retrieval_score=score,
                final_score=score, 
                rank=index + 1,
                token_count=estimated_tokens, 
                checksum=checksum 
            )
            normalized.append(chunk)

        return normalized
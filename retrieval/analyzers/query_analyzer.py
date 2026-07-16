import logging
import re
from typing import List, Dict

from retrieval.models import SearchQuery, AnalyzedQuery, QueryIntent, SearchType
from config.constants import (
    DEFAULT_TOP_K,
    MAX_TOP_K,
    MIN_TOP_K,
    SHORT_QUERY_WORDS,
    DEFAULT_SEARCH_TYPE
)
from retrieval.analyzers.base import BaseQueryAnalyzer
from retrieval.analyzers.intent import IntentDetector

logger = logging.getLogger(__name__)


class QueryAnalyzer(BaseQueryAnalyzer):
    """
    The central decision engine for the Retrieval Subsystem.
    
    Responsibility:
    Evaluates a raw SearchQuery and produces a detailed retrieval plan (AnalyzedQuery).
    It orchestrates intent detection and strictly determines downstream pipeline settings.
    """
    
    def __init__(self, intent_detector: IntentDetector):
        self.intent_detector = intent_detector

    def analyze(self, search_query: SearchQuery) -> AnalyzedQuery:
        """
        Orchestrates the decision-making process to build a comprehensive retrieval plan.
        """

        normalized_query = self._normalize(search_query.query)
        word_count = len(normalized_query.split())
     
        intent = self.intent_detector.detect(normalized_query)
        
        
        safe_top_k = max(MIN_TOP_K, min(search_query.top_k, MAX_TOP_K))
        
        top_k = self._determine_top_k(intent, safe_top_k)
        search_type = self._determine_search_strategy(intent, top_k)
        needs_history = self._determine_history(intent, search_query.chat_history)
        needs_rewrite = self._determine_rewrite(intent, word_count)
        
     
        keywords = self._extract_keywords(normalized_query)
        

        logger.info(
            f"Query Plan | Intent: {intent.value.upper()} | Search: {search_type.value.upper()} | "
            f"Top K: {top_k} | Rewrite: {needs_rewrite} | History: {needs_history} | "
            f"Length: {word_count} words"
        )
        
       
        return AnalyzedQuery(
            original_query=search_query.query,
            normalized_query=normalized_query,
            intent=intent,
            needs_history=needs_history,
            needs_rewrite=needs_rewrite,
            search_type=search_type,
            top_k=top_k,
            filters=search_query.filters, 
            entities=[],                 
            keywords=keywords
        )



    def _normalize(self, query: str) -> str:
        """Standardizes text for internal keyword extraction."""
        clean_query = re.sub(r'[^\w\s]', '', query.lower())
        return re.sub(r'\s+', ' ', clean_query).strip()

    def _determine_top_k(self, intent: QueryIntent, safe_top_k: int) -> int:
        """
        Dynamically adjusts retrieval depth based on intent overrides.
        """
        if intent == QueryIntent.SUMMARY:
            return MAX_TOP_K
        if intent == QueryIntent.COMPARISON:
            return max(12, safe_top_k)  
        if intent == QueryIntent.EXPLANATION:
            return max(8, safe_top_k)
            
        return safe_top_k

    def _determine_history(self, intent: QueryIntent, chat_history: List[Dict[str, str]]) -> bool:
        """Decides if the conversation history is required to resolve ambiguity."""
        if not chat_history:
            return False
            
        return intent == QueryIntent.FOLLOW_UP

    def _determine_rewrite(self, intent: QueryIntent, word_count: int) -> bool:
        """
        Flags the query for LLM expansion if it is too vague, short, or context-dependent.
        """
        if intent == QueryIntent.FOLLOW_UP:
            return True
            
        if word_count <= SHORT_QUERY_WORDS:
            return True
            
        return False

    def _determine_search_strategy(self, intent: QueryIntent, top_k: int) -> SearchType:
        """
        Selects the optimal vector database search execution strategy.
        """
        if intent in [QueryIntent.SUMMARY, QueryIntent.COMPARISON] or top_k >= 10:
            return SearchType.MMR
            
        return DEFAULT_SEARCH_TYPE

    def _extract_keywords(self, normalized_query: str) -> List[str]:
        """Simple heuristic keyword extraction."""
        return [w for w in normalized_query.split() if len(w) > 3]
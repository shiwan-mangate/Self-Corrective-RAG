import re
from typing import Set, Tuple

from retrieval.models import QueryIntent
from config.constants import (
    SUMMARY_KEYWORDS,
    COMPARISON_KEYWORDS,
    FOLLOW_UP_KEYWORDS,
    EXPLANATION_KEYWORDS
)


class IntentDetector:
    """
    Determines the categorical intent of a user's query using fast, 
    deterministic rule-based matching.
    
    Responsibility:
    Map a raw query string to a predefined QueryIntent Enum. 
    Does not modify the query, access databases, or call LLMs.
    """

    def detect(self, query: str) -> QueryIntent:
        """
        Evaluates the query against predefined global keyword sets.
        Returns the highest-priority matching QueryIntent, defaulting to UNKNOWN.
        """
        clean_query, words = self._normalize_query(query)

        if self._is_summary(clean_query, words):
            return QueryIntent.SUMMARY

        
        if self._is_comparison(clean_query, words):
            return QueryIntent.COMPARISON

       
        if self._is_follow_up(clean_query, words):
            return QueryIntent.FOLLOW_UP

       
        if self._is_explanation(clean_query, words):
            return QueryIntent.EXPLANATION

    
        return QueryIntent.UNKNOWN

    def _normalize_query(self, query: str) -> Tuple[str, Set[str]]:
        """
        Normalizes the query for both substring phrase matching and exact word matching.
        """
        clean_query = re.sub(r'[^\w\s]', '', query.lower())
        clean_query = re.sub(r'\s+', ' ', clean_query).strip()
        
        words = set(clean_query.split())
        return clean_query, words



    def _is_summary(self, clean_query: str, words: Set[str]) -> bool:
        return self._has_match(clean_query, words, SUMMARY_KEYWORDS)

    def _is_comparison(self, clean_query: str, words: Set[str]) -> bool:
        return self._has_match(clean_query, words, COMPARISON_KEYWORDS)

    def _is_follow_up(self, clean_query: str, words: Set[str]) -> bool:
        return self._has_match(clean_query, words, FOLLOW_UP_KEYWORDS)

    def _is_explanation(self, clean_query: str, words: Set[str]) -> bool:
        return self._has_match(clean_query, words, EXPLANATION_KEYWORDS)



    def _has_match(self, clean_query: str, words: Set[str], keywords: Set[str]) -> bool:
        """
        Routes the matching strategy dynamically: 
        $O(1)$ set lookup for single words, substring matching for multi-word phrases.
        """
        for kw in keywords:
            if " " in kw:
                if self._contains_phrase(clean_query, kw):
                    return True
            else:
                if self._contains_keyword(words, kw):
                    return True
        return False

    def _contains_keyword(self, words: Set[str], keyword: str) -> bool:
        """Exact match for single words."""
        return keyword in words

    def _contains_phrase(self, clean_query: str, phrase: str) -> bool:
        """Substring match for multi-word phrases."""
        return phrase in clean_query
from abc import ABC, abstractmethod
from typing import List, Dict
from retrieval.models import AnalyzedQuery


class BaseQueryRewriter(ABC):
    """
    Abstract contract for query expansion and rewriting.
    Must return a *new* instance of AnalyzedQuery to maintain pipeline immutability.
    """
    @abstractmethod
    def rewrite(
        self, 
        analyzed_query: AnalyzedQuery, 
        chat_history: List[Dict[str, str]],
        query_id: str = "unknown_query"
    ) -> AnalyzedQuery:
        raise NotImplementedError
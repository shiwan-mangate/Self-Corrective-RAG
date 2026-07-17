## retrieval/analyzers/base.py
from abc import ABC, abstractmethod
from retrieval.models import SearchQuery, AnalyzedQuery

class BaseQueryAnalyzer(ABC):
    """
    Abstract contract for all query analyzers.

    Implementations are responsible only for understanding the user's query
    and producing a standardized AnalyzedQuery.

    They must not rewrite queries, generate embeddings, perform vector search,
    rerank results, call LLMs, or interact with the database.
    """

    @abstractmethod
    def analyze(self, search_query: SearchQuery) -> AnalyzedQuery:
        raise NotImplementedError
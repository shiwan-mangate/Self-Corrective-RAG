# retrieval/models.py
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field



class SearchType(str, Enum):
    SIMILARITY = "similarity"
    MMR = "mmr"
    HYBRID = "hybrid"


class QueryIntent(str, Enum):
    FACTUAL = "factual"
    COMPARISON = "comparison"
    SUMMARY = "summary"
    FOLLOW_UP = "follow_up"
    EXPLANATION = "explanation"
    UNKNOWN = "unknown"




class MetadataFilter(BaseModel):
    """Explicit contract for filtering vector searches."""
    language: Optional[str] = None
    source: Optional[str] = None
    document_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class Citation(BaseModel):
    """A first-class representation of a source used in generation."""
    chunk_id: str = Field(...)
    document_id: str = Field(...)
    title: Optional[str] = Field(default=None)
    source: Optional[str] = Field(default=None)  
    page: Optional[int] = Field(default=None)
    score: float = Field(..., description="The final relevance score of this chunk.")


class RetrievalStatistics(BaseModel):
    """Telemetry tracking for the retrieval pipeline's performance."""
    total_chunks_retrieved: int = 0
    filtered_chunks: int = 0
    reranked_chunks: int = 0
    context_tokens: int = 0
    duplicates_removed: int = 0
    average_similarity: float = 0.0


class RetrievalMetadata(BaseModel):
    """Consolidated metadata wrapper for the final context."""
    search_strategy: SearchType = Field(...)
    latency_ms: float = Field(default=0.0)
    statistics: RetrievalStatistics = Field(default_factory=RetrievalStatistics)





class SearchQuery(BaseModel):
    """
    Equivalent to RawDocument. 
    The raw incoming request from the user. Ignorant of search mechanics.
    """
    query_id: str = Field(..., description="Unique trace ID for distributed logging.")
    query: str = Field(..., description="The exact question asked by the user.")
    chat_history: List[Dict[str, str]] = Field(default_factory=list)
    top_k: int = Field(default=5)
    filters: MetadataFilter = Field(default_factory=MetadataFilter)


class AnalyzedQuery(BaseModel):
    original_query: str = Field(...)
    normalized_query: str = Field(...)
    rewritten_query: Optional[str] = Field(default=None)
    rewrite_performed: bool = Field(default=False)  # Explicit telemetry flag
    
    intent: QueryIntent = Field(default=QueryIntent.UNKNOWN)
    needs_history: bool = Field(...)
    needs_rewrite: bool = Field(...)
    search_type: SearchType = Field(...)
    top_k: int = Field(...)
    filters: MetadataFilter = Field(default_factory=MetadataFilter)
    entities: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    """
    The payload returned by pgvector before application-level reranking.
    """
    chunk_id: str = Field(...)
    document_id: str = Field(...)
    source: Optional[str] = Field(default=None, description="e.g., pdf, url, docx")
    document_title: Optional[str] = Field(default=None)
    text: str = Field(...)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    similarity_score: float = Field(..., description="The native vector distance/similarity score.")
    token_count: int = Field(..., description="Pre-calculated token cost of this chunk.")
    checksum: str = Field(..., description="SHA-256 hash for O(1) deduplication.")

class RankedChunk(RetrievedChunk):
    """
    A chunk that has been evaluated, filtered, and mathematically ranked.
    """
    retrieval_score: float = Field(..., description="The original score from the database.")
    rerank_score: Optional[float] = Field(default=None, description="Score from a CrossEncoder or Cohere reranker.")
    final_score: float = Field(...)
    rank: int = Field(..., description="Final position in the context window (1 = best).")

class RetrievalContext(BaseModel):
    """
    The final, packaged object handed off to the Generation Subsystem.
    """
    question: str = Field(...)
    rewritten_question: Optional[str] = Field(default=None)
    context: str = Field(..., description="The fully assembled, prompt-ready context string.")
    chunks: List[RankedChunk] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    metadata: RetrievalMetadata = Field(...)
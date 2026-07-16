# schemas/retrieval.py

from typing import Optional
from pydantic import BaseModel, Field


class CitationResponse(BaseModel):
    """
    Public representation of a specific chunk of evidence used by the LLM.
    Provides traceability from a generated claim back to the vector database.
    """
    inline_reference: str = Field(
        ..., 
        description="The marker used in the answer text (e.g., '[1]') that maps to this citation."
    )
    chunk_id: str = Field(
        ..., 
        description="The unique identifier of the specific text chunk used as evidence."
    )
    document_id: str = Field(
        ..., 
        description="The unique identifier of the parent document."
    )
    document_name: Optional[str] = Field(
        default=None, 
        description="A human-readable title or filename of the document."
    )
    source_type: Optional[str] = Field(
        default=None, 
        description="The format or origin of the source (e.g., 'pdf', 'url', 'docx')."
    )
    page_number: Optional[int] = Field(
        default=None, 
        description="The page number where the chunk was found, if applicable."
    )
    relevance_score: float = Field(
        ..., 
        description="The final mathematical relevance score assigned by the retrieval subsystem."
    )


class SourceResponse(BaseModel):
    """
    A deduplicated, high-level summary of a document used in the answer.
    Used by the frontend to render a clean 'Sources' list without repeating chunks.
    """
    document_id: str = Field(
        ..., 
        description="The unique identifier of the source document."
    )
    document_name: Optional[str] = Field(
        default=None, 
        description="A human-readable title or filename."
    )
    source_type: Optional[str] = Field(
        default=None, 
        description="The format or origin of the source (e.g., 'pdf', 'url')."
    )


class RetrievalMetadataResponse(BaseModel):
    """
    Public telemetry for the retrieval phase.
    Exposes system performance and context-budget metrics without leaking internal logic.
    """
    search_strategy: str = Field(
        ..., 
        description="The search algorithm used (e.g., 'similarity', 'hybrid', 'mmr')."
    )
    chunks_retrieved: int = Field(
        ..., 
        description="The raw number of candidate chunks retrieved before filtering and reranking."
    )
    chunks_used: int = Field(
        ..., 
        description="The final number of chunks that survived filtering/budgeting and were sent to the LLM."
    )
    latency_ms: float = Field(
        ..., 
        description="The wall-clock time taken to complete the retrieval phase in milliseconds."
    )
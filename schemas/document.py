# schemas/document.py

from typing import List
from pydantic import BaseModel, Field, field_validator


class DocumentIngestionRequest(BaseModel):
    """
    The external API contract for triggering a document ingestion pipeline.
    Accepts a local file path or a URL and acts as the entry boundary to Layer 1.
    """
    
    source: str = Field(
        ..., 
        min_length=1,
        description="The file path or URL to load, parse, chunk, and embed.",
        json_schema_extra={"example": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"}
    )

    @field_validator('source')
    @classmethod
    def validate_source_not_empty(cls, v: str) -> str:
        """
        Ensures the source string is not just whitespace.
        """
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Source cannot be empty or solely whitespace.")
        return cleaned


class DocumentIngestionResponse(BaseModel):
    """
    Public telemetry representing the outcome of an ingestion process.
    Strictly hides the raw embeddings and internal chunk text from the frontend.
    """
    
    documents_processed: int = Field(
        ..., 
        ge=0,
        description="The total number of documents successfully parsed and processed."
    )
    
    chunks_generated: int = Field(
        ..., 
        ge=0,
        description="The total number of semantic chunks yielded by the chunking strategy."
    )
    
    chunks_persisted: int = Field(
        ..., 
        ge=0,
        description="The number of vectorized chunks successfully committed to the database."
    )
    
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal anomalies detected during ingestion (e.g., 'No chunks generated for source')."
    )
    
    elapsed_time_sec: float = Field(
        ..., 
        ge=0.0,
        description="The wall-clock execution time of the entire ingestion pipeline in seconds."
    )
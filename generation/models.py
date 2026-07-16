from typing import List, Optional, Dict
from pydantic import BaseModel, Field

from retrieval.models import QueryIntent, Citation, RetrievalMetadata


class GenerationRequest(BaseModel):
    """
    The strict input boundary for the Generation Subsystem.
    Decouples text synthesis from how the context was retrieved.
    """
    query_id: str = Field(..., description="Unique trace ID for distributed logging.")
    original_query: str = Field(..., description="The raw question asked by the user.")
    optimized_query: str = Field(..., description="The LLM-rewritten or normalized query used to fetch the context.")
    context: str = Field(..., description="The fully assembled, budgeted markdown evidence string.")
    intent: QueryIntent = Field(default=QueryIntent.UNKNOWN, description="Guides the Prompt Builder on which instructional template to use.")
    available_citations: List[Citation] = Field(default_factory=list,description="The pool of valid sources the LLM is authorized to cite.")
    retrieval_metadata: RetrievalMetadata = Field(..., description="Carried forward so Evaluation has complete visibility into the search phase.")
    chat_history: List[Dict[str, str]] = Field(default_factory=list, description="Recent conversation turns for contextual tone.")

class TokenUsage(BaseModel):
    """
    Standardized telemetry for LLM token consumption.
    """
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)


class GeneratedAnswer(BaseModel):
    """
    The raw output from the AI Text Generation Service.
    Before post-processing, citation extraction, or final formatting.
    """
    answer: str = Field(
        ..., 
        description="The raw text string returned by the LLM."
    )
    generation_time_ms: float = Field(..., description="Latency of the LLM API call.")
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    finish_reason: Optional[str] = Field(
        default=None, 
        description="Why the model stopped (e.g., 'stop', 'length')."
    )


class GenerationCitation(Citation):
    """
    A specific, verified citation actively used in the final answer.
    Inherits (chunk_id, document_id, title, source, page, score) from the base Citation model 
    so no retrieval telemetry is lost, while adding Generation-specific markers.
    """
    inline_reference: str = Field(
        ..., 
        description="The inline marker used by the LLM in the text, e.g., '[1]' or '[doc_3]'."
    )


class GenerationMetadata(BaseModel):
    """
    Consolidated telemetry wrapper for the Generation phase.
    Maintains perfect architectural symmetry with RetrievalMetadata.
    """
    latency_ms: float = Field(...)
    token_usage: TokenUsage = Field(...)
    finish_reason: Optional[str] = Field(default=None)
    model_name: str = Field(..., description="The specific LLM used for generation (e.g., 'llama3-70b-8192').")
    temperature: float = Field(...)
    template_name: str = Field(..., description="The name of the prompt template used.") # Added!


class GenerationResponse(BaseModel):
    """
    The final, public output of the Generation Subsystem.
    This comprehensive payload is handed off to the Evaluation Subsystem, 
    the API router, or the end user.
    """
    answer: str = Field(
        ..., 
        description="The cleaned, formatted final answer string."
    )
    citations: List[GenerationCitation] = Field(
        default_factory=list, 
        description="Explicit, validated sources actively referenced in the generated answer."
    )
    documents_used: List[str] = Field(
        default_factory=list, 
        description="A flat, unique list of document names/sources for easy UI rendering."
    )
    
    
    generation_metadata: GenerationMetadata = Field(
        ..., 
        description="Performance and usage metrics from the LLM execution."
    )
    retrieval_metadata: RetrievalMetadata = Field(
        ..., 
        description="Passthrough of search performance metrics."
    )


class PromptPackage(BaseModel):
    """
    The strict output contract of the Prompt Builder.
    Carries the finalized strings for the LLM execution, plus metadata for downstream observability.
    """
    query_id: str = Field(default="unknown_query", description="Passed forward for distributed tracing.")
    system_prompt: str = Field(...)
    user_prompt: str = Field(...)
    template_name: str = Field(..., description="The name of the template used (e.g., 'summary', 'qa').")
    intent: QueryIntent = Field(..., description="Passed along for logging and monitoring in the Generator.")
    citation_map: Dict[int, Citation] = Field(default_factory=dict)
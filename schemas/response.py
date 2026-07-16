# schemas/response.py

from typing import List
from pydantic import BaseModel, Field

from graph.models import ResponseStatus
from schemas.retrieval import CitationResponse


class ChatResponse(BaseModel):
    """
    Public API response for a completed Self-Healing RAG workflow.
    Exposes the final generated answer, the evidence used, and high-level 
    telemetry about the self-healing recovery process.
    """

    query_id: str = Field(
        ...,
        description="Unique trace identifier for this workflow execution."
    )

    session_id: str = Field(
        ...,
        description="Active conversation session identifier."
    )

    status: ResponseStatus = Field(
        ...,
        description="Final workflow status (e.g., SUCCESS, PARTIAL, FAILED)."
    )

    answer: str = Field(
        ...,
        min_length=1,
        description="Final answer returned by the workflow."
    )

    citations: List[CitationResponse] = Field(
        default_factory=list,
        description="Validated evidence sources referenced by the answer."
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Final aggregated answer confidence."
    )

    recovery_used: bool = Field(
        default=False,
        description="Whether the self-healing subsystem participated in the workflow."
    )

    correction_path: List[str] = Field(
        default_factory=list,
        description="Ordered self-healing actions executed during recovery."
    )

    retry_count: int = Field(
        default=0,
        ge=0,
        description="Number of recovery retries executed."
    )

    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Total LangGraph workflow latency in milliseconds."
    )

    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal operational warnings."
    )
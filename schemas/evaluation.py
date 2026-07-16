# schemas/evaluation.py

from typing import Optional
from pydantic import BaseModel, Field


class GroundingResponse(BaseModel):
    """
    Public representation of the grounding evaluation.
    Indicates whether the generated answer is supported by the retrieved context.
    """
    is_grounded: bool = Field(
        ...,
        description="Whether the answer is supported by retrieved evidence."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Judge confidence in the grounding decision."
    )


class HallucinationResponse(BaseModel):
    """
    Public representation of the hallucination evaluation.
    Flags whether the LLM fabricated claims not present in the evidence.
    """
    detected: bool = Field(
        ...,
        description="Whether unsupported fabricated claims were detected."
    )
    risk: str = Field(
        ...,
        description="Categorical hallucination risk level."
    )


class ConfidenceResponse(BaseModel):
    """
    Public representation of the system's overall confidence in the generated answer.
    """
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Final aggregated system confidence."
    )
    retrieval_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score derived from retrieval metrics."
    )
    grounding_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score derived from the grounding evaluation."
    )
    hallucination_risk: str = Field(
        ...,
        description="Categorical indicator of hallucination severity extracted from the evaluation run."
    )


class RagasResponse(BaseModel):
    """
    Public representation of RAGAS benchmarking metrics.
    Features a strict separation between Live metrics (always calculated) 
    and Offline Benchmark metrics (calculated only against ground-truth datasets).
    """
    faithfulness: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Measures if the claims in the answer are supported by the retrieved context. (Live)"
    )
    answer_relevancy: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Measures how well the answer addresses the user's initial question. (Live)"
    )
    context_recall: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Measures if all required ground-truth information was retrieved. (Benchmark only)"
    )
    context_precision: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Measures the density of relevant vs irrelevant chunks retrieved. (Benchmark only)"
    )


class EvaluationResponse(BaseModel):
    """
    The unified public contract for all answer-quality metrics.
    Used by the frontend to render health indicators and trust scores for the user.
    """
    grounding: GroundingResponse = Field(
        ...,
        description="The outcome of the context-support verification."
    )
    hallucination: HallucinationResponse = Field(
        ...,
        description="The outcome of the fabrication detection."
    )
    confidence: ConfidenceResponse = Field(
        ...,
        description="The overall system trust in the generated payload."
    )
    ragas: RagasResponse = Field(
        ...,
        description="Standardized RAGAS quality metrics."
    )
    retry_recommended: bool = Field(
        ...,
        description="Whether the evaluator recommended a recovery action."
    )
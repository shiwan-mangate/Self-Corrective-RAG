from enum import Enum
from typing import List, Optional, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field

# Domain boundaries
from retrieval.models import RetrievalMetadata
from generation.models import GenerationMetadata, GenerationCitation


class HallucinationRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EvaluationDecision(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NEEDS_REVIEW = "needs_review"


class EvaluationMode(str, Enum):
    STRICT = "strict"        # High thresholds, uses most capable LLM
    PERMISSIVE = "permissive" # Lower thresholds, allows minor unverified claims
    FAST = "fast"            # Uses smaller LLM for speed


# ==========================================
# Input Contract
# ==========================================

class EvaluationRequest(BaseModel):
    """
    The strict input boundary for the Evaluation Subsystem.
    """
    query_id: str = Field(
        ...,
        description="Unique identifier for this evaluation request to enable distributed tracing."
    )
    original_query: str = Field(..., description="The raw question asked by the user.")
    optimized_query: str = Field(..., description="The query used to retrieve context.")
    context: str = Field(..., description="The exact markdown evidence string provided to the LLM.")
    answer: str = Field(..., description="The final generated answer to evaluate.")
    citations: List[GenerationCitation] = Field(default_factory=list)
    
    retrieval_metadata: RetrievalMetadata = Field(...)
    generation_metadata: GenerationMetadata = Field(...)
    
    retrieved_chunks: List[str] = Field(
        default_factory=list, 
        description="The raw, individual chunk texts required for accurate offline RAGAS context metrics."
    )


# ==========================================
# Component Results
# ==========================================

class GroundingResult(BaseModel):
    is_grounded: bool = Field(...)
    supported_claims: List[str] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)
    confidence: float = Field(...)
    explanation: str = Field(...)


class HallucinationResult(BaseModel):
    has_hallucination: bool = Field(...)
    hallucinated_claims: List[str] = Field(default_factory=list)
    reasoning: str = Field(...)
    risk: HallucinationRisk = Field(...)


class RagasResult(BaseModel):
    faithfulness_score: float = Field(default=0.0)
    answer_relevancy_score: float = Field(default=0.0)
    context_precision_score: Optional[float] = Field(default=None)
    context_recall_score: Optional[float] = Field(default=None)
    latency_ms: Optional[float] = Field(default=None)


class ConfidenceResult(BaseModel):
    overall_score: float = Field(...)
    retrieval_confidence: float = Field(...)
    grounding_confidence: float = Field(...)
    hallucination_risk: HallucinationRisk = Field(...)
    explanation: str = Field(...)


# ==========================================
# Metadata & Output Contract
# ==========================================

class EvaluationMetadata(BaseModel):
    """
    Observability and execution telemetry for the Evaluation phase.
    """
    latency_ms: float = Field(...)
    evaluation_mode: str = Field(..., description="The mode executed (e.g., live, benchmark, strict).")
    judge_model_name: str = Field(..., description="The LLM used to judge the response.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = Field(default="1.0", description="Version of the evaluation pipeline/prompts.")


class RetryRecommendation(str, Enum):
    EXPAND_SEARCH = "expand_search"
    REWRITE_QUERY = "rewrite_query"
    WEB_SEARCH = "web_search"
    ASK_CLARIFICATION = "ask_clarification"
    LOWER_TEMPERATURE = "lower_temperature"
    STRICT_GROUNDING = "strict_grounding"


class EvaluationReport(BaseModel):
    """
    The final, public output of the Evaluation Subsystem.
    """
    decision: EvaluationDecision = Field(...)
    
    grounding: GroundingResult = Field(...)
    hallucination: HallucinationResult = Field(...)
    ragas: RagasResult = Field(...)
    confidence: ConfidenceResult = Field(...)
    
    evaluation_metadata: EvaluationMetadata = Field(...)
    
    warnings: List[str] = Field(default_factory=list)
    retry_recommendation: Optional[RetryRecommendation] = Field(default=None)
from sqlalchemy import (
    Column,
    BigInteger,
    String,
    Text,
    Boolean,
    Double,
    Integer,
    DateTime,
    text
)
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB
from database.models.base import Base

class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    query_id = Column(String(100), nullable=False)
    evaluation_timestamp = Column(
        DateTime(timezone=True), 
        nullable=False, 
        server_default=func.now()
    )


    original_query = Column(Text, nullable=False)
    optimized_query = Column(Text, nullable=False)
    generated_answer = Column(Text, nullable=False)

 
    is_grounded = Column(Boolean, nullable=False)
    supported_claims = Column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    unsupported_claims = Column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    grounding_reason = Column(Text, nullable=False)

 
    has_hallucination = Column(Boolean, nullable=False)
    hallucinated_claims = Column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    hallucination_reason = Column(Text, nullable=False)

    overall_confidence = Column(Double, nullable=False)
    retrieval_confidence = Column(Double, nullable=False)
    grounding_confidence = Column(Double, nullable=False)
    hallucination_risk = Column(String(20), nullable=False)
    confidence_reason = Column(Text, nullable=False)


    faithfulness = Column(Double, nullable=False)
    answer_relevancy = Column(Double, nullable=False)
    context_precision = Column(Double, nullable=True)
    context_recall = Column(Double, nullable=True)
    ragas_latency_ms = Column(Double, nullable=True)


    decision = Column(String(30), nullable=False)
    retry_recommendation = Column(String(50), nullable=True)
    warnings = Column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))

    retrieval_latency_ms = Column(Double, nullable=False)
    generation_latency_ms = Column(Double, nullable=False)
    evaluation_latency_ms = Column(Double, nullable=False)
    total_tokens = Column(Integer, nullable=False)
    generation_model_name = Column(String(100), nullable=False)
    judge_model_name = Column(String(100), nullable=False)
    evaluation_mode = Column(String(20), nullable=False)
    evaluation_version = Column(String(20), nullable=False)
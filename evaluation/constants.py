"""
Global Configuration and Policies for the Evaluation Subsystem.
Centralizes all LLM Judge behaviors, scoring weights, thresholds, and limits.
"""
from evaluation.models import EvaluationMode

# ==========================================
# LLM Judge Defaults
# ==========================================
DEFAULT_JUDGE_MODEL = "openai/gpt-oss-120b" 
DEFAULT_JUDGE_TEMPERATURE = 0.0  
DEFAULT_EVALUATION_MODE = EvaluationMode.STRICT
MAX_EVALUATION_TIME_MS = 15000

# ==========================================
# Pass/Fail Thresholds (Hard Limits)
# ==========================================
# Scores below these numbers trigger an automatic FAIL or NEEDS_REVIEW.
MIN_FAITHFULNESS_SCORE = 0.85
MIN_ANSWER_RELEVANCY_SCORE = 0.80
MIN_GROUNDING_CONFIDENCE = 0.90
MIN_OVERALL_CONFIDENCE_SCORE = 0.85
MIN_REVIEW_CONFIDENCE_SCORE = 0.75

# Strict RAG: zero tolerance for unsupported claims.
MAX_UNSUPPORTED_CLAIMS_ALLOWED = 0  


# ==========================================
# Warning Thresholds (Borderline Limits)
# ==========================================
# Scores between the MIN threshold and these values trigger a non-fatal Warning.
WARNING_FAITHFULNESS_SCORE = 0.90
WARNING_ANSWER_RELEVANCY_SCORE = 0.85
WARNING_GROUNDING_CONFIDENCE = 0.95


# ==========================================
# Hallucination Risk Mapping
# ==========================================
# Maps a raw hallucination severity score (0.0 - 1.0) to the HallucinationRisk Enum.
LOW_HALLUCINATION_THRESHOLD = 0.10     # <= 0.10 is LOW risk
MEDIUM_HALLUCINATION_THRESHOLD = 0.40  # > 0.10 and <= 0.40 is MEDIUM risk
                                       # > 0.40 is HIGH risk

HIGH_HALLUCINATION_PENALTY = 0.20
MEDIUM_HALLUCINATION_PENALTY = 0.80
# ==========================================
# Confidence Weighting Formula
# ==========================================
# How the ConfidenceScorer calculates the `overall_score`. Must sum to 1.0.
# In a grounded RAG, Grounding and Faithfulness are the most critical signals.
WEIGHT_GROUNDING = 0.40
WEIGHT_FAITHFULNESS = 0.35
WEIGHT_ANSWER_RELEVANCY = 0.15
WEIGHT_RETRIEVAL = 0.10


# ==========================================
# Evaluation Run Modes
# ==========================================
# LIVE: Optimized for speed during an active user request.
LIVE_EVALUATION_METRICS = [
    "faithfulness", 
    "answer_relevancy"
]

# BENCHMARK: Optimized for offline nightly runs. Includes context metrics.
BENCHMARK_EVALUATION_METRICS = [
    "faithfulness", 
    "answer_relevancy", 
    "context_precision", 
    "context_recall"
]


# ==========================================
# Observability & Logging
# ==========================================
# Prevents massive markdown contexts from flooding standard output logs.
LOG_EVALUATION_PREVIEW_CHARS = 250
MIN_GOOD_RETRIEVAL_SIMILARITY = 0.80
# shared/exceptions.py

class SelfHealingRAGError(Exception):
    """Base exception class for all custom errors in the Self-Healing RAG project."""
    pass


# ==========================================
# Memory Exceptions
# ==========================================
class SessionNotFoundError(SelfHealingRAGError):
    """Raised when an active session is required but missing, and auto-create is disabled."""
    pass


# ==========================================
# Generation Exceptions
# ==========================================
class GenerationExecutionError(SelfHealingRAGError):
    """Raised when the LLM text synthesis pipeline fails or encounters an API error."""
    pass


# ==========================================
# Evaluation Exceptions
# ==========================================
class EvaluationExecutionError(SelfHealingRAGError):
    """Raised when a third-party evaluation framework (like RAGAS) fails to execute."""
    pass
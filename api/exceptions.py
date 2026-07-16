# api/exceptions.py

from shared.exceptions import SelfHealingRAGError

class EvaluationNotFoundError(SelfHealingRAGError):
    """
    Raised when an evaluation result cannot be found in the database.
    Mapped to HTTP 404 in the exception handler.
    """
    pass
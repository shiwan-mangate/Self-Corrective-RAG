# database/repositories/evaluation_repository.py
from typing import Optional
from sqlalchemy.orm import Session
from database.models.evaluation_run import EvaluationRun

class EvaluationRepository:
    """
    The exclusive Read-Side gateway for the Evaluation Subsystem.
    Retrieves persisted telemetry without triggering LLM Judges or RAGAS.
    """
    def __init__(self, session: Session):
        self.session = session

    def get_latest_by_query_id(self, query_id: str) -> Optional[EvaluationRun]:
        """
        Retrieves the most recent evaluation run for a given query_id.
        """
        return (
            self.session.query(EvaluationRun)
            .filter(EvaluationRun.query_id == query_id)
            .order_by(EvaluationRun.evaluation_timestamp.desc())
            .first()
        )
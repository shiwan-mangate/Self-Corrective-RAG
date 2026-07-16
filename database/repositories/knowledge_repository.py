import logging
from typing import List, Optional
from datetime import datetime, timezone

from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from database.models.knowledge_gap import KnowledgeGapModel
from self_healing.models import KnowledgeGap

logger = logging.getLogger(__name__)

class PostgresKnowledgeStorage:
    """
    Concrete implementation of KnowledgeStorage.
    Translates Pydantic KnowledgeGaps into SQLAlchemy ORM models and persists
    them to PostgreSQL. Encapsulates all session and transaction logic.
    Assumes all strings (like missing_topic) are already normalized by the Manager.
    """

    def __init__(self, session_factory: sessionmaker):
        self.session_factory = session_factory

    def get_by_topic(self, missing_topic: str) -> Optional[KnowledgeGap]:
        """Fetches a gap by its exact topic and converts it back to Pydantic."""
        with self.session_factory() as session:
            orm_model = session.query(KnowledgeGapModel).filter(
                KnowledgeGapModel.missing_topic == missing_topic
            ).first()
            
            if not orm_model:
                return None
                
            return self._to_pydantic(orm_model)

    def save(self, gap: KnowledgeGap) -> None:
        """
        Upsert operation. If the topic exists, updates the row. 
        If not, inserts a new row.
        """
        with self.session_factory() as session:
            try:
                orm_model = session.query(KnowledgeGapModel).filter(
                    KnowledgeGapModel.missing_topic == gap.missing_topic
                ).first()
                
                if orm_model:
                    
                    orm_model.failed_queries = gap.failed_queries
                    orm_model.frequency = gap.frequency
                    orm_model.resolved = gap.resolved
                    orm_model.last_detected = gap.last_detected
                    orm_model.last_query_id = gap.last_query_id
                    orm_model.resolved_at = gap.resolved_at
                else:
                   
                    orm_model = KnowledgeGapModel(
                        missing_topic=gap.missing_topic,
                        failed_queries=gap.failed_queries,
                        frequency=gap.frequency,
                        resolved=gap.resolved,
                        first_detected=gap.first_detected,
                        last_detected=gap.last_detected,
                        last_query_id=gap.last_query_id,
                        resolved_at=gap.resolved_at
                    )
                    session.add(orm_model)
                    
                session.commit()
                logger.info(f"Knowledge Gap persisted | Topic='{gap.missing_topic}'")
                
            except SQLAlchemyError:
                session.rollback()
                logger.exception(f"Database error while saving Knowledge Gap '{gap.missing_topic}'")
                raise

    def get_all(self, include_resolved: bool = False) -> List[KnowledgeGap]:
        """
        Returns gaps, explicitly ordering by frequency and recency.
        Lets the database handle the sorting instead of Python.
        """
        with self.session_factory() as session:
            query = session.query(KnowledgeGapModel)
            
            if not include_resolved:
                query = query.filter(KnowledgeGapModel.resolved.is_(False))
                
            query = query.order_by(
                KnowledgeGapModel.frequency.desc(),
                KnowledgeGapModel.last_detected.desc()
            )
                
            orm_results = query.all()
            return [self._to_pydantic(row) for row in orm_results]

    
    def get_top_missing_topics(self, limit: int) -> List[KnowledgeGap]:
        """Uses SQL to efficiently fetch and sort only the top N unresolved gaps."""
        with self.session_factory() as session:
            orm_results = session.query(KnowledgeGapModel)\
                .filter(KnowledgeGapModel.resolved.is_(False))\
                .order_by(KnowledgeGapModel.frequency.desc(), KnowledgeGapModel.last_detected.desc())\
                .limit(limit)\
                .all()
            return [self._to_pydantic(row) for row in orm_results]

    def get_resolved_topics(self) -> List[KnowledgeGap]:
        """Uses SQL to fetch only resolved gaps."""
        with self.session_factory() as session:
            orm_results = session.query(KnowledgeGapModel)\
                .filter(KnowledgeGapModel.resolved.is_(True))\
                .order_by(KnowledgeGapModel.resolved_at.desc())\
                .all()
            return [self._to_pydantic(row) for row in orm_results]
    

    def delete_expired(self, expiry_days: int) -> int:
        """
        Bulk deletion for cleanup jobs. 
        Deletes UNRESOLVED gaps that haven't been seen in `expiry_days`.
        """
        from datetime import timedelta
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=expiry_days)
        
        with self.session_factory() as session:
            try:
                deleted_count = session.query(KnowledgeGapModel).filter(
                    KnowledgeGapModel.resolved.is_(False),
                    KnowledgeGapModel.last_detected < cutoff_date
                ).delete(synchronize_session=False)
                
                session.commit()
                return deleted_count
                
            except SQLAlchemyError:
                session.rollback()
                logger.exception("Database error during expired gap cleanup")
                return 0

    def _to_pydantic(self, orm_model: KnowledgeGapModel) -> KnowledgeGap:
        """
        Safe translation mapping. Ensures the business logic never sees 
        a detached SQLAlchemy instance.
        """
        def ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
            if dt and dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt

        return KnowledgeGap(
            missing_topic=orm_model.missing_topic,
            failed_queries=list(orm_model.failed_queries), 
            frequency=orm_model.frequency,
            resolved=orm_model.resolved,
            first_detected=ensure_utc(orm_model.first_detected),
            last_detected=ensure_utc(orm_model.last_detected),
            last_query_id=orm_model.last_query_id,
            resolved_at=ensure_utc(orm_model.resolved_at)
        )
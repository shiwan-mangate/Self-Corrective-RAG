from sqlalchemy import Column, BigInteger, String, Integer, Boolean, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy import Index

from database.models.base import Base


class KnowledgeGapModel(Base):
    """
    SQLAlchemy ORM model mapping to the `knowledge_gaps` table.
    Acts as the Long-Term Memory state table for the Self-Healing Subsystem.
    Tracks missing knowledge over time to trigger automated learning (ingestion).
    """
    __tablename__ = "knowledge_gaps"

    id = Column(
        BigInteger, 
        primary_key=True, 
        autoincrement=True,
        comment="Internal database identifier."
    )
    
    missing_topic = Column(
        String(300), 
        unique=True, 
        nullable=False,
        comment="The canonical subject missing from the database (e.g., 'GPT-6')."
    )


    failed_queries = Column(
        JSONB, 
        nullable=False, 
        server_default='[]',
        comment="List of specific user phrasings that hit this gap."
    )
    
    frequency = Column(
        Integer, 
        nullable=False, 
        default=1,
        comment="Number of times this gap has been detected across all users."
    )


    resolved = Column(
        Boolean, 
        nullable=False, 
        default=False,
        comment="True if the IngestionPipeline has successfully learned this topic."
    )


    first_detected = Column(
        DateTime(timezone=True), 
        nullable=False, 
        server_default=func.now(),
        comment="When this topic was first discovered missing."
    )
    
    last_detected = Column(
        DateTime(timezone=True), 
        nullable=False, 
        server_default=func.now(),
        onupdate=func.now(),
        comment="Most recent occurrence. Automatically updates when touched."
    )
    
    last_query_id = Column(
        String(100), 
        nullable=True,
        comment="Foreign correlation ID bridging back to evaluation_runs and distributed traces."
    )
    
    resolved_at = Column(
        DateTime(timezone=True), 
        nullable=True,
        comment="Timestamp when the IngestionPipeline successfully resolved the gap."
    )
    __table_args__ = (
        Index("ix_knowledge_gaps_topic", "missing_topic"),
        Index("ix_knowledge_gaps_resolved", "resolved"),
        Index("ix_knowledge_gaps_freq_desc", "frequency", postgresql_using="btree"),
        Index("ix_knowledge_gaps_last_detected", "last_detected"),
    )

    def __repr__(self):
        return f"<KnowledgeGap(id={self.id}, topic='{self.missing_topic}', freq={self.frequency}, resolved={self.resolved})>"
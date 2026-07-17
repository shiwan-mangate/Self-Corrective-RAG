## database/models/vector.py
import uuid
from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from pgvector.sqlalchemy import Vector

from database.models.base import Base


class VectorChunkModel(Base):
    """
    SQLAlchemy ORM Model representing a vectorized text chunk in Neon Postgres.
    Strictly enforces 384 dimensions and server-side timestamps.
    """
    __tablename__ = "document_chunks"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chunk_id = Column(String, index=True, nullable=False, unique=True)
    document_id = Column(String, index=True, nullable=False)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(384), nullable=False)
    checksum = Column(String, nullable=False, index=True)
    metadata_ = Column("metadata", JSONB, default=dict, nullable=False)
    created_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        nullable=False
    )
    __table_args__ = (
        # Example for later: Index('ix_document_chunk_composite', 'document_id', 'chunk_id'),
    )

    def __repr__(self):
        return f"<VectorChunkModel(chunk_id='{self.chunk_id}', document_id='{self.document_id}')>"
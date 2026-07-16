import logging
import time
from typing import List, Optional, TypedDict, Any
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

from database.models.vector import VectorChunkModel

logger = logging.getLogger(__name__)


class VectorChunkRecord(TypedDict):
    """
    Explicit compilation contract for database storage.
    Enforces compilation type-safety at the compiler/IDE level.
    """
    chunk_id: str
    document_id: str
    text: str
    embedding: List[float]
    checksum: str
    metadata_: dict  # Binds directly to the model's metadata_ property


class VectorRepository:
    """
    The exclusive gatekeeper for SQL execution against document_chunks.
    Accepts only strict contract records. Does zero domain mapping.
    """
    def __init__(self, session: Session):
        self.session = session

    def bulk_upsert_chunk_records(self, record_mappings: List[VectorChunkRecord]) -> int:
        """
        Executes a single high-performance multi-valued bulk UPSERT.
        Sends the entire array to Neon in one network round-trip.
        """
        if not record_mappings:
            return 0

        start_time = time.time()
        doc_id = record_mappings[0]["document_id"] if record_mappings else "unknown"
        rows_attempted = len(record_mappings)

        try:
            # 1. Initialize the target insert statement
            stmt = insert(VectorChunkModel)
            
            # 2. Define the conflict-handling strategy for matching chunk_ids
            upsert_stmt = stmt.on_conflict_do_update(
                index_elements=[VectorChunkModel.chunk_id],
                set_={
                    "text": stmt.excluded.text,
                    "embedding": stmt.excluded.embedding,
                    "checksum": stmt.excluded.checksum,
                    "metadata": stmt.excluded.metadata,  # <--- Target DB column : Excluded payload
                }
            )

            # 3. Mass pass mappings directly into the single execution context
            self.session.execute(upsert_stmt, record_mappings)
            self.session.commit()
            
            return rows_attempted

        except Exception as e:
            self.session.rollback()
            elapsed_time = time.time() - start_time
            logger.error(
                f"DATABASE CRASH | Subsystem: VectorRepository | Doc: {doc_id} | "
                f"Rows Attempted: {rows_attempted} | Time Elapsed: {elapsed_time:.2f}s | Error: {str(e)}"
            )
            raise RuntimeError(f"Database bulk write operation aborted: {str(e)}")

    def delete_by_document_id(self, document_id: str) -> int:
        """Purges rows bound to a unique document identifier."""
        start_time = time.time()
        try:
            deleted_rows = (
                self.session.query(VectorChunkModel)
                .filter(VectorChunkModel.document_id == document_id)
                .delete(synchronize_session=False)
            )
            self.session.commit()
            return deleted_rows
        except Exception as e:
            self.session.rollback()
            elapsed_time = time.time() - start_time
            logger.error(
                f"DATABASE CRASH | Subsystem: VectorRepository | Operation: Delete | "
                f"Doc: {document_id} | Time Elapsed: {elapsed_time:.2f}s | Error: {str(e)}"
            )
            raise RuntimeError(f"Database deletion failed: {str(e)}")

    def get_record_by_chunk_id(self, chunk_id: str) -> Optional[VectorChunkModel]:
        """Fetches a specific database record row by its natural chunk_id index."""
        return (
            self.session.query(VectorChunkModel)
            .filter(VectorChunkModel.chunk_id == chunk_id)
            .first()
        )
    
    def has_any_records_for_document(self, document_id: str) -> bool:
        """Checks for the presence of a document without downloading heavy text payloads."""
        from database.models.vector import VectorChunkModel
        
        # Using .first() combined with an indexed column yields maximum performance
        record = (
            self.session.query(VectorChunkModel.id)
            .filter(VectorChunkModel.document_id == document_id)
            .first()
        )
        return record is not None
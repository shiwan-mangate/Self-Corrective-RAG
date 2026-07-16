import logging
import time
import uuid
from typing import List

# Domain Boundaries
from ingestion.models import EmbeddedChunk

# Infrastructure/Data Access Contracts
from database.repositories.vector_repository import VectorRepository, VectorChunkRecord

logger = logging.getLogger(__name__)


class VectorService:
    """
    The exclusive public-facing gateway for all Command-side (write) database operations.
    Fully decoupled from SQLAlchemy sessions via Repository Pattern injection, ensuring 
    that infrastructure-specific transaction frames never leak into the ingestion domain.
    """
    
    def __init__(self, repository: VectorRepository):
        """
        True Dependency Injection patterns keep this service engine-agnostic and 
        perfectly mockable during isolated unit-testing runs.
        """
        self.repository = repository

    def persist_embedded_chunks(self, embedded_chunks: List[EmbeddedChunk]) -> int:
        """
        Transforms incoming domain-level EmbeddedChunk models into typed infrastructure 
        records and delegates to the atomic repository layer for high-performance storage.
        """
        if not embedded_chunks:
            return 0

        start_time = time.time()
        doc_id = embedded_chunks[0].document_id
        
        # Symmetrical Domain-to-Infrastructure Record Mapping Transition
        record_mappings = [self._map_chunk_to_record(chunk) for chunk in embedded_chunks]

        # Execute high-velocity atomic bulk UPSERT down at the Neon/PostgreSQL wire level
        upserted_count = self.repository.bulk_upsert_chunk_records(record_mappings)
        
        elapsed_time = time.time() - start_time
        logger.info(
            f"VectorService Storage Success | Doc: {doc_id} | Chunks Saved: {upserted_count} | "
            f"Execution Time: {elapsed_time:.2f}s"
        )
        
        return upserted_count

    def replace_document(self, embedded_chunks: List[EmbeddedChunk]) -> int:
        """
        Enforces the system's core Persistence Policy: Idempotent Sync.
        Safely evaluates if document footprints exist, purges structural assets 
        completely if present, and commits clean text payloads to prevent duplicate drift.
        """
        if not embedded_chunks:
            return 0
            
        doc_id = embedded_chunks[0].document_id
        
        if self.document_exists(doc_id):
            logger.info(f"Idempotency triggered. Purging existing vectors for Document ID: {doc_id}.")
            self.purge_document_knowledge(doc_id)
            
        return self.persist_embedded_chunks(embedded_chunks)

    def document_exists(self, document_id: str) -> bool:
        """
        Idempotency Guardrail.
        Verifies document status using performance-optimized primary key scanning 
        to skip unnecessary network roundtrips and text payload downloads.
        """
        return self.repository.has_any_records_for_document(document_id)

    def purge_document_knowledge(self, document_id: str) -> int:
        """
        Completely purges structural vectorized records bound to a unique document identifier.
        """
        return self.repository.delete_by_document_id(document_id)

    def _map_chunk_to_record(self, chunk: EmbeddedChunk) -> VectorChunkRecord:
        """
        Private Mapping Helper.
        
        Transforms a Pydantic Domain Model block into a primitive infrastructure TypedDict.
        Unwraps structured elements (like source locations) directly into JSONB bounds 
        to optimize down-stream storage alignment.
        """
        db_metadata = chunk.metadata.copy() if chunk.metadata else {}
        db_metadata.update({
            "token_count": chunk.token_count,
            "start_order": chunk.start_order,
            "end_order": chunk.end_order,
            "start_location": chunk.start_location.model_dump(),
            "end_location": chunk.end_location.model_dump(),
        })

        return {
            "id": uuid.uuid4(),              # FIXED: Required for bulk inserts to bypass Python defaults
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "text": chunk.text,
            "embedding": chunk.embedding,
            "checksum": chunk.checksum,
            "metadata_": db_metadata         # FIXED: Maps to the DB column name, avoiding ORM proxy crash
        }
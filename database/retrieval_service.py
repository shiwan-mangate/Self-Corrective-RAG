## database/retrieval_service.py
import logging
import time
from typing import List, TypedDict

# Domain Context Boundaries
from retrieval.models import RetrievedChunk
from database.models.vector import VectorChunkModel

# Infrastructure Contract
from database.repositories.retrieval_repository import RetrievalRepository

logger = logging.getLogger(__name__)


class VectorSearchRecord(TypedDict):
    """
    Explicit read-side infrastructure type safety contract.
    Decouples raw SQLAlchemy database execution artifacts (like positional row tuples) 
    from the domain mapping layers by providing key-based semantic access.
    """
    model: VectorChunkModel
    distance: float


class RetrievalService:
    """
    The exclusive domain-infrastructure bridge for Query-side (read) database operations.
    
    Responsibility:
    Coordinates mathematical pgvector execution outputs through a granular, decoupled 
    mapping sequence to return pristine Pydantic domain models. Strictly enforces the 
    CQRS Read pattern by remaining completely stateless and execution-dumb.
    """
    
    def __init__(self, repository: RetrievalRepository):
        """
        Dependency Injection decouples this service from raw connection lifecycles,
        simplifying integration and testing.
        """
        self.repository = repository

    def search_by_similarity(
        self, 
        query_embedding: List[float], 
        top_k: int = 5
    ) -> List[RetrievedChunk]:
        """
        Executes a dense vector similarity search, measures transaction telemetry, 
        and routes the underlying datasets through the decoupled mapping engine.
        """
        start_time = time.time()
        strategy_label = "similarity"

        # 1. Fetch raw query execution details from the pgvector database layer
        raw_results = self.repository.similarity_search(
            query_embedding=query_embedding, 
            top_k=top_k
        )
        
        # 2. Normalize positional row data into strict, readable TypedDict structural records
        search_records: List[VectorSearchRecord] = [
            {"model": row[0], "distance": row[1]} for row in raw_results
        ]
        
        # 3. Hand records off to the decoupled structural mapping pipeline
        retrieved_chunks = self._map_results(search_records)
        
        # 4. Compile infrastructure telemetry matching the VectorService design
        elapsed_time = time.time() - start_time
        logger.info(
            f"RetrievalService Search Success | Strategy: {strategy_label} | "
            f"Top K: {top_k} | Returned: {len(retrieved_chunks)} | "
            f"Execution Time: {elapsed_time:.4f}s"
        )
        
        return retrieved_chunks

    def _map_results(self, records: List[VectorSearchRecord]) -> List[RetrievedChunk]:
        """
        Orchestrates bulk array translations. Ensures that any alternative search strategies 
        (e.g., MMR, hybrid keyword-vector pipelines) can completely reuse the same mapper.
        """
        return [self._map_record_to_chunk(record) for record in records]

    def _map_record_to_chunk(self, record: VectorSearchRecord) -> RetrievedChunk:
        model = record["model"]
        distance = record["distance"]

        similarity_score = self._compute_similarity(distance)
        db_metadata = model.metadata_ if model.metadata_ else {}

        return RetrievedChunk(
            chunk_id=model.chunk_id,
            document_id=model.document_id,
            text=model.text,
            metadata=db_metadata,
            source=db_metadata.get("source"),
            document_title=db_metadata.get("title") or db_metadata.get("document_title"),
            similarity_score=similarity_score,
            token_count=db_metadata.get("token_count", 0),  # Explicit token mapping
            checksum=model.checksum                         # Explicit checksum mapping
        )

    def _compute_similarity(self, distance: float) -> float:
        """
        Isolates mathematical distance-to-similarity conversions.
        
        Centralizing this operation guarantees that migrating from Cosine Distance 
        (1.0 - distance) to alternatives like Inner Product or L2 Squared will 
        never break downstream domain properties.
        """
        return 1.0 - distance
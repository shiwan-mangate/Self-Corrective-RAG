import logging
from typing import List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import select

from database.models.vector import VectorChunkModel

logger = logging.getLogger(__name__)


class RetrievalRepository:
    """
    The exclusive read-side gateway for the vector database (CQRS Read pattern).
    Responsible solely for executing mathematical search queries against pgvector.
    Strictly forbidden from executing INSERT, UPDATE, or DELETE operations.
    """
    def __init__(self, session: Session):
        self.session = session

    def similarity_search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5
    ) -> List[Tuple[VectorChunkModel, float]]:
        """
        Executes a native pgvector cosine distance search.
        
        Args:
            query_embedding: The vectorized user query.
            top_k: The maximum number of chunks to return.
            
        Returns:
            A list of tuples containing the infrastructure model and its distance score:
            [(VectorChunkModel, float), ...]
        """
        try:
            # We calculate distance natively in Postgres using pgvector's cosine_distance operator.
            # We label it so we can extract the exact distance score without recalculating it in Python.
            distance_col = VectorChunkModel.embedding.cosine_distance(query_embedding).label("distance")
            
            stmt = (
                select(VectorChunkModel, distance_col)
                .order_by(distance_col)  # Order by closest distance (smallest value first)
                .limit(top_k)
            )
            
            # Execute the statement and fetch all rows
            results = self.session.execute(stmt).all()
            
            # SQLAlchemy returns Row objects; unpack them into clean tuples
            return [(row[0], row.distance) for row in results]
            
        except Exception as e:
            logger.exception("Catastrophic failure during database similarity search.")
            raise RuntimeError(f"RetrievalRepository search aborted: {str(e)}")
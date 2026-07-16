import logging
import time
from typing import List
from ingestion.models import Chunk, EmbeddedChunk
from ai.embedding_service import AIEmbeddingService

logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    """
    The ingestion-specific wrapper for embedding generation.
    Maps pure text Chunks to EmbeddedChunks using the shared AI service.
    """
    def __init__(self, embedding_service: AIEmbeddingService, batch_size: int = 32):
        self.service = embedding_service
        self.batch_size = batch_size

    def embed(self, chunks: List[Chunk]) -> List[EmbeddedChunk]:
        if not chunks:
            return []

        start_time = time.time()
        texts = [chunk.text for chunk in chunks]
        vectors_np = self.service.embed_texts(texts, batch_size=self.batch_size)
        if len(vectors_np) != len(chunks):
            raise RuntimeError(
                f"Catastrophic mismatch in embedding pipeline. "
                f"Chunks: {len(chunks)}, Vectors: {len(vectors_np)}"
            )

       
        embedded_chunks: List[EmbeddedChunk] = []
        for chunk, vector_np in zip(chunks, vectors_np):
           
            embedded_chunks.append(self._build_embedded_chunk(chunk, vector_np.tolist()))

        elapsed_time = time.time() - start_time
        doc_id = chunks[0].document_id if chunks else "unknown"

        logger.info(
            f"Embedding Pipeline | Doc: {doc_id} | Chunks: {len(chunks)} | "
            f"Model: {self.service.model_name} | Time: {elapsed_time:.2f}s"
        )

        return embedded_chunks

    def _build_embedded_chunk(self, chunk: Chunk, vector: List[float]) -> EmbeddedChunk:
        """Translates the domain model and securely enriches metadata."""
        meta = chunk.metadata.copy()
        meta["embedding_model"] = self.service.model_name
        meta["embedding_dimension"] = self.service.dimension

        return EmbeddedChunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            text=chunk.text,
            token_count=chunk.token_count,
            start_order=chunk.start_order,
            end_order=chunk.end_order,
            start_location=chunk.start_location,
            end_location=chunk.end_location,
            metadata=meta,
            checksum=chunk.checksum, 
            embedding=vector,
            embedding_dimension=self.service.dimension
        )
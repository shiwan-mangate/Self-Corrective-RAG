# ingestion/chunker/pipeline.py
import logging
from typing import List

from ingestion.models import EnrichedDocument, Chunk
from ingestion.chunker.config import ChunkingConfig
from ingestion.chunker.factory import ChunkerFactory

logger = logging.getLogger(__name__)


class ChunkingPipeline:
    """
    The orchestrator and sole public entry point for the Chunking Subsystem.
    Hides the complexity of strategy selection and parameter routing.
    """
    def __init__(self, config: ChunkingConfig = None):
       
        self.config = config or ChunkingConfig()
        self.chunker = ChunkerFactory.create(self.config)

    def chunk(self, doc: EnrichedDocument) -> List[Chunk]:
        """
        Executes the configured chunking strategy on the enriched document.
        """
        if not doc or not doc.elements:
            logger.warning(f"Empty document provided to chunker: {getattr(doc, 'document_id', 'Unknown')}")
            return []
        chunks = self.chunker.chunk(doc)
        
        
        if chunks:
            avg_tokens = sum(c.token_count for c in chunks) // len(chunks)
            largest_chunk = max(c.token_count for c in chunks)
            
            logger.info(
                f"Chunking Complete | Strategy: {self.config.strategy} | "
                f"Document: {doc.document_id} | Chunks: {len(chunks)} | "
                f"Avg Tokens: {avg_tokens} | Largest: {largest_chunk}"
            )
            
        return chunks
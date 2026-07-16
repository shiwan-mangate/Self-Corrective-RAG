import logging
import time
from typing import List


from ingestion.models import RawDocument, IngestionResult


from ingestion.loaders.factory import LoaderFactory
from ingestion.parser import DocumentParser
from ingestion.cleaner import DocumentCleaner
from ingestion.metadata import MetadataPipeline
from ingestion.chunker.pipeline import ChunkingPipeline
from ingestion.embedding import EmbeddingPipeline


from database.vector_service import VectorService

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    The master orchestrator for Layer 1.
    Purely coordinates data flow. Holds zero persistence or loading logic.
    """
    def __init__(
        self,
        loader_factory: type[LoaderFactory],
        parser: DocumentParser,
        cleaner: DocumentCleaner,
        metadata_extractor: MetadataPipeline,
        chunker: ChunkingPipeline,
        embedder: EmbeddingPipeline,
        vector_service: VectorService
    ):
        self.loader_factory = loader_factory
        self.parser = parser
        self.cleaner = cleaner
        self.metadata_extractor = metadata_extractor
        self.chunker = chunker
        self.embedder = embedder
        self.vector_service = vector_service

    def ingest(self, source: str) -> IngestionResult:
        """
        The exclusive entry point for adding data to the RAG system.
        """
        logger.info(f"Initiating ingestion pipeline for: {source}")
        start_time = time.time()
        
        result = IngestionResult()
        
        try:
           
            loader = self.loader_factory.get_loader(source)
            
            
            loaded_data = loader.load()
            
            
            if not isinstance(loaded_data, list):
                raw_docs = [loaded_data]
            else:
                raw_docs = loaded_data

          
            for raw_doc in raw_docs:
                chunks_saved = self._process_document(raw_doc)
                
                if chunks_saved > 0:
                    result.documents_processed += 1
                    result.chunks_generated += chunks_saved
                    result.chunks_persisted += chunks_saved
                else:
                    result.warnings.append(f"No chunks generated for source: {source}")

        except Exception as e:
            logger.exception(f"Catastrophic failure during ingestion of {source}")
            raise
            
        result.elapsed_time_sec = round(time.time() - start_time, 2)
        logger.info(
            f"Ingestion Complete | Docs: {result.documents_processed} | "
            f"Vectors: {result.chunks_persisted} | Time: {result.elapsed_time_sec}s"
        )
        
        return result

    def _process_document(self, raw_doc: RawDocument) -> int:
        """
        Coordinates the transformation of a single document from raw text to database vectors.
        """
        
        parsed_doc = self.parser.parse(raw_doc)
        
       
        clean_doc = self.cleaner.clean(parsed_doc)
        

        enriched_doc = self.metadata_extractor.enrich(clean_doc)
        
       
        chunks = self.chunker.chunk(enriched_doc)
        
        if not chunks:
            return 0
            
       
        embedded_chunks = self.embedder.embed(chunks)
        
        
        return self.vector_service.replace_document(embedded_chunks)
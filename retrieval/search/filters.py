# retrieval/search/filters.py
import logging
import time
from typing import List, Set, Any
from retrieval.models import RetrievedChunk, AnalyzedQuery, MetadataFilter
from config.constants import MIN_SIMILARITY_SCORE
logger = logging.getLogger(__name__)


class RetrievalFilter:
    """
    The strict validation boundary for the Retrieval Subsystem.
    """

    def filter(
        self, 
        chunks: List[RetrievedChunk], 
        analyzed_query: AnalyzedQuery
    ) -> List[RetrievedChunk]:
        
        if not chunks:
            return []

        start_time = time.time()
        initial_count = len(chunks)

        deduplicated = self._remove_duplicates(chunks)
        dedup_count = initial_count - len(deduplicated)

        similarity_filtered = self._filter_by_similarity(deduplicated)
        sim_count = len(deduplicated) - len(similarity_filtered)

        metadata_filtered = self._filter_by_metadata(
            similarity_filtered, 
            analyzed_query.filters
        )
        meta_count = len(similarity_filtered) - len(metadata_filtered)

        final_count = len(metadata_filtered)
        elapsed_time = time.time() - start_time
        
        logger.info(
            f"Retrieval Filter Metrics | Initial: {initial_count} | "
            f"Dupes Dropped: {dedup_count} | Low Score Dropped: {sim_count} | "
            f"Meta Dropped: {meta_count} | Final Retained: {final_count} | "
            f"Time: {elapsed_time:.4f}s"
        )

        return metadata_filtered

    def _remove_duplicates(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        seen_checksums: Set[str] = set()
        unique_chunks: List[RetrievedChunk] = []

        for chunk in chunks:
            identifier = chunk.checksum if chunk.checksum else hash(chunk.text)
            
            if identifier not in seen_checksums:
                seen_checksums.add(identifier)
                unique_chunks.append(chunk)

        return unique_chunks

    def _filter_by_similarity(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        return [chunk for chunk in chunks if self._is_relevant(chunk)]

    def _is_relevant(self, chunk: RetrievedChunk) -> bool:
        return chunk.similarity_score >= MIN_SIMILARITY_SCORE

    def _filter_by_metadata(
        self, 
        chunks: List[RetrievedChunk], 
        explicit_filters: MetadataFilter
    ) -> List[RetrievedChunk]:
        if not explicit_filters:
            return chunks

       
        filter_dict = {k: v for k, v in explicit_filters.model_dump(exclude_none=True).items() if v != []}
        if not filter_dict:
            return chunks

        valid_chunks: List[RetrievedChunk] = []
        
        for chunk in chunks:
            is_valid = True
            
            for key, required_val in filter_dict.items():
                actual_val = getattr(chunk, key, None)
                if actual_val is None:
                    actual_val = chunk.metadata.get(key)
                
              
                if isinstance(required_val, list):
                    actual_list = actual_val if isinstance(actual_val, list) else [actual_val]
                    
                    if not all(str(r).lower() in [str(a).lower() for a in actual_list] for r in required_val):
                        is_valid = False
                        break
                
                else:
                    if str(actual_val).lower() != str(required_val).lower():
                        is_valid = False
                        break
                    
            if is_valid:
                valid_chunks.append(chunk)

        return valid_chunks
import logging
from typing import List, Set
from self_healing.models import RecoveryContext
from self_healing.constants import MAX_MERGED_CONTEXT_TOKENS, REMOVE_DUPLICATES

from retrieval.models import RankedChunk

logger = logging.getLogger(__name__)


class ContextMerger:
    """
    The Bridge between Recovery and Generation.
    Fuses Internal DB chunks and External Web chunks into a single, high-quality,
    deduplicated, and strictly budgeted markdown context string.
    """

    def __init__(
        self, 
        internal_trust_weight: float = 1.0, 
        web_trust_weight: float = 0.90  
    ):
        self.internal_trust_weight = internal_trust_weight
        self.web_trust_weight = web_trust_weight

    def merge(self, recovery_context: RecoveryContext) -> str:
        """
        Executes the 5-step merging pipeline.
        Returns the finalized markdown string ready for the LLM prompt.
        """
        logger.info(
            f"Context Merger started | "
            f"Internal={len(recovery_context.internal_chunks)} | "
            f"Web={len(recovery_context.web_chunks)}"
        )

        
        all_chunks = recovery_context.internal_chunks + recovery_context.web_chunks
        
        if not all_chunks:
            logger.warning("Context Merger received zero chunks.")
            return ""

        
        unique_chunks = self._deduplicate(all_chunks) if REMOVE_DUPLICATES else all_chunks
        ranked_chunks = self._apply_trust_weights_and_sort(unique_chunks)
        surviving_chunks = self._enforce_budget(ranked_chunks)
        final_markdown = self._format_as_markdown(surviving_chunks)
        recovery_context.merged_context = final_markdown

        logger.info(f"Context Merger complete. {len(surviving_chunks)} chunks survived budgeting.")
        return final_markdown



    def _deduplicate(self, chunks: List[RankedChunk]) -> List[RankedChunk]:
        """Removes exact duplicates using the pre-calculated SHA-256 checksums."""
        seen_hashes: Set[str] = set()
        unique = []
        
        for chunk in chunks:
            chunk_hash = getattr(chunk, "checksum", None) or chunk.chunk_id
            
            if chunk_hash not in seen_hashes:
                seen_hashes.add(chunk_hash)
                unique.append(chunk)
            else:
                logger.debug(f"Merger dropped exact duplicate chunk: {chunk.chunk_id}")
                
        return unique

    def _apply_trust_weights_and_sort(self, chunks: List[RankedChunk]) -> List[RankedChunk]:
        """
        Applies mathematical trust weights to normalize different search algorithms.
        Mutates the final_score to guarantee consistency between sorting and telemetry.
        """
        weighted_chunks = []
        
        for chunk in chunks:
            updated_chunk = chunk.model_copy()
            
            base_score = updated_chunk.final_score
            
            if updated_chunk.source == "web" or getattr(updated_chunk, "metadata", {}).get("provider") == "tavily":
                updated_chunk.final_score = base_score * self.web_trust_weight
            else:
                updated_chunk.final_score = base_score * self.internal_trust_weight
                
            weighted_chunks.append(updated_chunk)
            
       
        weighted_chunks.sort(key=lambda x: x.final_score, reverse=True)
        return weighted_chunks

    def _enforce_budget(self, sorted_chunks: List[RankedChunk]) -> List[RankedChunk]:
        """Iteratively adds the best chunks until the token limit is reached."""
        surviving = []
        current_tokens = 0
        
        for chunk in sorted_chunks:
            tokens = getattr(chunk, "token_count", 200) 
            
            if current_tokens + tokens <= MAX_MERGED_CONTEXT_TOKENS:
                surviving.append(chunk)
                current_tokens += tokens
            else:
                logger.info(
                    f"Merger token budget reached ({current_tokens}/{MAX_MERGED_CONTEXT_TOKENS}). "
                    f"Dropping lower-ranked chunks."
                )
                break
                
        return surviving

    def _format_as_markdown(self, chunks: List[RankedChunk]) -> str:
        """
        Builds the unified string. Explicitly tags sources so the LLM Generation
        Subsystem can accurately cite its claims. Hides raw scores from the LLM.
        """
        context_parts = []
        
        for chunk in chunks:
            source_label = "Internal Source" if chunk.source != "web" else "Web Source"
            doc_id = chunk.document_id
            
            part = f"[{source_label}: {doc_id}]\n"
            part += f"{chunk.text.strip()}\n"
            part += "-" * 40
            
            context_parts.append(part)
            
        return "\n\n".join(context_parts)
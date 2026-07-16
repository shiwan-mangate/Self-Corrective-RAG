from typing import List

from ingestion.models import EnrichedDocument, Chunk
from ingestion.chunker.base import BaseChunker


class FixedChunker(BaseChunker):
    """
    Naive token-based splitting. Used purely as a baseline for A/B testing 
    retrieval performance against the Semantic Chunker. 
    
    WARNING: This completely ignores document structure, slicing through 
    tables, code blocks, and sentences indiscriminately.
    """
    def __init__(self, max_tokens: int = 500, overlap_tokens: int = 50, encoder_model: str = "cl100k_base"):
        super().__init__(max_tokens, encoder_model)
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be strictly less than max_tokens.")
        self.overlap_tokens = overlap_tokens

    def chunk(self, doc: EnrichedDocument) -> List[Chunk]:
        if not doc.elements:
            return []

        
        full_text = "\n\n".join(e.text for e in doc.elements)
        
    
        tokens = self.encoder.encode(full_text)
        
        chunks: List[Chunk] = []
        step = max(1, self.max_tokens - self.overlap_tokens)

     
        default_start_loc = doc.elements[0].location
        default_end_loc = doc.elements[-1].location

        i = 0
        while i < len(tokens):
            chunk_tokens = tokens[i : i + self.max_tokens]
            
       
            remaining_tokens = len(tokens) - (i + self.max_tokens)
            if 0 < remaining_tokens < 15:
                chunk_tokens = tokens[i:]
            
            chunk_text = self.encoder.decode(chunk_tokens)
            
            chunks.append(Chunk(
                document_id=doc.document_id,
                text=chunk_text.strip(),
                token_count=len(chunk_tokens),
                
                
                start_order=0,
                end_order=len(doc.elements) - 1,
                start_location=default_start_loc,
                end_location=default_end_loc,
                
                metadata={
                    "chunk_strategy": "fixed",
                    "overlap_tokens": self.overlap_tokens,
                    "is_baseline": True,
                    "token_start": i,
                    "token_end": i + len(chunk_tokens) - 1
                }
            ))
       
            if i + len(chunk_tokens) >= len(tokens):
                break
                
            i += step

        return chunks
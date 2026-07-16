from typing import List, Tuple

from ingestion.models import EnrichedDocument, Chunk
from ingestion.chunker.base import BaseChunker


class RecursiveChunker(BaseChunker):
    """
    Standard recursive text splitting based on LangChain's core concepts.
    Attempts to split text on the largest semantic boundaries before falling 
    back to strictly token-enforced limits.
    """
    def __init__(self, max_tokens: int = 500, overlap_tokens: int = 50, encoder_model: str = "cl100k_base"):
        super().__init__(max_tokens, encoder_model)
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be strictly less than max_tokens.")
        self.overlap_tokens = overlap_tokens
        
        
        self.delimiters = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]

    def chunk(self, doc: EnrichedDocument) -> List[Chunk]:
        if not doc.elements:
            return []


        full_text = "\n\n".join(e.text for e in doc.elements)
        

        split_pieces = self._recursive_split(full_text, self.delimiters)
        

        return self._merge_pieces(split_pieces, doc)

    def _recursive_split(self, text: str, delimiters: List[str]) -> List[str]:
        """Recursively slices text using the hierarchy of delimiters."""
        if self.count_tokens(text) <= self.max_tokens:
            return [text]
            
        if not delimiters:
           
            tokens = self.encoder.encode(text)
            return [
                self.encoder.decode(tokens[i : i + self.max_tokens]) 
                for i in range(0, len(tokens), self.max_tokens)
            ]
            
        delimiter = delimiters[0]
        next_delimiters = delimiters[1:]
        
        splits = text.split(delimiter)
        valid_pieces = []
        
        for i, split in enumerate(splits):
      
            piece = split + delimiter if i < len(splits) - 1 else split
            
            if not piece:
                continue
                
            if self.count_tokens(piece) <= self.max_tokens:
                valid_pieces.append(piece)
            else:
             
                valid_pieces.extend(self._recursive_split(piece, next_delimiters))
                
        return valid_pieces

    def _merge_pieces(self, pieces: List[str], doc: EnrichedDocument) -> List[Chunk]:
        """Combines the valid pieces into chunks, applying dynamic overlap."""
        chunks: List[Chunk] = []
        current_chunk_text = ""
        current_chunk_tokens = 0
        
   
        current_pieces: List[Tuple[str, int]] = []
        
        
        default_start_loc = doc.elements[0].location
        default_end_loc = doc.elements[-1].location
        default_end_order = len(doc.elements) - 1

        for piece in pieces:
            piece_tokens = self.count_tokens(piece)
            
            if current_chunk_tokens + piece_tokens > self.max_tokens and current_chunk_text:
                
                chunks.append(
                    self._build_chunk(current_chunk_text, doc.document_id, default_start_loc, default_end_loc, default_end_order)
                )
                
                overlap_text = ""
                overlap_tokens = 0
                overlap_pieces: List[Tuple[str, int]] = []
                
                for p_text, p_tokens in reversed(current_pieces):
                 
                    if overlap_tokens + p_tokens > self.overlap_tokens and overlap_pieces:
                        break
                    overlap_text = p_text + overlap_text
                    overlap_tokens += p_tokens
                    overlap_pieces.insert(0, (p_text, p_tokens))
                    
                current_chunk_text = overlap_text + piece
                current_chunk_tokens = overlap_tokens + piece_tokens
                current_pieces = overlap_pieces + [(piece, piece_tokens)]
            else:
              
                current_chunk_text += piece
                current_chunk_tokens += piece_tokens
                current_pieces.append((piece, piece_tokens))
                
     
        if current_chunk_text.strip():
            chunks.append(
                self._build_chunk(current_chunk_text, doc.document_id, default_start_loc, default_end_loc, default_end_order)
            )
            
        return chunks

    def _build_chunk(self, text: str, doc_id: str, start_loc, end_loc, end_order: int) -> Chunk:
        clean_text = text.strip()
        
        return Chunk(
            document_id=doc_id,
            text=clean_text,
            token_count=self.count_tokens(clean_text),
            
         
            start_order=0,
            end_order=end_order,
            start_location=start_loc,
            end_location=end_loc,
            
            metadata={
                "chunk_strategy": "recursive",
                "overlap_tokens": self.overlap_tokens,
                "delimiter_strategy": "expanded",
                "is_baseline": True
            }
        )
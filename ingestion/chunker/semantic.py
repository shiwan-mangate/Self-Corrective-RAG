import copy
from typing import List

from ingestion.models import EnrichedDocument, EnrichedElement, Chunk, ElementType
from ingestion.chunker.base import BaseChunker
from ingestion.chunker.policies import PolicyRouter
from ingestion.chunker.overlap import OverlapCalculator


class SemanticChunker(BaseChunker):
    def __init__(
        self, 
        max_tokens: int = 500, 
        overlap_percent: float = 0.1, 
        max_context_levels: int = 2,           
        heading_break_threshold: float = 0.2,  
        encoder_model: str = "cl100k_base"
    ):
        super().__init__(max_tokens, encoder_model)
        self.max_context_levels = max_context_levels
        self.min_tokens_before_heading_break = int(self.max_tokens * heading_break_threshold)
        
        self.policy_router = PolicyRouter(self.encoder, self.max_tokens)
        self.overlap_calculator = OverlapCalculator(self.max_tokens, overlap_percent)

    def chunk(self, doc: EnrichedDocument) -> List[Chunk]:
        chunks: List[Chunk] = []
        current_elements: List[EnrichedElement] = []
        current_tokens = 0
        chunk_order = 0

        for element in doc.elements:
            el_tokens = element.metadata["statistics"]["token_count"]


            if el_tokens > self.max_tokens:
              
                if current_elements:
                    chunks.append(self._build_chunk(current_elements, doc.document_id, chunk_order))
                    chunk_order += 1
                    current_elements, current_tokens = self.overlap_calculator.get_overlap(current_elements)
                
              
                split_texts = self.policy_router.split_element(element)
                
                for i, text_piece in enumerate(split_texts):
                    piece_tokens = self.count_tokens(text_piece)
                    

                    synthetic_meta = copy.deepcopy(element.metadata)
                    synthetic_meta["statistics"]["token_count"] = piece_tokens
                    synthetic_meta["is_policy_split"] = True
                    
                    synthetic_el = EnrichedElement(
                        element_id=element.element_id,
                        document_id=doc.document_id,
                        type=element.type,
                        text=text_piece,
                        order=element.order,
                        location=element.location,
                        metadata=synthetic_meta
                    )
                    
             
                    chunks.append(
                        self._build_chunk(
                            [synthetic_el], 
                            doc.document_id, 
                            chunk_order, 
                            split_label=element.type.value.upper()
                        )
                    )
                    chunk_order += 1
                    
    
                    if i == len(split_texts) - 1:
                        current_elements, current_tokens = self.overlap_calculator.get_overlap([synthetic_el])
                continue

          
            if element.type == ElementType.HEADING and current_tokens > self.min_tokens_before_heading_break:
                if current_elements:
                    chunks.append(self._build_chunk(current_elements, doc.document_id, chunk_order))
                    chunk_order += 1
                    current_elements, current_tokens = self.overlap_calculator.get_overlap(current_elements)

      
            if current_tokens + el_tokens > self.max_tokens and current_elements:
                chunks.append(self._build_chunk(current_elements, doc.document_id, chunk_order))
                chunk_order += 1
                current_elements, current_tokens = self.overlap_calculator.get_overlap(current_elements)

            current_elements.append(element)
            current_tokens += el_tokens

        if current_elements:
            chunks.append(self._build_chunk(current_elements, doc.document_id, chunk_order))

        return chunks



    def _build_chunk(self, elements: List[EnrichedElement], doc_id: str, order: int, split_label: str = None) -> Chunk:
        """
        Assembles elements into a Chunk.
        Injects configured Parent Context, preventing both lost semantics and prompt bloat.
        """
        first_el = elements[0]
        last_el = elements[-1]
        
        relationships = first_el.metadata.get("relationships", {})
        context_dict = relationships.get("parent_context", {})
        section_id = relationships.get("section_id")

        combined_text = "\n\n".join(e.text for e in elements)

   
        final_text = ""
        if context_dict:
            sorted_keys = sorted([k for k in context_dict.keys() if k <= self.max_context_levels])
            if sorted_keys:
                context_header = " > ".join(context_dict[k] for k in sorted_keys)
                final_text += f"[{context_header}]\n\n"
        
   
        if split_label:
            final_text += f"[SPLIT: {split_label}]\n"
            
        final_text += combined_text
        final_text = final_text.strip()


        merged_meta = {
            "source_elements": [e.element_id for e in elements],
            "parent_context": context_dict,
            "section_id": section_id
        }
        if split_label:
            merged_meta["is_policy_split"] = True

        return Chunk(
            document_id=doc_id,
            text=final_text,
            token_count=self.count_tokens(final_text),
            start_order=first_el.order,
            end_order=last_el.order,
            start_location=first_el.location,
            end_location=last_el.location,
            metadata=merged_meta
        )
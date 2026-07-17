# ingestion/metadata.py
import hashlib
import tiktoken
from langdetect import detect, DetectorFactory
from typing import List, Dict, Any

from ingestion.models import CleanDocument, EnrichedDocument, EnrichedElement, ElementType


DetectorFactory.seed = 0

class MetadataPipeline:
    """
    A multi-step enrichment pipeline that analyzes cleaned documents to attach 
    statistical, structural, and semantic metadata to guide the downstream Chunker.
    """

    def __init__(self, encoding_model: str = "cl100k_base"):
        self.encoder = tiktoken.get_encoding(encoding_model)

    def enrich(self, clean_doc: CleanDocument) -> EnrichedDocument:
        enriched_elements: List[EnrichedElement] = []
        
   
        total_chars = 0
        total_tokens = 0
        longest_paragraph = 0
        
      
        current_hierarchy: Dict[int, str] = {}
        current_section_id = 0

        for element in clean_doc.elements:
          
            stats: Dict[str, Any] = {}
            relationships: Dict[str, Any] = {}
            chunk_hints: Dict[str, Any] = {}
            
          
            native_meta = element.metadata.copy()
            text = element.text
            
           
            char_count = len(text)
            token_count = len(self.encoder.encode(text))
            
            stats["character_count"] = char_count
            stats["token_count"] = token_count
            
            total_chars += char_count
            total_tokens += token_count
            
         
            if element.type == ElementType.HEADING:
                level = native_meta.get("heading_level", 1)
                current_hierarchy[level] = text
                
             
                keys_to_remove = [k for k in current_hierarchy.keys() if k > level]
                for k in keys_to_remove:
                    del current_hierarchy[k]
                    
               
                current_section_id += 1

            relationships["section_id"] = current_section_id
            relationships["parent_context"] = current_hierarchy.copy()
            
       
            if element.type == ElementType.PARAGRAPH:
                longest_paragraph = max(longest_paragraph, token_count)
                
            elif element.type == ElementType.TABLE:
                rows = text.split('\n')
                stats["row_count"] = len(rows)
                stats["column_count"] = len(rows[0].split('|')) if rows else 0
                
                if stats["row_count"] > 15:
                    chunk_hints["directive"] = "large_table_avoid_splitting"
                else:
                    chunk_hints["directive"] = "keep_together"
                    
            elif element.type == ElementType.CODE:
                stats["line_count"] = len(text.split('\n'))
                chunk_hints["directive"] = "keep_together"
                
            elif element.type == ElementType.LIST:
                stats["item_count"] = len(text.split('\n'))
                stats["is_ordered"] = text.strip().startswith("1.") 
                chunk_hints["directive"] = "keep_together"

         
            final_metadata = {
                "native": native_meta,
                "statistics": stats,
                "relationships": relationships,
                "chunk_hints": chunk_hints
            }

            enriched_elements.append(
                EnrichedElement(
                    element_id=element.element_id,
                    document_id=element.document_id,
                    type=element.type,
                    text=text,
                    order=element.order,
                    location=element.location,
                    metadata=final_metadata
                )
            )
            
      
        doc_meta = clean_doc.metadata.copy()
        full_text = "\n".join(e.text for e in clean_doc.elements)
        
 
        if "language" not in doc_meta:
            doc_meta["language"] = self._detect_language(full_text)
        

        doc_meta["content_fingerprint"] = self._generate_hash(full_text)
        
  
        doc_meta["statistics"] = {
            "character_count": total_chars,
            "token_count": total_tokens,
            "longest_paragraph_tokens": longest_paragraph,
            "average_tokens_per_element": (total_tokens // len(clean_doc.elements)) if clean_doc.elements else 0,
            "word_count": len(full_text.split()),
            "reading_time_minutes": max(1, round(len(full_text.split()) / 250))
        }

        return EnrichedDocument(
            document_id=clean_doc.document_id,
            source=clean_doc.source,
            title=clean_doc.title,
            metadata=doc_meta,
            elements=enriched_elements
        )

    def _generate_hash(self, content: str) -> str:
        """Helper to create SHA-256 hashes for content fingerprinting."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _detect_language(self, text: str) -> str:
        """Standardizes language detection across all file types."""
        try:
           
            sample = text[:1000] if len(text) > 1000 else text
            return detect(sample) if sample.strip() else "unknown"
        except Exception:
            return "unknown"
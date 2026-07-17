# ingestion/cleaner.py
import re
import unicodedata
from typing import List

from ingestion.models import ParsedDocument, CleanDocument, CleanElement, ElementType


class DocumentCleaner:
    """
    Improves the quality of the parsed document text without changing its meaning
    or logical structure. Orchestrates a series of focused cleaning strategies.
    """

    def clean(self, parsed_doc: ParsedDocument) -> CleanDocument:
        clean_elements: List[CleanElement] = []
        
        for element in parsed_doc.elements:
            if self._is_unwanted_header_footer(element):
                continue
                
          
            cleaned_text = self._apply_cleaning_pipeline(element.text, element.type)
            
            if not cleaned_text:
                continue
                
            clean_elements.append(
                CleanElement(
                    element_id=element.element_id,
                    document_id=element.document_id,
                    type=element.type,
                    text=cleaned_text,
                    order=element.order,
                    location=element.location,
                    metadata=element.metadata
                )
            )

        return CleanDocument(
            document_id=parsed_doc.document_id,
            source=parsed_doc.source,
            title=parsed_doc.title,
            metadata=parsed_doc.metadata,
            elements=clean_elements
        )



    def _apply_cleaning_pipeline(self, text: str, el_type: ElementType) -> str:
        """Routes text through the appropriate normalization steps."""
    
        if el_type == ElementType.CODE:
            return self._clean_control_characters(text)
            
        text = self._normalize_unicode(text)
        text = self._clean_control_characters(text)
        text = self._normalize_quotes_and_dashes(text)
        text = self._normalize_whitespace(text)
        
     
        if el_type == ElementType.LIST:
            text = self._normalize_bullets(text)
            
        return text.strip()


    def _normalize_unicode(self, text: str) -> str:
        """
        Fixes unicode artifacts like ligatures (ﬁ -> fi) and zero-width spaces.
        Extremely common in PDFs.
        """
        return unicodedata.normalize("NFKC", text)


    def _clean_control_characters(self, text: str) -> str:
        """
        Removes null bytes and other non-printable control characters, 
        but preserves valid whitespace like newlines and tabs.
        """
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)


    def _normalize_quotes_and_dashes(self, text: str) -> str:
        """
        Standardizes typography to prevent the LLM/Embedding model from 
        treating "word" and “word” as different tokens.
        """
        text = re.sub(r'[“”]', '"', text)
        text = re.sub(r'[‘’]', "'", text)
        text = re.sub(r'[—–]', '-', text)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """
        Removes multiple spaces and repeated blank lines.
        """
        text = re.sub(r'[^\S\r\n]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text


    def _normalize_bullets(self, text: str) -> str:
        """
        Ensures all list items start with a standard dash for LLM readability.
        """
        return re.sub(r'^[\u2022\u25E6\u25AA\u25A0]\s*', '- ', text, flags=re.MULTILINE)


    def _is_unwanted_header_footer(self, element) -> bool:
        """
        Identifies and flags isolated artifacts like page numbers or "Confidential".
        """
        if element.type != ElementType.PARAGRAPH:
            return False
        text = element.text.strip().lower()
        if re.match(r'^(page\s*\d+|-\s*\d+\s*-|\d+)$', text, flags=re.IGNORECASE):
            return True
        if text in ["confidential", "internal use only", "proprietary"]:
            return True
            
        return False
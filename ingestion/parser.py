import re
from typing import List, Dict, Any, Callable
import uuid
from ingestion.models import RawDocument, ParsedDocument, ParsedElement, ElementType, ElementLocation


class DocumentParser:
    """
    Normalizes a RawDocument into a standardized ParsedDocument.
    Preserves structural information passed from loaders, and uses heuristic 
    inference only when explicit structure is missing.
    """

    def __init__(self):
        
        self._strategies: Dict[str, Callable[[RawDocument, str], List[ParsedElement]]] = {
            "pdf": self._parse_paginated,
            "pptx": self._parse_slides,
        }

    def parse(self, raw_doc: RawDocument) -> ParsedDocument:
       
        document_id = str(uuid.uuid4())
        
      
        strategy = self._strategies.get(raw_doc.source.lower(), self._parse_standard)
        elements = strategy(raw_doc, document_id)

        return ParsedDocument(
            document_id=document_id,
            source=raw_doc.source,
            title=raw_doc.metadata.get("title"),
            metadata=raw_doc.metadata,
            elements=elements
        )



    def _parse_paginated(self, raw_doc: RawDocument, document_id: str) -> List[ParsedElement]:
        elements = []
        order = 0
        pages = raw_doc.pages or []
        
        for i, page_text in enumerate(pages):
            location = ElementLocation(page=i + 1)
            blocks = self._split_into_blocks(page_text)
            
            for block in blocks:
                elements.append(self._build_element(block, order, location, document_id))
                order += 1
                
        return elements

    def _parse_slides(self, raw_doc: RawDocument, document_id: str) -> List[ParsedElement]:
        elements = []
        order = 0
        slides = raw_doc.slides or []
        
        for i, slide_text in enumerate(slides):
            location = ElementLocation(slide=i + 1)
            blocks = self._split_into_blocks(slide_text)
            
            for block in blocks:
                elements.append(self._build_element(block, order, location, document_id))
                order += 1
                
        return elements

    def _parse_standard(self, raw_doc: RawDocument, document_id: str) -> List[ParsedElement]:
        elements = []
        order = 0
        location = ElementLocation()
        
        blocks = self._split_into_blocks(raw_doc.content)
        for block in blocks:
            elements.append(self._build_element(block, order, location, document_id))
            order += 1
            
        return elements



    def _split_into_blocks(self, content: str) -> List[str]:
        """
        Isolates coherent blocks of text. Future layout-aware parsers 
        will override or augment this step.
        """
        if not content:
            return []
       
        blocks = re.split(r'\n{2,}', content.strip())
        return [block.strip() for block in blocks if block.strip()]

    def _infer_element_type(self, block: str) -> ElementType:
        """
        Heuristic fallback to guess element types based on text patterns.
        Used only when the loader hasn't provided explicit structural tags.
        """
        if re.match(r'^(#{1,6})\s+(.+)$', block, flags=re.MULTILINE):
            return ElementType.HEADING
        elif block.startswith("```") and block.endswith("```"):
            return ElementType.CODE
        elif block.startswith(">"):
            return ElementType.QUOTE
        elif re.match(r'^\|?.*\|.*\|?$', block, flags=re.MULTILINE):
            return ElementType.TABLE
        elif re.match(r'^(\s*[-*+]\s+|\s*\d+\.\s+)', block):
            return ElementType.LIST
        else:
            return ElementType.PARAGRAPH

    def _build_element(self, text: str, order: int, location: ElementLocation, document_id: str) -> ParsedElement:
        """
        Assembles the ParsedElement, preserving metadata attributes if 
        destructive formatting removal is required by the heuristics.
        """
        el_type = self._infer_element_type(text)
        meta: Dict[str, Any] = {}
        clean_text = text

        
        if el_type == ElementType.HEADING:
            match = re.match(r'^(#{1,6})\s+(.+)$', text, flags=re.MULTILINE)
            if match:
                meta["heading_level"] = len(match.group(1))
                clean_text = match.group(2).strip()
                
        elif el_type == ElementType.CODE:
            lines = text.split('\n')
            if len(lines) > 1 and lines[0].strip('`').strip():
                meta["language"] = lines[0].strip('`').strip()
           
            clean_text = '\n'.join(lines[1:-1]).strip() if len(lines) > 2 else text
            
        elif el_type == ElementType.QUOTE:
            clean_text = text.replace(">", "").strip()

        return ParsedElement(
            document_id=document_id,
            type=el_type,
            text=clean_text,
            order=order,
            location=location,
            metadata=meta
        )
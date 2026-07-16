import uuid
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator
import hashlib





class RawDocument(BaseModel):
    content: str = Field(..., description="Complete raw text extracted from the source.")
    source: str = Field(..., description="Document source type (pdf, docx, pptx, url, text, markdown, html).")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Loader-level metadata.")
    pages: Optional[List[str]] = Field(default=None, description="Individual pages for paginated documents.")
    slides: Optional[List[str]] = Field(default=None, description="Individual slides for presentations.")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal issues encountered during loading.")


class ElementType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    CODE = "code"
    QUOTE = "quote"


class ElementLocation(BaseModel):
    """
    A first-class representation of where an element lived in the original source.
    """
    page: Optional[int] = Field(default=None, description="Page number, if applicable.")
    slide: Optional[int] = Field(default=None, description="Slide number, if applicable.")
   


class ParsedElement(BaseModel):
    """
    A single logical piece of knowledge from a document.
    """
    element_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = Field(..., description="The ID of the parent ParsedDocument.")
    type: ElementType = Field(..., description="The structural classification of the element.")
    text: str = Field(..., description="The actual text content of the element.")
    order: int = Field(..., description="The strict sequential order of this element in the document.")
    location: ElementLocation = Field(default_factory=lambda: ElementLocation(), description="Explicit location boundaries.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Attributes like heading_level or code_language.")


class ParsedDocument(BaseModel):
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str = Field(...)
    title: Optional[str] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    elements: List[ParsedElement] = Field(default_factory=list)

class CleanElement(BaseModel):
    """
    A fully normalized, structurally intact piece of knowledge.
    """
    element_id: str
    document_id: str
    type: ElementType
    text: str
    order: int
    location: ElementLocation
    metadata: Dict[str, Any]

class CleanDocument(BaseModel):
    """
    The pristine representation of a document, ready for chunking.
    """
    document_id: str
    source: str
    title: Optional[str] = None
    metadata: Dict[str, Any]
    elements: List[CleanElement]




class EnrichedElement(BaseModel):
    """
    A logical element enriched with statistical and structural metadata.
    """
    element_id: str
    document_id: str
    type: ElementType
    text: str
    order: int
    location: ElementLocation
    metadata: Dict[str, Any] 

class EnrichedDocument(BaseModel):
    """
    A fully analyzed document ready for intelligent chunking.
    """
    document_id: str
    source: str
    title: Optional[str] = None
    metadata: Dict[str, Any]
    elements: List[EnrichedElement]



class Chunk(BaseModel):
    """
    A standalone piece of semantic knowledge ready for embedding.
    """
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = Field(...)
    text: str = Field(...)
    token_count: int = Field(...)
    start_order: int = Field(...)
    end_order: int = Field(...)
    start_location: ElementLocation = Field(...)
    end_location: ElementLocation = Field(...)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    checksum: Optional[str] = Field(default=None)

    @model_validator(mode='after')
    def auto_generate_checksum(self):
        """Automatically hashes the text upon instantiation if not provided."""
        if not self.checksum and self.text:
            self.checksum = hashlib.sha256(self.text.encode('utf-8')).hexdigest()
        return self

class EmbeddedChunk(Chunk):
    embedding: List[float] = Field(...)
    embedding_dimension: Optional[int] = Field(default=None)



class IngestionResult(BaseModel):
    """Rich telemetry returned after an ingestion run."""
    documents_processed: int = 0
    chunks_generated: int = 0
    chunks_persisted: int = 0
    warnings: List[str] = Field(default_factory=list)
    elapsed_time_sec: float = 0.0
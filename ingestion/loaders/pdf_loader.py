import pdfplumber
from typing import Dict, Any

from ingestion.models import RawDocument
from ingestion.loaders.base import BaseLoader


class PDFLoader(BaseLoader):
    """
    Loader for PDF (.pdf) files.
    Extracts raw text page by page using pdfplumber, attempting to preserve 
    reading order where possible, and strictly preserving page boundaries.
    Extracts rich native PDF metadata and structural statistics.
    """

    @property
    def name(self) -> str:
        return "pdf"

    def load(self) -> RawDocument:
       
        path = self._validate_file([".pdf"])
        metadata = self._build_base_metadata(path)
        pages_text = []
        
        stats = {
            "total_pages": 0,
            "text_page_count": 0,
            "empty_page_count": 0,
            "image_only_pages": 0,
            "rotated_page_count": 0
        }

       
        try:
            with pdfplumber.open(path) as pdf:
               
                if pdf.metadata:
                    self._extract_pdf_metadata(pdf.metadata, metadata)

                stats["total_pages"] = len(pdf.pages)

                for i, page in enumerate(pdf.pages):
                   
                    if page.rotation != 0:
                        stats["rotated_page_count"] += 1
                        self.log_warning(f"Page {i + 1} is rotated ({page.rotation} degrees).")

                 
                    text = page.extract_text()
                    
                    if text and text.strip():
                        clean_text = text.strip()
                        pages_text.append(clean_text)
                        stats["text_page_count"] += 1
                    else:
                      
                        pages_text.append("")
                        stats["empty_page_count"] += 1
                        stats["image_only_pages"] += 1
                        self.log_warning(f"Page {i + 1} yielded no text (possible image-only or empty page).")

        except Exception as e:
            raise RuntimeError(f"Fatal error reading PDF file: {str(e)}")


        content = "\n\n--- PAGE BREAK ---\n\n".join(pages_text).strip()

       
        if not content:
            self.log_warning("No textual content extracted from PDF document. It may be a scanned document requiring OCR.")

       
        metadata.update(stats)
        metadata.update(self._calculate_text_statistics(content))
        
      
        if stats["total_pages"] > 0:
            metadata["average_characters_per_page"] = metadata["character_count"] // stats["total_pages"]
        else:
            metadata["average_characters_per_page"] = 0

        return RawDocument(
            content=content,
            source=self.name,
            metadata=metadata,
            pages=pages_text,  
            warnings=self.warnings
        )

    def _extract_pdf_metadata(self, pdf_meta: Dict[str, Any], metadata: Dict[str, Any]):
        """Helper to safely extract and normalize standard and enterprise PDF metadata fields."""
        

        keys_to_extract = {
            "author": ["Author", "/Author"],
            "title": ["Title", "/Title"],
            "created_at": ["CreationDate", "/CreationDate"],  
            "producer": ["Producer", "/Producer"],
            "creator": ["Creator", "/Creator"],
            "subject": ["Subject", "/Subject"],
            "keywords": ["Keywords", "/Keywords"]
        }

        for meta_key, pdf_keys in keys_to_extract.items():
            for pk in pdf_keys:
                val = pdf_meta.get(pk)
               
                if val and isinstance(val, str) and val.strip():
                    metadata[meta_key] = val.strip()
                    break  
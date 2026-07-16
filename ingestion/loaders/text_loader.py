from typing import Dict, Any

from ingestion.models import RawDocument
from ingestion.loaders.base import BaseLoader


class TextLoader(BaseLoader):
    """
    Loader for plain text (.txt, .log) files.
    Extracts raw text while utilizing a robust multi-encoding fallback 
    strategy to handle legacy enterprise files without crashing the pipeline.
    """

    @property
    def name(self) -> str:
        return "text"

    def load(self) -> RawDocument:
   
        path = self._validate_file([".txt", ".log", ".csv"])
        metadata = self._build_base_metadata(path)

      
        encodings = ["utf-8", "utf-8-sig", "windows-1252", "iso-8859-1"]
        content = None
        
        for enc in encodings:
            try:
                with open(path, "r", encoding=enc) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
                
        if content is None:
    
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            self.log_warning("Fell back to UTF-8 with replacement characters due to severe encoding issues.")

   
        if not content.strip():
            self.log_warning("No textual content extracted from text document.")

       
        metadata.update(self._calculate_text_statistics(content))

        return RawDocument(
            content=content,
            source=self.name,
            metadata=metadata,
            warnings=self.warnings
        )
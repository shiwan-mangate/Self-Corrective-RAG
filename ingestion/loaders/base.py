from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pathlib import Path

from ingestion.models import RawDocument


class BaseLoader(ABC):
    """
    Abstract contract for all document loaders.
    
    Implementations are responsible only for extracting raw document content 
    and returning a standardized RawDocument. They must not clean text, 
    generate metadata, chunk documents, create embeddings, or interact 
    with the database.
    """
    
    def __init__(self, source: str):
        self.source = source
        self._warnings: List[str] = []

    @property
    def warnings(self) -> List[str]:
        return self._warnings

    def log_warning(self, message: str):
        self._warnings.append(message)

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def _validate_file(self, allowed_extensions: List[str]) -> Path:
        """Validates file existence and extension, returning the Path object."""
        path = Path(self.source)
        if not path.exists():
            raise FileNotFoundError(f"{self.name.upper()} file not found: {path}")
            
        if path.suffix.lower() not in allowed_extensions:
            raise ValueError(f"Expected one of {allowed_extensions}, got '{path.suffix}'")
            
        return path

    def _build_base_metadata(self, path: Path) -> Dict[str, Any]:
        """Generates the standard metadata shared by all file loaders."""
        return {
            "filename": path.name,
            "file_path": str(path),
            "file_size_bytes": path.stat().st_size,
            "loader": self.name
        }

    def _calculate_text_statistics(self, content: str) -> Dict[str, int]:
        """Calculates standard text statistics for analytics."""
        return {
            "character_count": len(content),
            "word_count": len(content.split()),
            "line_count": 0 if not content else len(content.splitlines())
        }

    @abstractmethod
    def load(self) -> RawDocument:
        raise NotImplementedError
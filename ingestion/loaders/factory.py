from pathlib import Path
from urllib.parse import urlparse
from typing import Type, Dict

from ingestion.loaders.base import BaseLoader
from ingestion.loaders.pdf_loader import PDFLoader
from ingestion.loaders.docx_loader import DOCXLoader
from ingestion.loaders.pptx_loader import PPTXLoader
from ingestion.loaders.text_loader import TextLoader
from ingestion.loaders.markdown_loader import MarkdownLoader
from ingestion.loaders.html_loader import HTMLLoader
from ingestion.loaders.url_loader import URLLoader


class LoaderFactory:
    """
    Centralizes loader selection. Acts as the single entry point for routing 
    a source (URL or file path) to its appropriate BaseLoader implementation.
    """


    _file_loaders: Dict[str, Type[BaseLoader]] = {
        ".pdf": PDFLoader,
        ".docx": DOCXLoader,
        ".pptx": PPTXLoader,
        ".txt": TextLoader,
        ".log": TextLoader,
        ".csv": TextLoader, 
        ".md": MarkdownLoader,
        ".markdown": MarkdownLoader,
        ".html": HTMLLoader,
        ".htm": HTMLLoader,
    }

    @classmethod
    def get_loader(cls, source: str) -> BaseLoader:
        """
        Detects the source type and returns the instantiated loader.
        
        :param source: The file path or URL to load.
        :return: An instantiated loader that conforms to BaseLoader.
        :raises ValueError: If the source format is not supported.
        """
     
        parsed_url = urlparse(source)
        if parsed_url.scheme in ["http", "https"] and parsed_url.netloc:
            return URLLoader(source)

      
        path = Path(source)
        ext = path.suffix.lower()

        loader_class = cls._file_loaders.get(ext)
        
     
        if not loader_class:
            raise ValueError(
                f"Unsupported source format. Could not find a loader for extension: "
                f"'{ext}' (Source: {source})"
            )

       
        return loader_class(source)
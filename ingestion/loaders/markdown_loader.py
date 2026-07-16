import re
import yaml
from typing import Dict, Any

from ingestion.models import RawDocument
from ingestion.loaders.base import BaseLoader


class MarkdownLoader(BaseLoader):
    """
    Loader for Markdown (.md, .markdown) files.
    Extracts YAML front matter into metadata and captures rich structural 
    statistics natively without altering the Markdown formatting.
    """

    @property
    def name(self) -> str:
        return "markdown"

    def load(self) -> RawDocument:
        # 1. Boilerplate handled by BaseLoader
        path = self._validate_file([".md", ".markdown"])
        metadata = self._build_base_metadata(path)
        
        # 2. Robust Encoding Fallback Strategy
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

        
        front_matter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, flags=re.DOTALL)
        if front_matter_match:
            try:
                front_matter = yaml.safe_load(front_matter_match.group(1))
                if isinstance(front_matter, dict):
                    metadata.update(front_matter)
            except yaml.YAMLError as e:
                self.log_warning(f"Failed to parse YAML front matter: {str(e)}")
            
       
            content = content[front_matter_match.end():]

        
        self._extract_statistics(content, metadata)


        if not content.strip():
            self.log_warning("No textual content extracted from Markdown document.")

  
        metadata.update(self._calculate_text_statistics(content))

        return RawDocument(
            content=content,
            source=self.name,
            metadata=metadata,
            warnings=self.warnings
        )

    def _extract_statistics(self, content: str, metadata: Dict[str, Any]):
        """Helper to cleanly encapsulate the regex counting logic."""
        
  
        stats = {
            "heading_count": 0,
            "h1_count": 0, "h2_count": 0, "h3_count": 0, "h4_count": 0, "h5_count": 0, "h6_count": 0,
            "code_block_count": 0,
            "code_languages": {},
            "image_count": 0,
            "image_alt_count": 0,
            "internal_links": 0,
            "external_links": 0,
            "table_count": 0,
            "unordered_list_count": 0,
            "ordered_list_count": 0,
            "blockquote_count": 0
        }


        headings = re.findall(r'^(#{1,6})\s+(.+)$', content, flags=re.MULTILINE)
        stats["heading_count"] = len(headings)
        for hashes, _ in headings:
            stats[f"h{len(hashes)}_count"] += 1


        code_blocks = re.findall(r'```([a-zA-Z0-9_+-]+)?\n[\s\S]*?```', content)
        stats["code_block_count"] = len(code_blocks)
        for lang in code_blocks:
            if lang:
                lang_clean = lang.strip().lower()
                stats["code_languages"][lang_clean] = stats["code_languages"].get(lang_clean, 0) + 1

        images = re.findall(r'!\[(.*?)\]\((.*?)\)', content)
        stats["image_count"] = len(images)
        stats["image_alt_count"] = sum(1 for alt, _ in images if alt.strip())
        if stats["image_count"] > 0:
            self.log_warning(f"Document contains {stats['image_count']} Markdown images that were not processed.")

        
        links = re.findall(r'(?<!!)\[.*?\]\((.*?)\)', content)
        for url in links:
            if url.startswith("http://") or url.startswith("https://"):
                stats["external_links"] += 1
            else:
                stats["internal_links"] += 1

        
        stats["table_count"] = len(re.findall(r'^\|?\s*-{3,}\s*\|.*$', content, flags=re.MULTILINE))

       
        stats["unordered_list_count"] = len(re.findall(r'^\s*[-*+]\s+', content, flags=re.MULTILINE))
        stats["ordered_list_count"] = len(re.findall(r'^\s*\d+\.\s+', content, flags=re.MULTILINE))

        
        stats["blockquote_count"] = len(re.findall(r'^\s*>\s+', content, flags=re.MULTILINE))

       
        metadata.update(stats)
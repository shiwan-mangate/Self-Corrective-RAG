from pathlib import Path
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup

from ingestion.models import RawDocument
from ingestion.loaders.base import BaseLoader


class HTMLLoader(BaseLoader):
    """
    Loader for HTML (.html, .htm) files.
    Strips noisy elements while preserving semantic structure natively 
    (Markdown headings, code blocks, blockquotes, ordered/unordered lists) 
    and extracting rich enterprise metadata (canonical URLs, alt-text stats).
    """

    @property
    def name(self) -> str:
        return "html"

    def load(self) -> RawDocument:
        path = Path(self.source)

        if not path.exists():
            raise FileNotFoundError(f"HTML file not found: {path}")

        if path.suffix.lower() not in [".html", ".htm"]:
            raise ValueError(f"Expected a .html or .htm file, got '{path.suffix}'")

        # Initialize enriched metadata and counts
        metadata = {
            "filename": path.name,
            "file_path": str(path),
            "file_size_bytes": path.stat().st_size,
            "loader": self.name,
            "table_count": 0,
            "image_count": 0,
            "image_alt_count": 0,
            "internal_links": 0,
            "external_links": 0,
            "heading_count": 0
        }

        try:
           
            with open(path, "rb") as f:
                soup = BeautifulSoup(f, "lxml")
        except Exception as e:
            raise RuntimeError(f"Fatal error parsing HTML file: {str(e)}")

        
        try:
            if soup.title and soup.title.string:
                metadata["title"] = soup.title.string.strip()
            
            html_tag = soup.find("html")
            if html_tag and html_tag.get("lang"):
                metadata["language"] = html_tag.get("lang")

            canonical = soup.find("link", rel="canonical")
            if canonical and canonical.get("href"):
                metadata["canonical_url"] = canonical.get("href")

            author_meta = soup.find("meta", attrs={"name": "author"})
            if author_meta and author_meta.get("content"):
                metadata["author"] = author_meta["content"].strip()
                
            desc_meta = soup.find("meta", attrs={"name": "description"})
            if desc_meta and desc_meta.get("content"):
                metadata["description"] = desc_meta["content"].strip()
        except Exception as e:
            self.log_warning(f"Could not extract native HTML meta properties: {str(e)}")

        
        images = soup.find_all("img")
        if images:
            metadata["image_count"] = len(images)
            metadata["image_alt_count"] = sum(1 for img in images if img.get("alt") and img.get("alt").strip())
            self.log_warning(f"Document contains {len(images)} images.")
            
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if href.startswith("http") or href.startswith("//"):
                metadata["external_links"] += 1
            elif not href.startswith("javascript:") and not href.startswith("mailto:"):
                metadata["internal_links"] += 1

        
        noisy_tags = [
            "script", "style", "nav", "footer", "header", "aside", "form", 
            "noscript", "iframe", "svg", "canvas", "template", "dialog", 
            "picture", "source"
        ]
        for tag in soup(noisy_tags):
            tag.decompose()

        
      
        for table in soup.find_all("table"):
            metadata["table_count"] += 1
            table_data = []
            for row in table.find_all("tr"):
                row_cells = [cell.get_text(strip=True) for cell in row.find_all(["th", "td"])]
                if any(row_cells):
                    table_data.append(" | ".join(row_cells))
            if table_data:
                formatted_table = "\n\n" + "\n".join(table_data) + "\n\n"
                table.replace_with(soup.new_string(formatted_table))

       
        for i in range(1, 7):
            for heading in soup.find_all(f"h{i}"):
                metadata["heading_count"] += 1
                text = heading.get_text(strip=True)
                if text:
                    markdown_heading = f"\n\n{'#' * i} {text}\n\n"
                    heading.replace_with(soup.new_string(markdown_heading))

   
        for pre in soup.find_all("pre"):
            code_text = pre.get_text()
            if code_text.strip():
                pre.replace_with(soup.new_string(f"\n\n```\n{code_text}\n```\n\n"))

     
        for blockquote in soup.find_all("blockquote"):
            quote_text = blockquote.get_text(strip=True)
            if quote_text:
                formatted_quote = "\n".join(f"> {line}" for line in quote_text.split("\n"))
                blockquote.replace_with(soup.new_string(f"\n\n{formatted_quote}\n\n"))

        for ul in soup.find_all("ul"):
            for li in ul.find_all("li", recursive=False):
                text = li.get_text(strip=True)
                if text:
                    li.replace_with(soup.new_string(f"\n- {text}"))
            ul.insert_before(soup.new_string("\n"))
            ul.insert_after(soup.new_string("\n"))

        for ol in soup.find_all("ol"):
            for index, li in enumerate(ol.find_all("li", recursive=False), start=1):
                text = li.get_text(strip=True)
                if text:
                    li.replace_with(soup.new_string(f"\n{index}. {text}"))
            ol.insert_before(soup.new_string("\n"))
            ol.insert_after(soup.new_string("\n"))

       
        content = soup.get_text(separator="\n")
        

        content = re.sub(r'\n{3,}', '\n\n', content).strip()

        if not content:
            self.log_warning("No textual content extracted from HTML document.")

        metadata["character_count"] = len(content)
        metadata["word_count"] = len(content.split())

        return RawDocument(
            content=content,
            source=self.name,
            metadata=metadata,
            warnings=self.warnings
        )
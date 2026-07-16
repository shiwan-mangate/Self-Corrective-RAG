import re
import os
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional

from ingestion.models import RawDocument
from ingestion.loaders.base import BaseLoader


class URLLoader(BaseLoader):
    """
    Loader for web URLs.
    Implements robust network fetching (Timeouts, User-Agent spoofing), 
    a Tavily API fallback for blocked pages, and highly accurate 
    HTML-to-Markdown DOM mutation for semantic preservation.
    """

    @property
    def name(self) -> str:
        return "url"

    def load(self) -> RawDocument:
        
        parsed_url = urlparse(self.source)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError(f"Invalid URL provided: {self.source}")

        
        metadata = {
            "url": self.source,
            "domain": parsed_url.netloc,
            "loader": self.name,
            "table_count": 0,
            "image_count": 0,
            "internal_links": 0,
            "external_links": 0,
            "heading_count": 0
        }

      
        html_content = self._fetch_content()
        if not html_content:
            self.log_warning(f"Failed to extract any content from URL: {self.source}")
            return RawDocument(content="", source=self.name, metadata=metadata, warnings=self.warnings)

    
        try:
            soup = BeautifulSoup(html_content, "lxml")
        except Exception as e:
            raise RuntimeError(f"Fatal error parsing downloaded HTML from URL: {str(e)}")

     
        self._extract_html_metadata(soup, metadata)

        
        noisy_tags = [
            "script", "style", "nav", "footer", "header", "aside", "form", 
            "noscript", "iframe", "svg", "canvas", "template", "dialog", 
            "picture", "source"
        ]
        for tag in soup(noisy_tags):
            tag.decompose()

      
        self._mutate_dom_to_markdown(soup, metadata)

       
        content = soup.get_text(separator="\n")
       
        content = re.sub(r'\n{3,}', '\n\n', content).strip()

        if not content:
            self.log_warning("No textual content remained after HTML noise removal.")

     
        metadata.update(self._calculate_text_statistics(content))

        return RawDocument(
            content=content,
            source=self.name,
            metadata=metadata,
            warnings=self.warnings
        )

    def _fetch_content(self) -> Optional[bytes]:
        """Attempts a standard HTTP GET, falling back to Tavily if blocked/failed."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        }

        try:
            response = requests.get(self.source, headers=headers, timeout=15)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            self.log_warning(f"Standard HTTP fetch failed ({str(e)}). Attempting Tavily fallback...")
            return self._tavily_fallback_fetch()

    def _tavily_fallback_fetch(self) -> Optional[bytes]:
        """Uses the Tavily Extract API to bypass Cloudflare/JS-rendered blocks."""
        tavily_key = os.getenv("TAVILY_API_KEY")
        if not tavily_key:
            self.log_warning("Tavily API key not found in environment. Fallback aborted.")
            return None

        try:
            response = requests.post(
                "https://api.tavily.com/extract",
                json={"urls": [self.source]},
                headers={"Authorization": f"Bearer {tavily_key}"},
                timeout=20
            )
            response.raise_for_status()
            data = response.json()
            if data.get("results") and len(data["results"]) > 0:
                self.log_warning("Successfully recovered content using Tavily Fallback.")
                
                return f"<html><body>{data['results'][0]['raw_content']}</body></html>".encode('utf-8')
            return None
        except Exception as e:
            self.log_warning(f"Tavily fallback also failed: {str(e)}")
            return None

    def _extract_html_metadata(self, soup: BeautifulSoup, metadata: Dict[str, Any]):
        """Extracts standard meta tags and link/image statistics."""
        try:
            if soup.title and soup.title.string:
                metadata["title"] = soup.title.string.strip()
            
            html_tag = soup.find("html")
            if html_tag and html_tag.get("lang"):
                metadata["language"] = html_tag.get("lang")

        
            canonical = soup.find("link", rel="canonical")
            if canonical and canonical.get("href"):
                metadata["canonical_url"] = canonical.get("href")

            desc_meta = soup.find("meta", attrs={"name": "description"})
            if desc_meta and desc_meta.get("content"):
                metadata["description"] = desc_meta["content"].strip()
        except Exception as e:
            self.log_warning(f"Could not extract native HTML meta properties: {str(e)}")

      
        images = soup.find_all("img")
        metadata["image_count"] = len(images)
            
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if href.startswith("http") or href.startswith("//"):
                metadata["external_links"] += 1
            elif not href.startswith("javascript:") and not href.startswith("mailto:"):
                metadata["internal_links"] += 1

    def _mutate_dom_to_markdown(self, soup: BeautifulSoup, metadata: Dict[str, Any]):
        """Translates HTML structures into Markdown for LLM compatibility."""
       
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
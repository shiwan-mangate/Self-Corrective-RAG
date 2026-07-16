import re
import logging
from typing import Dict

from retrieval.models import Citation
from generation.constants import CitationStyle, CITATION_REGEX_PATTERN

logger = logging.getLogger(__name__)


class CitationFormatter:
    """
    Handles the presentation and styling logic for inline text citations.
    
    Responsibility:
    Cleans hallucinated markers from the raw text and applies UI-specific 
    styling (e.g., Unicode superscripting).
    """

    def format_text(
        self, 
        text: str, 
        citation_map: Dict[int, Citation],
        style: CitationStyle = CitationStyle.BRACKETS
    ) -> str:
        
        # 1. Surgically remove hallucinated/invalid markers (e.g., "[99]")
        clean_text = self._remove_invalid_markers(text, citation_map)
        
        # 2. Apply requested UI styling
        if style == CitationStyle.SUPERSCRIPT:
            clean_text = self._to_superscript(clean_text)
            
        return clean_text

    def _remove_invalid_markers(self, text: str, citation_map: Dict[int, Citation]) -> str:
        """
        Scans for citation brackets and removes numbers that don't exist in the map.
        If a bracket becomes empty (e.g., [99] -> []), it is deleted entirely.
        """
        def replace_match(match):
            parts = [p.strip() for p in match.group(1).split(",")]
            
            # Keep only the numbers that exist in our deterministic map
            valid_parts = [p for p in parts if p.isdigit() and int(p) in citation_map]
            
            if not valid_parts:
                return ""  # Remove the marker completely from the text
                
            return f"[{', '.join(valid_parts)}]"
            
        return re.sub(CITATION_REGEX_PATTERN, replace_match, text)

    def _to_superscript(self, text: str) -> str:
        """Translates standard bracket markers into Unicode superscripts (e.g., [1, 2] -> ¹, ²)."""
        sup_map = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        
        def replace_match(match):
            inner_text = match.group(1)
            # Translate digits, leave commas and spaces as is
            return inner_text.translate(sup_map)
            
        return re.sub(CITATION_REGEX_PATTERN, replace_match, text)
import tiktoken
from abc import ABC, abstractmethod
from typing import List

from ingestion.models import EnrichedElement, ElementType


class BaseSplitPolicy(ABC):
    """
    Abstract contract for element-specific splitting rules.
    Handles the core recursive splitting engine so derived classes 
    only need to define their specific delimiters.
    """
    def __init__(self, encoder: tiktoken.Encoding, max_tokens: int):
        self.encoder = encoder
        self.max_tokens = max_tokens

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    @abstractmethod
    def split(self, element: EnrichedElement) -> List[str]:
        """Returns a list of text strings that strictly conform to max_tokens."""
        pass

    def _recursive_split(self, text: str, delimiters: List[str]) -> List[str]:
        """A highly reusable recursive text splitter."""
        def _split(text_to_split: str, depth: int) -> List[str]:
            if depth >= len(delimiters):
                # Ultimate fallback: character-based slicing if delimiters fail
                char_limit = int(self.max_tokens * 3.5)
                return [text_to_split[i:i+char_limit] for i in range(0, len(text_to_split), char_limit)]
                
            delim = delimiters[depth]
            pieces = text_to_split.split(delim)
            
            valid_pieces = []
            current_piece = ""
            
            for piece in pieces:
                # Reconstruct the string with the delimiter (unless it's the first piece)
                test_str = (current_piece + delim + piece).strip() if current_piece else piece
                
                if self.count_tokens(test_str) <= self.max_tokens:
                    current_piece = test_str
                else:
                    if current_piece:
                        valid_pieces.append(current_piece)
                    
                    # If a single piece is STILL too large, recurse deeper
                    if self.count_tokens(piece) > self.max_tokens:
                        valid_pieces.extend(_split(piece, depth + 1))
                        current_piece = ""
                    else:
                        current_piece = piece
                        
            if current_piece:
                valid_pieces.append(current_piece)
                
            return valid_pieces

        return _split(text, 0)


class ParagraphPolicy(BaseSplitPolicy):
    """Splits long paragraphs safely by double newlines, single newlines, then sentences."""
    def split(self, element: EnrichedElement) -> List[str]:
        delimiters = ["\n\n", "\n", ". ", "? ", "! ", " "]
        return self._recursive_split(element.text, delimiters)


class CodePolicy(BaseSplitPolicy):
    """Splits massive code blocks by structural spacing before falling back to lines."""
    def split(self, element: EnrichedElement) -> List[str]:
        # Prioritizes keeping functions/classes intact via double newlines
        delimiters = ["\n\n", "\n", " "]
        return self._recursive_split(element.text, delimiters)


class ListPolicy(BaseSplitPolicy):
    """Splits massive lists safely by items."""
    def split(self, element: EnrichedElement) -> List[str]:
        delimiters = ["\n- ", "\n* ", "\n1. ", "\n", " "]
        return self._recursive_split(element.text, delimiters)


class TablePolicy(BaseSplitPolicy):
    """
    Splits massive markdown tables by rows, but crucially preserves 
    and injects the header row into every resulting chunk.
    """
    def split(self, element: EnrichedElement) -> List[str]:
        lines = element.text.split("\n")
        
        # Heuristic: Extract the Markdown header and separator (usually lines 0 and 1)
        header = ""
        body_lines = lines
        if len(lines) > 2 and "|" in lines[0] and "|-" in lines[1]:
            header = lines[0] + "\n" + lines[1] + "\n"
            body_lines = lines[2:]

        chunks = []
        current_chunk = header
        
        for line in body_lines:
            test_chunk = current_chunk + line + "\n"
            if self.count_tokens(test_chunk) <= self.max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk.strip() and current_chunk.strip() != header.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = header + line + "\n"
                
                # Extreme fallback if a single row is somehow thousands of tokens
                if self.count_tokens(current_chunk) > self.max_tokens:
                     chunks.extend(self._recursive_split(line, [" | ", " "]))
                     current_chunk = header
                     
        if current_chunk.strip() and current_chunk.strip() != header.strip():
            chunks.append(current_chunk.strip())
            
        return chunks


class PolicyRouter:
    """
    Acts as the receptionist for oversized elements. 
    Routes the element to the correct policy based on its ElementType.
    """
    def __init__(self, encoder: tiktoken.Encoding, max_tokens: int):
        self.policies = {
            ElementType.PARAGRAPH: ParagraphPolicy(encoder, max_tokens),
            ElementType.TABLE: TablePolicy(encoder, max_tokens),
            ElementType.CODE: CodePolicy(encoder, max_tokens),
            ElementType.LIST: ListPolicy(encoder, max_tokens),
        }
        self.default_policy = ParagraphPolicy(encoder, max_tokens)

    def split_element(self, element: EnrichedElement) -> List[str]:
        """Routes the element and returns perfectly sized text chunks."""
        policy = self.policies.get(element.type, self.default_policy)
        return policy.split(element)
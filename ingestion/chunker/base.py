import tiktoken
from abc import ABC, abstractmethod
from typing import List

from ingestion.models import EnrichedDocument, Chunk


class BaseChunker(ABC):
    """
    Abstract contract for all chunking strategies.
    
    Ensures that every chunking algorithm adheres to the exact same input/output 
    boundary, allowing the pipeline to switch strategies seamlessly.
    """

    def __init__(self, max_tokens: int = 500, encoder_model: str = "cl100k_base"):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero.")
            
        self.max_tokens = max_tokens
        self.encoder = tiktoken.get_encoding(encoder_model)

    def count_tokens(self, text: str) -> int:
        """Returns the exact number of tokens for the given text."""
        return len(self.encoder.encode(text))

    @abstractmethod
    def chunk(self, doc: EnrichedDocument) -> List[Chunk]:
        """
        Transforms an enriched document into a sequence of embeddable chunks.
        """
        raise NotImplementedError
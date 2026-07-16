from typing import Type, Dict

from ingestion.chunker.base import BaseChunker
from ingestion.chunker.semantic import SemanticChunker
from ingestion.chunker.recursive import RecursiveChunker
from ingestion.chunker.fixed import FixedChunker
from ingestion.chunker.config import ChunkingConfig


class ChunkerFactory:
    """
    Centralizes the instantiation of chunking strategies based on configuration.
    """
    
    _strategies: Dict[str, Type[BaseChunker]] = {
        "semantic": SemanticChunker,
        "recursive": RecursiveChunker,
        "fixed": FixedChunker
    }

    @classmethod
    def create(cls, config: ChunkingConfig) -> BaseChunker:
        """Instantiates the correct chunker using the provided config."""
        strategy_name = config.strategy.lower().strip()
        chunker_class = cls._strategies.get(strategy_name)
        
        if not chunker_class:
            raise ValueError(
                f"Unsupported chunking strategy: '{strategy_name}'. "
                f"Available strategies: {list(cls._strategies.keys())}."
            )
            
        # Route parameters based on the specific needs of the strategy
        if strategy_name == "semantic":
            return chunker_class(
                max_tokens=config.max_tokens,
                overlap_percent=config.overlap_percent,
                max_context_levels=config.max_context_levels,
                heading_break_threshold=config.heading_break_threshold,
                encoder_model=config.encoder_model
            )
        else:
            return chunker_class(
                max_tokens=config.max_tokens,
                overlap_tokens=config.overlap_tokens,
                encoder_model=config.encoder_model
            )
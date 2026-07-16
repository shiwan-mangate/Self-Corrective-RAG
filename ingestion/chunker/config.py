from pydantic import BaseModel, Field

class ChunkingConfig(BaseModel):
    """
    Centralized configuration for the Chunking Subsystem.
    Provides automatic validation for all tuning parameters.
    """
    strategy: str = Field(
        default="semantic", 
        description="Available: 'semantic', 'recursive', 'fixed'."
    )
    max_tokens: int = Field(
        default=500, 
        gt=0, 
        description="The hard token limit for the embedding model."
    )
    encoder_model: str = Field(default="cl100k_base")

    # --- Semantic Chunker Parameters ---
    overlap_percent: float = Field(default=0.1, ge=0.0, lt=1.0)
    max_context_levels: int = Field(default=2, ge=0)
    heading_break_threshold: float = Field(default=0.2, ge=0.0, lt=1.0)

    # --- Baseline Chunker Parameters (Recursive/Fixed) ---
    overlap_tokens: int = Field(default=50, ge=0)
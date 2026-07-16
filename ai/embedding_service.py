import logging
import torch
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class AIEmbeddingService:
    """
    Core AI utility for vector generation.
    Maintains a singleton-ready lifecycle to prevent expensive reloading of weights.
    Returns native numpy arrays for downstream mathematical efficiency.
    """
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self._load_model()
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Initialized Embedding Model '{self.model_name}' on {self.device} (Dim: {self.dimension})")

    def _load_model(self) -> SentenceTransformer:
        try:
            return SentenceTransformer(self.model_name, device=self.device)
        except Exception as e:
            logger.error(f"Failed to load embedding model '{self.model_name}': {e}")
            raise RuntimeError(f"Embedding model initialization failed: {e}")

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embeds a list of strings and guarantees dimension integrity.
        Returns normalized unit vectors as a numpy array.
        """
        if not texts:
           
            return np.empty((0, self.dimension), dtype=np.float32)

        try:
            vectors: np.ndarray = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
          
            self._validate_vectors(vectors, expected_count=len(texts))
            
            return vectors
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")

    def _validate_vectors(self, vectors: np.ndarray, expected_count: int) -> None:
        """Ensures the mathematical shape of the output perfectly matches the input."""
        if vectors.ndim != 2:
            raise RuntimeError(f"Expected 2D array of vectors, got shape {vectors.shape}")
            
        actual_count, actual_dim = vectors.shape
        
        if actual_count != expected_count:
            raise RuntimeError(f"Count mismatch: sent {expected_count} texts, received {actual_count} vectors.")
            
        if actual_dim != self.dimension:
            raise RuntimeError(f"Dimension mismatch: expected {self.dimension}, received {actual_dim}.")
# ai/embedding_service.py

import os
import logging
import numpy as np
from config.settings import settings
from typing import List
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)

class AIEmbeddingService:
    """
    Core AI utility for vector generation.
    Maintains a singleton-ready lifecycle to prevent expensive reloading of weights.
    Returns native numpy arrays for downstream mathematical efficiency.
    (Refactored to use Hugging Face API to completely eliminate memory/OOM errors).
    """
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = None):
        self.model_name = model_name
        

        self.device = device or "api" 

        self.client = self._load_model()
        

        dummy_vector = self.client.feature_extraction("dummy", model=self.model_name)
        
        if isinstance(dummy_vector[0], list):
            self.dimension = len(dummy_vector[0])
        else:
            self.dimension = len(dummy_vector)
            
        logger.info(f"Initialized Embedding Model '{self.model_name}' via HF API (Dim: {self.dimension})")

    def _load_model(self) -> InferenceClient:
        try:
            
            hf_token = settings.HF_TOKEN
            
            if not hf_token:
                logger.warning("HF_TOKEN not found in settings. Add it to prevent API rate limits.")
                
            return InferenceClient(token=hf_token)
        except Exception as e:
            logger.error(f"Failed to load HF InferenceClient: {e}")
            raise RuntimeError(f"Embedding API client initialization failed: {e}")

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embeds a list of strings and guarantees dimension integrity.
        Returns normalized unit vectors as a numpy array.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        try:
            all_vectors = []
            
         
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
               
                batch_vectors = self.client.feature_extraction(batch, model=self.model_name)
                all_vectors.extend(batch_vectors)
                
            vectors = np.array(all_vectors, dtype=np.float32)
            
            
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vectors = vectors / norms
            
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
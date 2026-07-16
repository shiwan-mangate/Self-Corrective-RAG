import os
import logging
import requests
import numpy as np
from typing import List

logger = logging.getLogger(__name__)


class AIEmbeddingService:
    """
    Core AI utility for vector generation via Hugging Face API.
    Replaces local SentenceTransformers to save massive amounts of RAM.
    Returns native numpy arrays for downstream pgvector efficiency.
    """
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = None):
        self.model_name = model_name
        

        self.dimension = 384 
        
       
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_name}"
        
      
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            logger.warning("HF_TOKEN is missing from environment. API calls may be rate-limited.")

        self.headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
        
        logger.info(f"Initialized API-based Embedding Model '{self.model_name}' (Dim: {self.dimension})")

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embeds a list of strings via API and guarantees dimension integrity.
        Returns normalized unit vectors as a numpy array.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        all_vectors = []
        
        try:
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
               
                response = requests.post(
                    self.api_url, 
                    headers=self.headers, 
                    json={
                        "inputs": batch_texts, 
                        "options": {"wait_for_model": True} 
                    },
                    timeout=30
                )
                
                if response.status_code != 200:
                    raise RuntimeError(f"HF API Error ({response.status_code}): {response.text}")
                    
                all_vectors.extend(response.json())
                
        except Exception as e:
            logger.error(f"API Embedding generation failed: {e}")
            raise RuntimeError(f"API Embedding generation failed: {e}")

     
        vectors = np.array(all_vectors, dtype=np.float32)
        
        
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms) 
        vectors = vectors / norms
        
        self._validate_vectors(vectors, expected_count=len(texts))
        
        return vectors

    def _validate_vectors(self, vectors: np.ndarray, expected_count: int) -> None:
        """Ensures the mathematical shape of the output perfectly matches the input."""
        if vectors.ndim != 2:
            raise RuntimeError(f"Expected 2D array of vectors, got shape {vectors.shape}")
            
        actual_count, actual_dim = vectors.shape
        
        if actual_count != expected_count:
            raise RuntimeError(f"Count mismatch: sent {expected_count} texts, received {actual_count} vectors.")
            
        if actual_dim != self.dimension:
            raise RuntimeError(f"Dimension mismatch: expected {self.dimension}, received {actual_dim}.")
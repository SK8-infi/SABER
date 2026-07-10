import os
import logging
import faiss
import numpy as np
from typing import Tuple

logger = logging.getLogger("saber")

class FAISSIndex:
    """
    Wrapper around the FAISS library for high-performance vector search.
    Supports building a flat index with L2 distance or Cosine Similarity (Inner Product).
    """
    def __init__(self, dimension: int = 384, metric: str = "cosine") -> None:
        """
        Args:
            dimension: Feature dimension of input vectors.
            metric: "cosine" (uses Inner Product flat index) or "l2" (uses Euclidean flat index).
        """
        self.dimension = dimension
        self.metric = metric.lower()
        self.index = None

    def build_index(self, embeddings: np.ndarray) -> None:
        """
        Builds the FAISS index and registers the reference embeddings database.
        
        Args:
            embeddings: Numpy array of shape (N, dimension).
        """
        embeddings = embeddings.astype(np.float32)
        
        if self.metric == "cosine":
            # Normalized vectors with IndexFlatIP yields cosine similarity
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.metric == "l2":
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            raise ValueError(f"Invalid metric: '{self.metric}'. Use 'cosine' or 'l2'.")

        self.index.add(embeddings)
        logger.info(f"Built FAISS {self.metric} index with {self.index.ntotal} items.")

    def save_index(self, path: str) -> None:
        """Saves the active index state to a file."""
        if self.index is None:
            raise ValueError("No active FAISS index to save. Build or load an index first.")
        
        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
            
        faiss.write_index(self.index, path)
        logger.info(f"Successfully saved FAISS index to: {path}")

    def load_index(self, path: str) -> None:
        """Loads a saved FAISS index from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"FAISS index file not found at: {path}")
            
        self.index = faiss.read_index(path)
        self.dimension = self.index.d
        logger.info(f"Loaded FAISS index from: {path} (Dimension: {self.dimension}, Size: {self.index.ntotal})")

    def search(self, query_embeddings: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Searches the index for query matches.
        
        Args:
            query_embeddings: Query vectors of shape (Q, dimension).
            k: Number of nearest neighbors to retrieve.
            
        Returns:
            Tuple of (scores/distances, index_ranks) both of shape (Q, k).
        """
        if self.index is None:
            raise ValueError("FAISS index has not been built or loaded.")
            
        query_embeddings = query_embeddings.astype(np.float32)
        scores, indices = self.index.search(query_embeddings, k)
        return scores, indices

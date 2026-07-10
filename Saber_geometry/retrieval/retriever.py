import numpy as np
from typing import List, Dict, Any, Union
from Saber_geometry.retrieval.faiss_index import FAISSIndex

class Retriever:
    """
    Integrates the raw FAISS Index with gallery item metadata (filenames, labels).
    Converts search rank indices into descriptive dictionaries.
    """
    def __init__(
        self,
        index: FAISSIndex,
        gallery_names: List[str],
        gallery_labels: np.ndarray
    ) -> None:
        """
        Args:
            index: A configured FAISSIndex instance.
            gallery_names: Filenames mapping to the gallery database.
            gallery_labels: Labels array mapping to the gallery database.
        """
        self.index = index
        self.gallery_names = gallery_names
        self.gallery_labels = gallery_labels

    def retrieve(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs search for a query embedding and returns matched items with metadata.
        
        Args:
            query_embedding: Target representation vector of shape (dimension,) or (1, dimension).
            k: Top-K neighbors count.
            
        Returns:
            A list of dictionary objects, each representing a matched image.
        """
        # Ensure query has shape (1, dimension)
        if len(query_embedding.shape) == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)
            
        scores, indices = self.index.search(query_embedding, k=k)
        
        results = []
        for score, rank_idx in zip(scores[0], indices[0]):
            if rank_idx == -1:
                continue  # Skip invalid FAISS match sentinel
                
            results.append({
                "name": self.gallery_names[rank_idx],
                "score": float(score),
                "label": self.gallery_labels[rank_idx]
            })
            
        return results

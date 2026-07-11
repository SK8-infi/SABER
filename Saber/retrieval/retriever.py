import numpy as np
from typing import List, Dict, Any, Optional

class Retriever:
    """
    Integrates the raw FAISS Index with gallery item metadata (filenames, labels).
    Converts search rank indices into descriptive dictionaries.
    Supports top-K Reciprocal Graph Re-ranking.
    """
    def __init__(
        self,
        index: Any,
        gallery_names: List[str],
        gallery_labels: np.ndarray,
        gallery_embeddings: Optional[np.ndarray] = None,
        rerank_enabled: bool = False,
        rerank_shortlist_k: int = 100,
        rerank_neighbor_k: int = 10,
        reciprocal_weight: float = 0.15,
        label_weight: float = 0.05
    ) -> None:
        self.index = index
        self.gallery_names = gallery_names
        self.gallery_labels = gallery_labels
        self.gallery_embeddings = gallery_embeddings
        self.rerank_enabled = rerank_enabled
        
        if rerank_enabled:
            from Saber.retrieval.rerank import ReciprocalReranker
            self.reranker = ReciprocalReranker(
                shortlist_k=rerank_shortlist_k,
                neighbor_k=rerank_neighbor_k,
                reciprocal_weight=reciprocal_weight,
                label_weight=label_weight
            )
        else:
            self.reranker = None

    def retrieve(self, query_embedding: np.ndarray, k: int = 5, uncertainty: float = 0.0) -> List[Dict[str, Any]]:
        """
        Performs search for a query embedding and returns matched items with metadata.
        """
        # Ensure query has shape (1, dimension)
        if len(query_embedding.shape) == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)
            
        if self.rerank_enabled and self.reranker is not None and self.gallery_embeddings is not None:
            # Query the FAISS index with shortlist K
            search_k = max(k, self.reranker.shortlist_k)
            scores, indices = self.index.search(query_embedding, k=search_k)
            
            # Apply graph re-ranking on the shortlist
            refined_scores, refined_indices = self.reranker.rerank(
                query_embedding=query_embedding[0],
                gallery_embeddings=self.gallery_embeddings,
                indices=indices[0],
                scores=scores[0],
                gallery_labels=self.gallery_labels,
                uncertainty=uncertainty,
                final_k=k
            )
            scores = np.expand_dims(refined_scores, axis=0)
            indices = np.expand_dims(refined_indices, axis=0)
        else:
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

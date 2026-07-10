from typing import Optional, Tuple

import numpy as np

from Saber.retrieval.faiss_index import l2_normalize


class ReciprocalReranker:
    """
    Lightweight top-K re-ranker for FAISS shortlists.

    This intentionally runs only on a small shortlist and can be disabled. The
    uncertainty value should be in [0, 1]; higher uncertainty reduces reranking
    influence and returns behavior closer to raw FAISS.
    """

    def __init__(
        self,
        shortlist_k: int = 100,
        neighbor_k: int = 10,
        reciprocal_weight: float = 0.15,
        label_weight: float = 0.05,
    ) -> None:
        self.shortlist_k = shortlist_k
        self.neighbor_k = neighbor_k
        self.reciprocal_weight = reciprocal_weight
        self.label_weight = label_weight

    def rerank(
        self,
        query_embedding: np.ndarray,
        gallery_embeddings: np.ndarray,
        indices: np.ndarray,
        scores: np.ndarray,
        gallery_labels: Optional[np.ndarray] = None,
        uncertainty: float = 0.0,
        final_k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        indices = np.asarray(indices).reshape(-1)
        scores = np.asarray(scores).reshape(-1)
        valid = indices >= 0
        indices = indices[valid][: self.shortlist_k]
        scores = scores[valid][: self.shortlist_k]
        if len(indices) == 0:
            return scores[:final_k], indices[:final_k]

        query = l2_normalize(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1))
        candidates = l2_normalize(gallery_embeddings[indices])
        query_scores = (query @ candidates.T).reshape(-1)

        if len(indices) == 1:
            return query_scores[:final_k], indices[:final_k]

        sim = candidates @ candidates.T
        local_k = min(self.neighbor_k, len(indices) - 1)
        neighbor_order = np.argsort(-sim, axis=1)[:, 1 : local_k + 1]

        top_query_neighbors = set(np.argsort(-query_scores)[:local_k].tolist())
        reciprocal = np.zeros(len(indices), dtype=np.float32)
        for cand_pos in range(len(indices)):
            if cand_pos in top_query_neighbors:
                reciprocal[cand_pos] = 1.0
            reciprocal[cand_pos] += np.mean(query_scores[neighbor_order[cand_pos]])

        label_bonus = np.zeros(len(indices), dtype=np.float32)
        if gallery_labels is not None:
            labels = gallery_labels[indices]
            if labels.ndim == 2:
                overlap = labels.astype(np.float32) @ labels.astype(np.float32).T
                denom = np.maximum(labels.sum(axis=1, keepdims=True), 1.0)
                label_bonus = (overlap / denom).mean(axis=1)
            else:
                label_bonus = np.mean(labels.reshape(-1, 1) == labels.reshape(1, -1), axis=1).astype(np.float32)

        uncertainty = float(np.clip(uncertainty, 0.0, 1.0))
        rerank_strength = 1.0 - uncertainty
        combined = (
            query_scores
            + rerank_strength * self.reciprocal_weight * reciprocal
            + rerank_strength * self.label_weight * label_bonus
        )
        order = np.argsort(-combined)[:final_k]
        return combined[order], indices[order]

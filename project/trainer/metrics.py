import numpy as np
from typing import Dict

def compute_retrieval_metrics(
    similarity_matrix: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    top_k: int = 5,
    is_multilabel: bool = False
) -> Dict[str, float]:
    """
    Computes Precision@K, Recall@K, F1@K, and mAP (mean Average Precision) for retrieval evaluation.
    
    Args:
        similarity_matrix: Cosine similarities of shape (num_queries, num_gallery).
        query_labels: Ground truth labels for queries (single-label indices or multi-hot binary vectors).
        gallery_labels: Ground truth labels for gallery (single-label indices or multi-hot binary vectors).
        top_k: The top K ranking items to retrieve.
        is_multilabel: Set to True for multi-label datasets (BEN-14K) and False for single-label (DSRSID).
        
    Returns:
        Dictionary containing precision@K, recall@K, f1@K, and map@K scores.
    """
    num_queries = similarity_matrix.shape[0]
    
    precisions = []
    recalls = []
    aps = []

    for q_idx in range(num_queries):
        q_label = query_labels[q_idx]

        # Calculate query relevance against all gallery items
        if is_multilabel:
            # Multi-label relevance: intersection of active classes is non-zero
            relevance = (gallery_labels @ q_label) > 0
        else:
            # Single-label relevance: identical class indices
            relevance = (gallery_labels == q_label)

        total_relevant = np.sum(relevance)
        if total_relevant == 0:
            continue

        # Rank similarity indices in descending order
        ranked_indices = np.argsort(-similarity_matrix[q_idx])
        top_k_indices = ranked_indices[:top_k]
        
        # Relevance values of the retrieved top-k items
        retrieved_relevance = relevance[top_k_indices]

        # Precision@K
        num_relevant_retrieved = np.sum(retrieved_relevance)
        precision_val = num_relevant_retrieved / top_k
        precisions.append(precision_val)

        # Recall@K
        recall_val = num_relevant_retrieved / total_relevant
        recalls.append(recall_val)

        # Average Precision (AP) for this query
        ap = 0.0
        relevant_count = 0
        for rank, is_rel in enumerate(retrieved_relevance):
            if is_rel:
                relevant_count += 1
                ap += relevant_count / (rank + 1)
        
        # Normalize AP by the maximum possible relevant retrievals in top K
        denominator = min(total_relevant, top_k)
        ap /= (denominator if denominator > 0 else 1e-8)
        aps.append(ap)

    # Average metrics across all valid queries
    mean_precision = float(np.mean(precisions)) if precisions else 0.0
    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    mean_map = float(np.mean(aps)) if aps else 0.0

    # F1 Score
    if mean_precision + mean_recall > 0:
        mean_f1 = 2.0 * (mean_precision * mean_recall) / (mean_precision + mean_recall)
    else:
        mean_f1 = 0.0

    return {
        f"precision@{top_k}": mean_precision,
        f"recall@{top_k}": mean_recall,
        f"f1@{top_k}": mean_f1,
        f"map@{top_k}": mean_map
    }

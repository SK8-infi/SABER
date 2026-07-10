import numpy as np
from typing import Dict

def compute_retrieval_metrics(
    query_embeds: np.ndarray,
    gallery_embeds: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    top_k: int = 5,
    is_multilabel: bool = False
) -> Dict[str, float]:
    """
    Computes retrieval metrics exactly matching the paper specifications:
    - Multi-label (BEN-14K): Item-level precision, recall, and F1 over top-K, averaged.
    - Single-label (DSRSID): Precision@K and global mAP over the full gallery.
    """
    num_queries = query_embeds.shape[0]
    num_gallery = gallery_embeds.shape[0]
    
    precisions = []
    recalls = []
    f1s = []
    aps = []

    # Precompute a relevance matrix for multi-label (sharing at least one active class)
    # query_labels: (Q, C), gallery_labels: (G, C)
    if is_multilabel:
        relevance_matrix = (query_labels @ gallery_labels.T > 0.5).astype(np.float32)

    for q_idx in range(num_queries):
        q_label = query_labels[q_idx]

        # Compute similarity for this specific query to prevent OOM
        q_sims = (query_embeds[q_idx:q_idx+1] @ gallery_embeds.T).flatten()

        # Rank all similarity indices in descending order
        ranked_indices = np.argsort(-q_sims)
        top_k_indices = ranked_indices[:top_k]

        if is_multilabel:
            # Equation (S2) & (S3) multi-label relevance overlap
            # Active labels of query: q_label is multi-hot vector [0, 1, 0, 1...]
            q_active = set(np.where(q_label > 0.5)[0])
            if len(q_active) == 0:
                continue

            q_prec_k = []
            q_rec_k = []
            q_f1_k = []

            for r_idx in top_k_indices:
                r_label = gallery_labels[r_idx]
                r_active = set(np.where(r_label > 0.5)[0])
                
                intersection = q_active.intersection(r_active)
                num_inter = len(intersection)
                
                p_qr = num_inter / len(r_active) if len(r_active) > 0 else 0.0
                r_qr = num_inter / len(q_active)
                f1_qr = (2 * p_qr * r_qr) / (p_qr + r_qr + 1e-8)
                
                q_prec_k.append(p_qr)
                q_rec_k.append(r_qr)
                q_f1_k.append(f1_qr)

            # Average item-level scores over the top-K retrieved images
            precisions.append(np.mean(q_prec_k))
            recalls.append(np.mean(q_rec_k))
            f1s.append(np.mean(q_f1_k))

            # Compute Average Precision (AP) using fast vectorized operations
            relevance = relevance_matrix[q_idx]
            total_relevant = np.sum(relevance)
            if total_relevant > 0:
                all_relevance = relevance[ranked_indices]
                relevant_ranks = np.where(all_relevance)[0]
                num_rel = len(relevant_ranks)
                if num_rel > 0:
                    pk = np.arange(1, num_rel + 1) / (relevant_ranks + 1)
                    ap = np.sum(pk) / total_relevant
                    aps.append(ap)
        else:
            # Single-label relevance (DSRSID)
            # Relevance defined by single-label class equality
            # query_labels are class index numbers, gallery_labels are class index numbers
            relevance = (gallery_labels == q_label).astype(np.float32)
            total_relevant = np.sum(relevance)
            if total_relevant == 0:
                continue

            # Precision@K: Equation (S4)
            retrieved_relevance = relevance[top_k_indices]
            precision_val = np.sum(retrieved_relevance) / top_k
            precisions.append(precision_val)

            # Global Average Precision (AP) over the full gallery: Equations (S5) & (S6)
            all_relevance = relevance[ranked_indices]
            relevant_ranks = np.where(all_relevance)[0]
            num_rel = len(relevant_ranks)
            if num_rel > 0:
                pk = np.arange(1, num_rel + 1) / (relevant_ranks + 1)
                ap = np.sum(pk) / total_relevant
                aps.append(ap)

    # Average metrics across all valid queries
    mean_precision = float(np.mean(precisions)) if precisions else 0.0
    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    
    if is_multilabel:
        mean_f1 = float(np.mean(f1s)) if f1s else 0.0
        mean_map = float(np.mean(aps)) if aps else 0.0
    else:
        # For single-label, F1 is standard formula from P and R, and map is the average of APs
        mean_f1 = 2.0 * (mean_precision * mean_recall) / (mean_precision + mean_recall + 1e-8)
        mean_map = float(np.mean(aps)) if aps else 0.0

    return {
        f"precision@{top_k}": mean_precision,
        f"recall@{top_k}": mean_recall,
        f"f1@{top_k}": mean_f1,
        f"map@{top_k}": mean_map
    }

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict

def compute_retrieval_metrics(
    query_embeds: np.ndarray,
    gallery_embeds: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    top_k: int = 5,
    is_multilabel: bool = False,
    rerank_config: dict = None,
    query_names: np.ndarray = None,
    gallery_names: np.ndarray = None,
    exclude_self_matches: bool = False
) -> Dict[str, float]:
    """
    Computes retrieval metrics exactly matching the paper specifications.
    Uses GPU-accelerated PyTorch operations if CUDA is available for massive speedups.
    Falls back to CPU/numpy if reranking is enabled.
    """
    rerank_enabled = rerank_config is not None and rerank_config.get("rerank_enabled", False)
    
    # If rerank is enabled, use the original CPU/numpy implementation
    if rerank_enabled:
        return _compute_retrieval_metrics_numpy(
            query_embeds, gallery_embeds, query_labels, gallery_labels,
            top_k, is_multilabel, rerank_config, query_names, gallery_names, exclude_self_matches
        )
        
    # Accelerated PyTorch path (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def to_tensor(x, dtype, device):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        return torch.tensor(x, dtype=dtype, device=device)
    
    q_emb = to_tensor(query_embeds, torch.float32, device)
    g_emb = to_tensor(gallery_embeds, torch.float32, device)
    
    # Compute similarity matrix (Q, G)
    sims = torch.matmul(q_emb, g_emb.t())
    
    # Mask out self-matches if needed
    if exclude_self_matches and query_names is not None and gallery_names is not None:
        mask_cpu = query_names[:, None] != gallery_names[None, :]
        mask = torch.tensor(mask_cpu, dtype=torch.bool, device=device)
        sims = sims.masked_fill(~mask, float('-inf'))
    else:
        mask = None
        
    # Sort similarity scores
    sorted_sims, sorted_indices = torch.sort(sims, dim=1, descending=True)
    top_k_indices = sorted_indices[:, :top_k]
    
    if is_multilabel:
        q_lbl = to_tensor(query_labels, torch.float32, device=device)
        g_lbl = to_tensor(gallery_labels, torch.float32, device=device)
        
        q_active_counts = q_lbl.sum(dim=1, keepdim=True)
        valid_queries = (q_active_counts.squeeze(1) > 0)
        
        if not valid_queries.any():
            return {
                f"precision@{top_k}": 0.0,
                f"recall@{top_k}": 0.0,
                f"f1@{top_k}": 0.0,
                f"map@{top_k}": 0.0
            }
            
        # Gather labels for top-k retrieved gallery items
        retrieved_lbl = g_lbl[top_k_indices] # shape: (Q, top_k, C)
        
        # Calculate intersection counts
        intersection = q_lbl.unsqueeze(1) * retrieved_lbl
        num_inter = intersection.sum(dim=2)
        
        num_ret_active = retrieved_lbl.sum(dim=2)
        
        # Precision, Recall, and F1 per rank per query
        p_qr = num_inter / (num_ret_active + 1e-8)
        r_qr = num_inter / (q_active_counts + 1e-8)
        f1_qr = (2 * p_qr * r_qr) / (p_qr + r_qr + 1e-8)
        
        # Mean scores over top-k per query
        q_prec = p_qr.mean(dim=1)
        q_rec = r_qr.mean(dim=1)
        q_f1 = f1_qr.mean(dim=1)
        
        mean_precision = q_prec[valid_queries].mean().item()
        mean_recall = q_rec[valid_queries].mean().item()
        mean_f1 = q_f1[valid_queries].mean().item()
        
        # Compute Average Precision (AP) for each query over the full gallery ranking
        relevance_matrix = (q_lbl @ g_lbl.t() > 0.5).float()
        if exclude_self_matches and mask is not None:
            relevance_matrix = relevance_matrix.masked_fill(~mask, 0.0)
            
        total_relevant = relevance_matrix.sum(dim=1)
        all_relevance = torch.gather(relevance_matrix, 1, sorted_indices)
        
        aps = []
        for q_idx in range(q_lbl.shape[0]):
            tot_rel = total_relevant[q_idx].item()
            if tot_rel > 0:
                rel_indices = torch.where(all_relevance[q_idx])[0]
                num_rel = len(rel_indices)
                if num_rel > 0:
                    pk = torch.arange(1, num_rel + 1, device=device) / (rel_indices + 1)
                    ap = pk.sum() / tot_rel
                    aps.append(ap.item())
        mean_map = float(np.mean(aps)) if aps else 0.0
    else:
        q_lbl = to_tensor(query_labels, torch.long, device=device)
        g_lbl = to_tensor(gallery_labels, torch.long, device=device)
        
        relevance_matrix = (q_lbl.unsqueeze(1) == g_lbl.unsqueeze(0)).float()
        if exclude_self_matches and mask is not None:
            relevance_matrix = relevance_matrix.masked_fill(~mask, 0.0)
            
        total_relevant = relevance_matrix.sum(dim=1)
        all_relevance = torch.gather(relevance_matrix, 1, sorted_indices)
        top_k_relevance = all_relevance[:, :top_k]
        
        num_hits = top_k_relevance.sum(dim=1)
        precision_val = num_hits / top_k
        
        recall_denominator = torch.clamp(total_relevant, max=top_k)
        recall_val = num_hits / (recall_denominator + 1e-8)
        
        valid_queries = (total_relevant > 0)
        
        if not valid_queries.any():
            return {
                f"precision@{top_k}": 0.0,
                f"recall@{top_k}": 0.0,
                f"f1@{top_k}": 0.0,
                f"map@{top_k}": 0.0
            }
            
        mean_precision = precision_val[valid_queries].mean().item()
        mean_recall = recall_val[valid_queries].mean().item()
        mean_f1 = 2.0 * (mean_precision * mean_recall) / (mean_precision + mean_recall + 1e-8)
        
        aps = []
        for q_idx in range(q_lbl.shape[0]):
            tot_rel = total_relevant[q_idx].item()
            if tot_rel > 0:
                rel_indices = torch.where(all_relevance[q_idx])[0]
                num_rel = len(rel_indices)
                if num_rel > 0:
                    pk = torch.arange(1, num_rel + 1, device=device) / (rel_indices + 1)
                    ap = pk.sum() / tot_rel
                    aps.append(ap.item())
        mean_map = float(np.mean(aps)) if aps else 0.0

    return {
        f"precision@{top_k}": mean_precision,
        f"recall@{top_k}": mean_recall,
        f"f1@{top_k}": mean_f1,
        f"map@{top_k}": mean_map
    }

def _compute_retrieval_metrics_numpy(
    query_embeds: np.ndarray,
    gallery_embeds: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    top_k: int = 5,
    is_multilabel: bool = False,
    rerank_config: dict = None,
    query_names: np.ndarray = None,
    gallery_names: np.ndarray = None,
    exclude_self_matches: bool = False
) -> Dict[str, float]:
    """Original numpy CPU implementation of retrieval metrics (fallback)."""
    num_queries = query_embeds.shape[0]
    precisions = []
    recalls = []
    f1s = []
    aps = []

    from Saber.retrieval.rerank import ReciprocalReranker
    shortlist_k = rerank_config.get("rerank_shortlist_k", 100)
    neighbor_k = rerank_config.get("rerank_neighbor_k", 10)
    reciprocal_weight = rerank_config.get("reciprocal_weight", 0.15)
    label_weight = rerank_config.get("label_weight", 0.05)
    
    reranker = ReciprocalReranker(
        shortlist_k=shortlist_k,
        neighbor_k=neighbor_k,
        reciprocal_weight=reciprocal_weight,
        label_weight=label_weight
    )

    if is_multilabel:
        relevance_matrix = (query_labels @ gallery_labels.T > 0.5).astype(np.float32)

    for q_idx in range(num_queries):
        q_label = query_labels[q_idx]
        q_sims = (query_embeds[q_idx:q_idx+1] @ gallery_embeds.T).flatten()

        if exclude_self_matches and query_names is not None and gallery_names is not None:
            q_name = query_names[q_idx]
            mask = (gallery_names != q_name)
            q_sims_filtered = q_sims[mask]
            gallery_labels_filtered = gallery_labels[mask]
            if is_multilabel:
                relevance_filtered = relevance_matrix[q_idx][mask]
            else:
                relevance_filtered = (gallery_labels_filtered == q_label).astype(np.float32)
            gallery_embeds_filtered = gallery_embeds[mask]
        else:
            q_sims_filtered = q_sims
            gallery_labels_filtered = gallery_labels
            if is_multilabel:
                relevance_filtered = relevance_matrix[q_idx]
            else:
                relevance_filtered = (gallery_labels_filtered == q_label).astype(np.float32)
            gallery_embeds_filtered = gallery_embeds

        ranked_indices = np.argsort(-q_sims_filtered)
        
        shortlist_idx = ranked_indices[:reranker.shortlist_k]
        shortlist_scores = q_sims_filtered[shortlist_idx]
        _, reranked_idx = reranker.rerank(
            query_embedding=query_embeds[q_idx],
            gallery_embeddings=gallery_embeds_filtered,
            indices=shortlist_idx,
            scores=shortlist_scores,
            gallery_labels=gallery_labels_filtered,
            uncertainty=0.0,
            final_k=top_k
        )
        top_k_indices = reranked_idx

        if is_multilabel:
            q_active = set(np.where(q_label > 0.5)[0])
            if len(q_active) == 0:
                continue

            q_prec_k = []
            q_rec_k = []
            q_f1_k = []

            for r_idx in top_k_indices:
                r_label = gallery_labels_filtered[r_idx]
                r_active = set(np.where(r_label > 0.5)[0])
                
                intersection = q_active.intersection(r_active)
                num_inter = len(intersection)
                
                p_qr = num_inter / len(r_active) if len(r_active) > 0 else 0.0
                r_qr = num_inter / len(q_active)
                f1_qr = (2 * p_qr * r_qr) / (p_qr + r_qr + 1e-8)
                
                q_prec_k.append(p_qr)
                q_rec_k.append(r_qr)
                q_f1_k.append(f1_qr)

            precisions.append(np.mean(q_prec_k))
            recalls.append(np.mean(q_rec_k))
            f1s.append(np.mean(q_f1_k))

            relevance = relevance_filtered
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
            relevance = relevance_filtered
            total_relevant = np.sum(relevance)
            if total_relevant == 0:
                continue

            retrieved_relevance = relevance[top_k_indices]
            num_hits = np.sum(retrieved_relevance)
            precision_val = num_hits / top_k
            precisions.append(precision_val)

            recall_denominator = min(total_relevant, top_k)
            recall_val = num_hits / recall_denominator
            recalls.append(recall_val)

            all_relevance = relevance[ranked_indices]
            relevant_ranks = np.where(all_relevance)[0]
            num_rel = len(relevant_ranks)
            if num_rel > 0:
                pk = np.arange(1, num_rel + 1) / (relevant_ranks + 1)
                ap = np.sum(pk) / total_relevant
                aps.append(ap)

    mean_precision = float(np.mean(precisions)) if precisions else 0.0
    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    
    if is_multilabel:
        mean_f1 = float(np.mean(f1s)) if f1s else 0.0
        mean_map = float(np.mean(aps)) if aps else 0.0
    else:
        mean_f1 = 2.0 * (mean_precision * mean_recall) / (mean_precision + mean_recall + 1e-8)
        mean_map = float(np.mean(aps)) if aps else 0.0



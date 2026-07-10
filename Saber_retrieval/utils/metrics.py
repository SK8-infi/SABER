from typing import Dict

import numpy as np


def metrics_from_ranked_indices(
    ranked_indices: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    top_k: int = 5,
    is_multilabel: bool = False,
) -> Dict[str, float]:
    precisions = []
    recalls = []
    f1s = []
    aps = []

    for q_idx, ranked in enumerate(ranked_indices):
        ranked = ranked[ranked >= 0]
        top = ranked[:top_k]
        if len(top) == 0:
            continue

        q_label = query_labels[q_idx]
        if is_multilabel:
            q_active = set(np.where(q_label > 0.5)[0])
            if not q_active:
                continue
            q_prec, q_rec, q_f1 = [], [], []
            relevance = []
            top_set = set(top.tolist())
            for idx in ranked:
                r_active = set(np.where(gallery_labels[idx] > 0.5)[0])
                inter = len(q_active.intersection(r_active))
                relevance.append(inter > 0)
                if idx in top_set:
                    p = inter / len(r_active) if r_active else 0.0
                    r = inter / len(q_active)
                    q_prec.append(p)
                    q_rec.append(r)
                    q_f1.append((2.0 * p * r) / (p + r + 1e-8))
            precisions.append(float(np.mean(q_prec)))
            recalls.append(float(np.mean(q_rec)))
            f1s.append(float(np.mean(q_f1)))
            relevance = np.asarray(relevance, dtype=bool)
            total_relevant = int(np.sum(relevance))
            if total_relevant > 0:
                relevant_ranks = np.where(relevance)[0]
                pk = np.arange(1, len(relevant_ranks) + 1) / (relevant_ranks + 1)
                aps.append(float(np.sum(pk) / total_relevant))
        else:
            relevance = gallery_labels[ranked] == q_label
            top_rel = gallery_labels[top] == q_label
            precisions.append(float(np.sum(top_rel) / top_k))
            total_relevant = int(np.sum(gallery_labels == q_label))
            recalls.append(float(np.sum(top_rel) / max(total_relevant, 1)))
            relevant_ranks = np.where(relevance)[0]
            if total_relevant > 0 and len(relevant_ranks) > 0:
                pk = np.arange(1, len(relevant_ranks) + 1) / (relevant_ranks + 1)
                aps.append(float(np.sum(pk) / total_relevant))

    mean_precision = float(np.mean(precisions)) if precisions else 0.0
    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    mean_f1 = float(np.mean(f1s)) if is_multilabel and f1s else (2.0 * mean_precision * mean_recall) / (mean_precision + mean_recall + 1e-8)
    mean_map = float(np.mean(aps)) if aps else 0.0
    return {
        f"precision@{top_k}": mean_precision,
        f"recall@{top_k}": mean_recall,
        f"f1@{top_k}": mean_f1,
        f"map@{top_k}": mean_map,
    }

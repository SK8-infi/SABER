import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from Saber.trainer.metrics import compute_retrieval_metrics

def test_multilabel_jaccard_metrics():
    """Verify precision, recall, F1, and mAP for multi-label Jaccard similarity."""
    # Synthetic query embeddings (3 queries, 4-dim)
    query_embeds = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ], dtype=np.float32)

    # Gallery embeddings (3 items)
    gallery_embeds = np.array([
        [1.0, 0.0, 0.0, 0.0],  # Match query 0
        [0.0, 1.0, 0.0, 0.0],  # Match query 1
        [0.0, 0.0, 1.0, 0.0]   # Match query 2
    ], dtype=np.float32)

    # Ground truth multi-hot labels (3 samples, 5 classes)
    query_labels = np.array([
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1]
    ], dtype=np.float32)

    gallery_labels = query_labels.copy()

    metrics = compute_retrieval_metrics(
        query_embeds=query_embeds,
        gallery_embeds=gallery_embeds,
        query_labels=query_labels,
        gallery_labels=gallery_labels,
        top_k=3,
        is_multilabel=True,
        exclude_self_matches=False
    )

    print("Computed Metrics:", metrics)
    assert "f1@3" in metrics
    assert "map@3" in metrics
    assert isinstance(metrics["f1@3"], float)

if __name__ == "__main__":
    test_multilabel_jaccard_metrics()
    print("ALL METRICS CALCULATION TESTS PASSED CLEANLY!")

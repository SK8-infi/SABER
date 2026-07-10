import os
import logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger("saber")

def plot_umap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    n_neighbors: int = 15,
    min_dist: float = 0.1
) -> None:
    """
    Computes and plots a 2D UMAP representation of image embeddings.
    Falls back gracefully to PCA if UMAP or its dependencies fail to load.
    
    Args:
        embeddings: Feature matrix of shape (N, dimension).
        labels: Class labels of shape (N,) or binary multi-hot vectors of shape (N, classes).
        save_path: Location where the resulting plot will be saved.
        n_neighbors: UMAP graph construction neighbor count.
        min_dist: UMAP minimum distance packing constraint.
    """
    if len(labels.shape) == 2:
        color_labels = np.argmax(labels, axis=1)
    else:
        color_labels = labels

    try:
        import umap
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
        title = "UMAP Clustering of Extracted Embeddings"
        xlabel, ylabel = "UMAP Dimension 1", "UMAP Dimension 2"
    except Exception as e:
        logger.warning(f"Could not compute UMAP projection ({e}). Falling back to PCA...")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)
        title = "PCA Projection of Extracted Embeddings (UMAP Fallback)"
        xlabel, ylabel = "PCA Component 1", "PCA Component 2"

    # Plot
    plt.figure(figsize=(10, 8), dpi=150)
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=color_labels,
        cmap="tab20",
        alpha=0.8,
        edgecolors="none",
        s=25
    )
    plt.colorbar(scatter, label="Class (Argmax for Multi-label)")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

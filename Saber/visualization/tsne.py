import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    perplexity: int = 30,
    n_iter: int = 1000
) -> None:
    """
    Computes and plots a 2D t-SNE representation of image embeddings.
    
    Args:
        embeddings: Feature matrix of shape (N, dimension).
        labels: Class labels of shape (N,) or binary multi-hot vectors of shape (N, classes).
        save_path: Location where the resulting plot will be saved.
        perplexity: t-SNE perplexity parameter.
        n_iter: Number of optimization iterations.
    """
    # Convert multi-label to single-label representation for visual grouping
    if len(labels.shape) == 2:
        color_labels = np.argmax(labels, axis=1)
    else:
        color_labels = labels

    # Apply t-SNE dimensionality reduction (supports scikit-learn API changes)
    import inspect
    sig = inspect.signature(TSNE.__init__)
    if "max_iter" in sig.parameters:
        tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=42)
    else:
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

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
    plt.title("t-SNE Visual Clustering of Extracted Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True, linestyle="--", alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

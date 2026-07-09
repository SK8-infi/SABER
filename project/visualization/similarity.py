import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List

def plot_similarity_matrix(
    similarity_matrix: np.ndarray,
    save_path: str,
    title: str = "Query-Gallery Cosine Similarity Heatmap"
) -> None:
    """
    Plots a heatmap representing the similarity matrix between queries and gallery items.
    
    Args:
        similarity_matrix: Distance/Similarity matrix of shape (num_queries, num_gallery).
        save_path: Filepath where the heatmap will be saved.
        title: Title of the heatmap.
    """
    plt.figure(figsize=(10, 8), dpi=150)
    plt.imshow(similarity_matrix, cmap="viridis", aspect="auto")
    plt.colorbar(label="Cosine Similarity Score")
    plt.title(title)
    plt.xlabel("Gallery Items (Database)")
    plt.ylabel("Query Items")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_retrieval_grid(
    query_img: np.ndarray,
    retrieved_imgs: List[np.ndarray],
    scores: List[float],
    names: List[str],
    save_path: str
) -> None:
    """
    Generates a horizontal grid showing the query image followed by top-k matches.
    Handles adapting variable channels (SAR, Multi-spectral, PAN) to 3-channel RGB for display.
    
    Args:
        query_img: Query image array of shape (H, W, C) or (C, H, W).
        retrieved_imgs: List of gallery match image arrays of shape (H, W, C).
        scores: Similarity score values of matches.
        names: Filenames of matched items.
        save_path: Target save location.
    """
    num_matches = len(retrieved_imgs)
    fig, axes = plt.subplots(1, 1 + num_matches, figsize=(3 * (1 + num_matches), 3), dpi=150)

    # Standardize matplotlib array input formats to (H, W, 3)
    def to_plottable_rgb(img: np.ndarray) -> np.ndarray:
        # Move channel dimension if it's (C, H, W)
        if img.shape[0] in [1, 2, 4, 12] and img.shape[2] not in [1, 2, 3, 4]:
            img = np.moveaxis(img, 0, -1)
            
        h, w, c = img.shape
        if c > 3:
            # For multi-spectral, grab the first 3 channels (e.g. RGB subset)
            rgb = img[..., :3]
        elif c == 1:
            # Grayscale -> RGB
            rgb = np.repeat(img, 3, axis=-1)
        elif c == 2:
            # Sentinel-1 (VV, VH) -> Pad 3rd channel with zeros
            pad = np.zeros((h, w, 1), dtype=img.dtype)
            rgb = np.concatenate([img, pad], axis=-1)
        else:
            rgb = img

        # Normalize values to [0.0, 1.0] for plotting safety
        img_min = rgb.min()
        img_max = rgb.max()
        if img_max - img_min > 0:
            rgb = (rgb - img_min) / (img_max - img_min)
            
        return rgb

    # Plot Query
    axes[0].imshow(to_plottable_rgb(query_img))
    axes[0].set_title("Query Image\n(Input)", fontsize=9, fontweight="bold")
    axes[0].axis("off")

    # Plot Matches
    for idx in range(num_matches):
        ax = axes[idx + 1]
        ax.imshow(to_plottable_rgb(retrieved_imgs[idx]))
        clean_name = os.path.basename(names[idx])
        ax.set_title(f"Match #{idx+1}\nSim: {scores[idx]:.4f}\n{clean_name}", fontsize=8)
        ax.axis("off")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

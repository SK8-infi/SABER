import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class RetrievalHead(nn.Module):
    """
    Prepares representation embeddings for FAISS indexing and retrieval search.
    Performs optional linear projection and L2 normalization.
    """
    def __init__(self, dim: int = 384, out_dim: Optional[int] = None, normalize: bool = True) -> None:
        """
        Args:
            dim: Input embedding dimension (from Predictor/Projection Head).
            out_dim: Optional projection output dimension. If None, uses dim.
            normalize: If True, applies L2 normalization (enabling Cosine Similarity).
        """
        super().__init__()
        self.normalize = normalize
        
        if out_dim is not None and out_dim != dim:
            self.proj = nn.Linear(dim, out_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings of shape (B, dim) or (B, L, dim).
            
        Returns:
            Normalized and adapted retrieval embeddings of shape (B, out_dim).
        """
        x = self.proj(x)
        if self.normalize:
            # Normalize along the last feature dimension
            x = F.normalize(x, p=2, dim=-1, eps=1e-4)
        return x

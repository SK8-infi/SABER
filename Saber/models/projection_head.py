import torch
import torch.nn as nn

class ProjectionHead(nn.Module):
    """
    Projection Head module:
    A two-layer MLP with LayerNorm, GELU activation, and a residual shortcut.
    Maps high-dimensional ViT features (e.g. 768-D) to projection space (e.g. 384/768-D).
    """
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 384) -> None:
        """
        Args:
            in_dim: Input dimension from the ViT backbone.
            hidden_dim: Dimension of the hidden layer.
            out_dim: Target output dimension (e.g., 384 or 768).
        """
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.gelu1 = nn.GELU()

        self.fc2 = nn.Linear(hidden_dim, out_dim)

        # Residual shortcut mapping if input and output dimensions differ
        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, in_dim).

        Returns:
            Projected representation tensor of shape (B, out_dim).
        """
        out = self.fc2(self.gelu1(self.ln(self.fc1(x))))
        res = self.shortcut(x)
        return out + res

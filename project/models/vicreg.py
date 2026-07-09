import torch
import torch.nn as nn
from project.models.projection_head import ProjectionHead

class VICRegProjectionHead(nn.Module):
    """
    VICReg projection wrapper.
    Responsible for projecting the encoder outputs (from the frozen ViT backbone)
    into a representation space where the VICReg loss terms are applied.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 384) -> None:
        """
        Args:
            in_dim: Input dimension from the ViT backbone.
            hidden_dim: Hidden dimension size.
            out_dim: Target output dimension (e.g., 384).
        """
        super().__init__()
        self.projector = ProjectionHead(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor of shape (B, in_dim).
            
        Returns:
            Projected representation tensor of shape (B, out_dim).
        """
        return self.projector(x)

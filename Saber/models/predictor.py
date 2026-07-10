import torch
import torch.nn as nn
from typing import Optional

class MLPPredictor(nn.Module):
    """
    MLP-based Predictor for JEPA.
    Maps context embeddings to target embeddings with residual connections.
    """
    def __init__(self, dim: int = 384, hidden_dim: int = 512, num_layers: int = 2) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        layers = []
        in_d = dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_d, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ])
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, dim))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TransformerPredictor(nn.Module):
    """
    Transformer-based Predictor for JEPA.
    Uses multi-head self-attention to contextualize representations.
    """
    def __init__(
        self,
        dim: int = 384,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 512
    ) -> None:
        super().__init__()
        self.dim = dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check if input is a 2D tensor (B, D) or 3D tensor (B, L, D)
        is_global = (len(x.shape) == 2)
        if is_global:
            # Add a dummy sequence dimension: (B, 1, D)
            x = x.unsqueeze(1)
            
        out = self.transformer(x)
        
        if is_global:
            out = out.squeeze(1)
        return out


class Predictor(nn.Module):
    """
    Configurable Predictor module that can switch between MLP and Transformer architectures.
    """
    def __init__(
        self,
        predictor_type: str = "mlp",
        dim: int = 384,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 4
    ) -> None:
        """
        Args:
            predictor_type: Type of architecture. Either "mlp" or "transformer".
            dim: Dimension of input and output embeddings.
            hidden_dim: Hidden dimension size.
            num_layers: Number of layers.
            num_heads: Number of attention heads (for transformer).
        """
        super().__init__()
        self.predictor_type = predictor_type.lower()
        
        if self.predictor_type == "mlp":
            self.predictor = MLPPredictor(dim=dim, hidden_dim=hidden_dim, num_layers=num_layers)
        elif self.predictor_type == "transformer":
            self.predictor = TransformerPredictor(
                dim=dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dim_feedforward=hidden_dim
            )
        else:
            raise ValueError(f"Invalid predictor_type: '{predictor_type}'. Use 'mlp' or 'transformer'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings of shape (B, dim) or (B, L, dim).
            
        Returns:
            Predicted embeddings of shape (B, dim) or (B, L, dim).
        """
        return self.predictor(x)

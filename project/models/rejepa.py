import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from project.models.input_adapter import InputAdapter
from project.models.backbone import FrozenViTBackbone
from project.models.projection_head import ProjectionHead
from project.models.predictor import Predictor
from project.models.retrieval_head import RetrievalHead

class REJEPA(nn.Module):
    """
    REJEPA: Remote Sensing Joint Embedding Predictive Architecture model.
    Binds the input adapter, frozen ViT backbone, projection head, predictor, and retrieval head.
    Can be replaced in configs with a future SABER model without altering the trainer/retriever.
    """
    def __init__(self, config: Dict[str, Any], in_channels: int) -> None:
        """
        Args:
            config: Model configuration sub-dictionary.
            in_channels: Number of input channels (e.g. 2, 4, 12).
        """
        super().__init__()
        self.config = config
        self.in_channels = in_channels

        # 1. Input Adapter
        self.adapter = InputAdapter(
            in_channels=in_channels,
            adapter_type=config.model.input_adapter_type
        )

        # 2. Frozen Backbone
        self.backbone = FrozenViTBackbone(
            model_name=config.model.backbone_name,
            pretrained=config.model.pretrained
        )

        # 3. Projection Head
        self.projection_head = ProjectionHead(
            in_dim=self.backbone.embed_dim,
            hidden_dim=config.model.projection_head.hidden_dim,
            out_dim=config.model.projection_head.out_dim
        )

        # 4. Predictor
        self.predictor = Predictor(
            predictor_type=config.model.predictor.type,
            dim=config.model.projection_head.out_dim,
            hidden_dim=config.model.predictor.hidden_dim,
            num_layers=config.model.predictor.num_layers,
            num_heads=config.model.predictor.num_heads
        )

        # 5. Retrieval Head
        self.retrieval_head = RetrievalHead(
            dim=config.model.projection_head.out_dim,
            out_dim=None,  # Keeps standard embedding size
            normalize=True  # L2 normalization for Cosine Similarity search
        )

    def forward(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.
        In training, expects two views: x1 (context view) and x2 (target view).
        In inference, works with a single view x1.
        
        Args:
            x1: Context view tensor of shape (B, C, H, W).
            x2: Optional target view tensor of shape (B, C, H, W).
            
        Returns:
            If target view x2 is provided (training):
                Returns tuple (z1, z2, z1_pred) where:
                    z1: projection of context view x1.
                    z2: projection of target view x2.
                    z1_pred: predicted target embedding from context z1.
            If target view is NOT provided (inference):
                Returns predicted embedding z1_pred.
        """
        # Run context view through input adapter, backbone, and projection head
        feats1 = self.backbone(self.adapter(x1))
        z1 = self.projection_head(feats1)
        
        # Run through predictor
        z1_pred = self.predictor(z1)

        if x2 is not None:
            # Run target view through same path to obtain target projection
            feats2 = self.backbone(self.adapter(x2))
            z2 = self.projection_head(feats2)
            return z1, z2, z1_pred
            
        return z1_pred

    def get_retrieval_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes L2-normalized embeddings for indexing and retrieval.
        Corresponds to: Input Adapter -> Frozen ViT -> Projection Head -> Predictor -> Retrieval Head.
        
        Args:
            x: Input image tensor of shape (B, C, H, W).
            
        Returns:
            Normalized retrieval embeddings of shape (B, out_dim).
        """
        # Set modules to evaluation mode
        self.eval()
        with torch.no_grad():
            z_pred = self.forward(x)
            embeddings = self.retrieval_head(z_pred)
        return embeddings

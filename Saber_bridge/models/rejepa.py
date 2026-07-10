import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from Saber_bridge.models.input_adapter import InputAdapter
from Saber_bridge.models.backbone import FrozenViTBackbone
from Saber_bridge.models.projection_head import ProjectionHead
from Saber_bridge.models.predictor import Predictor
from Saber_bridge.models.retrieval_head import RetrievalHead

class REJEPA(nn.Module):
    """
    REJEPA: Remote Sensing Joint Embedding Predictive Architecture model.
    Binds the input adapter, frozen ViT backbone, projection head, predictor, and retrieval head.
    Supports same-modal (single adapter) and cross-modal (dual adapters for S1/S2) routing.
    """
    def __init__(self, config: Dict[str, Any], in_channels: int) -> None:
        """
        Args:
            config: Model configuration sub-dictionary.
            in_channels: Number of input channels (e.g. 2, 4, 12, 14).
        """
        super().__init__()
        self.config = config
        self.in_channels = in_channels

        # 1. Input Adapter
        if in_channels == 14:
            # Dual-modality cross-modal: S1 (2 channels) and S2 (12 channels)
            self.adapter_s1 = InputAdapter(in_channels=2, adapter_type=config.model.input_adapter_type)
            self.adapter_s2 = InputAdapter(in_channels=12, adapter_type=config.model.input_adapter_type)
        else:
            # Same-modality
            self.adapter = InputAdapter(in_channels=in_channels, adapter_type=config.model.input_adapter_type)

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
        In training, x1 is context view, x2 is target view.
        """
        if self.in_channels == 14:
            # Split concatenated 14 channels into S1 context (first 2) and S2 target (remaining 12)
            if x1.shape[1] == 14:
                x_s1 = x1[:, :2, :, :]
                target_tensor = x2 if x2 is not None else x1
                x_s2 = target_tensor[:, 2:, :, :]
            else:
                x_s1 = x1
                x_s2 = x2
                
            feats1 = self.backbone(self.adapter_s1(x_s1))
            z1 = self.projection_head(feats1)
            z1_pred = self.predictor(z1)
            
            if x_s2 is not None:
                feats2 = self.backbone(self.adapter_s2(x_s2))
                z2 = self.projection_head(feats2)
                return z1, z2, z1_pred
            return z1_pred
        else:
            # Same-modality path
            feats1 = self.backbone(self.adapter(x1))
            z1 = self.projection_head(feats1)
            z1_pred = self.predictor(z1)
            
            if x2 is not None:
                feats2 = self.backbone(self.adapter(x2))
                z2 = self.projection_head(feats2)
                return z1, z2, z1_pred
            return z1_pred

    def get_retrieval_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes L2-normalized embeddings for indexing and retrieval.
        """
        self.eval()
        with torch.no_grad():
            if self.in_channels == 14:
                if x.shape[1] == 14:
                    # Default: extract target S2 features for gallery
                    x_target = x[:, 2:, :, :]
                    feats = self.backbone(self.adapter_s2(x_target))
                elif x.shape[1] == 12:
                    feats = self.backbone(self.adapter_s2(x))
                elif x.shape[1] == 2:
                    # For S1 query, project it and run predictor to align with target space
                    feats = self.backbone(self.adapter_s1(x))
                    z = self.projection_head(feats)
                    z_pred = self.predictor(z)
                    return self.retrieval_head(z_pred)
                else:
                    raise ValueError(f"Unexpected channel dimension in cross-modal retrieval: {x.shape}")
                
                z = self.projection_head(feats)
                embeddings = self.retrieval_head(z)
            else:
                z_pred = self.forward(x)
                embeddings = self.retrieval_head(z_pred)
        return embeddings

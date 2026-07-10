import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional, List
from peft import LoraConfig, get_peft_model
from Saber_dofa.models.backbone import FrozenDOFABackbone
from Saber_dofa.models.projection_head import ProjectionHead
from Saber_dofa.models.predictor import Predictor
from Saber_dofa.models.retrieval_head import RetrievalHead

logger = logging.getLogger("saber")

class SABER(nn.Module):
    """
    SABER: Sensor-Agnostic Bridged Embedding Retrieval model.
    Utilizes a frozen DOFA backbone adapted via parameter-efficient LoRA.
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

        # Wavelengths in micrometers (Sentinel-1 Radar uses C-band 5.405 GHz frequency representation in DOFA)
        self.s1_wvs = [5.405, 5.405]
        # Sentinel-2: B1 (0.443), B2 (0.490), B3 (0.560), B4 (0.665), B5 (0.705), B6 (0.740), B7 (0.783), B8 (0.842), B8A (0.865), B9 (0.945), B11 (1.610), B12 (2.190)
        self.s2_wvs = [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.945, 1.610, 2.190]

        # 1. Wavelength-Conditioned Foundation Backbone
        self.backbone = FrozenDOFABackbone(pretrained=config.model.pretrained)

        # 2. LoRA Adaptation on Attention Projections (qkv)
        # Freeze backbone parameters
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Apply LoRA specifically to timm ViT attention projection blocks
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["qkv"],  # Applies to attention weights in vit blocks
            lora_dropout=0.1,
            bias="none"
        )
        self.backbone.model = get_peft_model(self.backbone.model, lora_config)
        logger.info("Successfully wrapped DOFA ViT blocks with LoRA adapters.")
        self.backbone.model.print_trainable_parameters()

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
            out_dim=None,
            normalize=True
        )

    def _get_wvs_for_channels(self, num_channels: int) -> List[float]:
        if num_channels == 2:
            return self.s1_wvs
        elif num_channels == 12:
            return self.s2_wvs
        else:
            raise ValueError(f"Unsupported channel dimension: {num_channels}")

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
                
            # Embed SAR S1
            feats1 = self.backbone(x_s1, self.s1_wvs)
            z1 = self.projection_head(feats1)
            z1_pred = self.predictor(z1)
            
            if x_s2 is not None:
                # Embed Optical S2
                feats2 = self.backbone(x_s2, self.s2_wvs)
                z2 = self.projection_head(feats2)
                return z1, z2, z1_pred
            return z1_pred
        else:
            # Same-modality path
            wvs = self._get_wvs_for_channels(self.in_channels)
            feats1 = self.backbone(x1, wvs)
            z1 = self.projection_head(feats1)
            z1_pred = self.predictor(z1)
            
            if x2 is not None:
                feats2 = self.backbone(x2, wvs)
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
                    feats = self.backbone(x_target, self.s2_wvs)
                elif x.shape[1] == 12:
                    feats = self.backbone(x, self.s2_wvs)
                elif x.shape[1] == 2:
                    # For S1 query, project it and run predictor to align with target space
                    feats = self.backbone(x, self.s1_wvs)
                    z = self.projection_head(feats)
                    z_pred = self.predictor(z)
                    return self.retrieval_head(z_pred)
                else:
                    raise ValueError(f"Unexpected channel dimension in cross-modal retrieval: {x.shape}")
                
                z = self.projection_head(feats)
                embeddings = self.retrieval_head(z)
            else:
                wvs = self._get_wvs_for_channels(self.in_channels)
                feats = self.backbone(x, wvs)
                z = self.projection_head(feats)
                embeddings = self.retrieval_head(z)
        return embeddings

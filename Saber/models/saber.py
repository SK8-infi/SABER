import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional, List
from peft import LoraConfig, get_peft_model
from Saber.models.backbone import FrozenDOFABackbone
from Saber.models.projection_head import ProjectionHead
from Saber.models.predictor import Predictor
from Saber.models.retrieval_head import RetrievalHead
from Saber.models.bridge import CFMBridge, CFMBridgeWrapper
from Saber.models.hashing_head import HashingHead

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

        if self.in_channels in [14, 12, 2]:
            self.s1_channels = 2
            self.s2_channels = 12
            self.s1_wvs = [5.405, 5.405]
            self.s2_wvs = [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.945, 1.610, 2.190]
        elif self.in_channels in [5, 4, 1]:
            self.s1_channels = 1
            self.s2_channels = 4
            self.s1_wvs = [0.675]
            self.s2_wvs = [0.485, 0.555, 0.660, 0.830]
        else:
            self.s1_channels = 0
            self.s2_channels = 0
            self.s1_wvs = []
            self.s2_wvs = []

        # 1. Wavelength-Conditioned Foundation Backbone
        self.backbone = FrozenDOFABackbone(pretrained=config.model.pretrained)

        # 2. LoRA Adaptation on Attention Projections (qkv)
        # Freeze backbone parameters
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Apply LoRA specifically to timm ViT attention projection blocks and MLP blocks
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["qkv", "fc1", "fc2"],  # Applies to attention and MLP weights in vit blocks
            lora_dropout=0.05,
            bias="none"
        )
        self.backbone.model = get_peft_model(self.backbone.model, lora_config)
        logger.info("Successfully wrapped DOFA ViT blocks with LoRA adapters (Rank 16, Target: qkv, fc1, fc2).")
        self.backbone.model.print_trainable_parameters()

        # 3. Projection Head
        if self.in_channels in [14, 5]:
            self.s1_projection = ProjectionHead(
                in_dim=self.backbone.embed_dim,
                hidden_dim=config.model.projection_head.hidden_dim,
                out_dim=config.model.projection_head.out_dim
            )
            self.s2_projection = ProjectionHead(
                in_dim=self.backbone.embed_dim,
                hidden_dim=config.model.projection_head.hidden_dim,
                out_dim=config.model.projection_head.out_dim
            )
            self.projection_head = self.s2_projection  # Fallback reference
        else:
            self.projection_head = ProjectionHead(
                in_dim=self.backbone.embed_dim,
                hidden_dim=config.model.projection_head.hidden_dim,
                out_dim=config.model.projection_head.out_dim
            )
            self.s1_projection = self.projection_head
            self.s2_projection = self.projection_head

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

        # 5b. Multi-Label Classification Head
        self.classifier = nn.Linear(config.model.projection_head.out_dim, 19)

        # 6. Optional CFM Latent Bridge (Dev 2)
        if config.get("bridge", {}).get("enabled", False):
            bridge_net = CFMBridge(
                dim=config.model.projection_head.out_dim,
                hidden_dim=config.bridge.get("hidden_dim", 768),
                num_blocks=config.bridge.get("num_blocks", 5),
                dropout=config.bridge.get("dropout", 0.1)
            )
            self.bridge = CFMBridgeWrapper(bridge_net, ode_steps=config.bridge.get("ode_steps", 10))
            logger.info("Successfully instantiated CFM Latent Bridge wrapper inside SABER.")
        else:
            self.bridge = None

        # 7. Optional Hashing Head (Dev 4)
        if config.get("hashing", {}).get("enabled", False):
            self.hashing_head = HashingHead(
                in_dim=config.model.projection_head.out_dim,
                num_bits=config.hashing.get("num_bits", 256),
                hidden_dim=config.hashing.get("hidden_dim", 512)
            )
            logger.info(f"Successfully instantiated HashingHead ({config.hashing.num_bits}-bit) inside SABER.")
        else:
            self.hashing_head = None

        # Attributes to cache soft codes during forward pass (for joint hashing loss)
        self.soft_codes1 = None
        self.soft_codes2 = None

    def _get_wvs_for_channels(self, num_channels: int) -> List[float]:
        if num_channels == 1:
            return [0.675] # Gaofen-1 Panchromatic central wavelength
        elif num_channels == 2:
            return self.s1_wvs # Sentinel-1 SAR
        elif num_channels == 4:
            return [0.485, 0.555, 0.660, 0.830] # Gaofen-1 Multispectral central wavelengths
        elif num_channels == 12:
            return self.s2_wvs # Sentinel-2 MS
        elif num_channels == 14:
            return self.s1_wvs + self.s2_wvs # Concatenated Sentinel-1 + Sentinel-2
        else:
            raise ValueError(f"Unsupported channel dimension: {num_channels}")

    def forward(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.
        In training, x1 is context view, x2 is target view.
        """
        if self.in_channels in [14, 5]:
            # Cross-modal path (Sentinel-1 and Sentinel-2)
            if x1.shape[1] == self.in_channels:
                x_s1 = x1[:, :self.s1_channels, :, :]
                target_tensor = x2 if x2 is not None else x1
                x_s2 = target_tensor[:, self.s1_channels:, :, :]
            else:
                x_s1 = x1
                x_s2 = x2

            # 1. Project context S1
            feats1 = self.backbone(x_s1, self.s1_wvs)
            z1 = self.s1_projection(feats1)
            z1_pred = self.predictor(z1)
            
            # Classification logits
            logits_s1 = self.classifier(z1)

            if x_s2 is not None:
                # 2. Project target S2
                feats2 = self.backbone(x_s2, self.s2_wvs)
                z2 = self.s2_projection(feats2)
                
                logits_s2 = self.classifier(z2)
                
                # Cache soft codes if hashing head is present
                if self.hashing_head is not None:
                    self.soft_codes1 = self.hashing_head(z1)
                    self.soft_codes2 = self.hashing_head(z2)
                    
                return z1, z2, z1_pred, logits_s1, logits_s2
            return z1_pred, logits_s1
        else:
            # Same-modality path
            wvs = self._get_wvs_for_channels(self.in_channels)
            feats1 = self.backbone(x1, wvs)
            z1 = self.projection_head(feats1)
            z1_pred = self.predictor(z1)
            
            logits_s1 = self.classifier(z1)
            
            if x2 is not None:
                feats2 = self.backbone(x2, wvs)
                z2 = self.projection_head(feats2)
                
                logits_s2 = self.classifier(z2)
                
                # Cache soft codes if hashing head is present
                if self.hashing_head is not None:
                    self.soft_codes1 = self.hashing_head(z1)
                    self.soft_codes2 = self.hashing_head(z2)
                    
                return z1, z2, z1_pred, logits_s1, logits_s2
            return z1_pred, logits_s1

    def get_retrieval_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes L2-normalized embeddings for indexing and retrieval.
        """
        self.eval()
        with torch.no_grad():
            if self.in_channels in [14, 5]:
                if x.shape[1] == self.in_channels:
                    # Default: extract target S2 features for gallery
                    x_target = x[:, self.s1_channels:, :, :]
                    feats = self.backbone(x_target, self.s2_wvs)
                    z = self.s2_projection(feats)
                elif x.shape[1] == self.s2_channels:
                    feats = self.backbone(x, self.s2_wvs)
                    z = self.s2_projection(feats)
                elif x.shape[1] == self.s1_channels:
                    # For S1 query, project it and run predictor/bridge to align with target space
                    feats = self.backbone(x, self.s1_wvs)
                    z = self.s1_projection(feats)
                    if self.bridge is not None:
                        z_pred = self.bridge(z)
                    else:
                        z_pred = self.predictor(z)
                        
                    if self.hashing_head is not None:
                        return self.hashing_head.hard_codes(z_pred)
                    return self.retrieval_head(z_pred)
                else:
                    raise ValueError(f"Unexpected channel dimension in cross-modal retrieval: {x.shape}")
                
                if self.hashing_head is not None:
                    return self.hashing_head.hard_codes(z)
                embeddings = self.retrieval_head(z)
            else:
                wvs = self._get_wvs_for_channels(self.in_channels)
                feats = self.backbone(x, wvs)
                z = self.projection_head(feats)
                if self.hashing_head is not None:
                    return self.hashing_head.hard_codes(z)
                embeddings = self.retrieval_head(z)
        return embeddings

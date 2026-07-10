import logging
import torch
import torch.nn as nn
import timm
from typing import Optional

logger = logging.getLogger("saber")

class FrozenViTBackbone(nn.Module):
    """
    Loads and freezes a pretrained Vision Transformer backbone from timm.
    Supports ViT-B/16, DINOv2, MAE, etc., with automatic fallback mechanisms.
    """
    def __init__(self, model_name: str = "vit_base_patch16_224", pretrained: bool = True) -> None:
        """
        Args:
            model_name: The timm model identifier.
            pretrained: Whether to load pretrained weights.
        """
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        
        self.model = self._load_model_with_fallbacks()
        self.embed_dim = self.model.num_features
        
        # Freeze backbone parameters
        for p in self.model.parameters():
            p.requires_grad = False
            
    def _load_model_with_fallbacks(self) -> nn.Module:
        """Load timm ViT with fallbacks if preferred models are not available."""
        preferred_models = [
            self.model_name,
            "vit_base_patch16_224",
            "vit_base_patch16_224.dino",
            "vit_base_patch16_224.mae"
        ]
        
        for name in preferred_models:
            try:
                logger.info(f"Attempting to load backbone '{name}' (pretrained={self.pretrained})...")
                # num_classes=0 removes the head and returns the pooled features or CLS token
                model = timm.create_model(name, pretrained=self.pretrained, num_classes=0)
                logger.info(f"Successfully loaded backbone '{name}' with feature dimension {model.num_features}")
                return model
            except Exception as e:
                logger.warning(f"Failed to load '{name}': {e}")
                
        # Last resort: Try loading ViT base patch 16 without pretrained weights if offline or unavailable
        try:
            logger.info("Attempting to load un-pretrained vit_base_patch16_224 as last resort...")
            return timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        except Exception as e:
            raise RuntimeError(f"Could not initialize any ViT backbone: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract frozen features from input.
        Always runs in eval mode with torch.no_grad().
        
        Args:
            x: Input tensor of shape (B, 3, H, W).
            
        Returns:
            Extracted feature tensor of shape (B, embed_dim).
        """
        self.model.eval()
        with torch.no_grad():
            features = self.model(x)
        return features

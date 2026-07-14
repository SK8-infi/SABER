import logging
import torch
import torch.nn as nn
import timm
from typing import Optional, List
import os
import sys

logger = logging.getLogger("saber")

# Add local dofa dir to sys.path so we can import from dofa_v1
dofa_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dofa"))
if dofa_dir not in sys.path:
    sys.path.insert(0, dofa_dir)

try:
    from dofa_v1 import vit_base_patch16
except ImportError as e:
    logger.error(f"Could not import dofa_v1 from local dofa directory: {e}")

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


class FrozenDOFABackbone(nn.Module):
    """
    Loads and freezes the pre-trained DOFA ViT backbone.
    Utilizes a wavelength-conditioned patch embedding hypernetwork.
    """
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.model = vit_base_patch16()
        self.embed_dim = 768
        
        if pretrained:
            url = "https://huggingface.co/earthflow/DOFA/resolve/main/DOFA_ViT_base_e100.pth"
            try:
                logger.info("Loading DOFA pretrained weights...")
                
                # Check standard torch hub cache path to avoid unnecessary downloads
                cached_file = os.path.expanduser("~/.cache/torch/hub/checkpoints/DOFA_ViT_base_e100.pth")
                if os.path.exists(cached_file):
                    logger.info(f"Loading weights from local cache: {cached_file}")
                    state_dict = torch.load(cached_file, map_location='cpu', weights_only=False)
                else:
                    logger.info(f"Downloading DOFA pretrained weights from HF: {url}")
                    state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
                
                self.model.load_state_dict(state_dict, strict=False)
                logger.info("Successfully loaded DOFA pretrained weights.")
            except Exception as e:
                logger.error(f"Failed to load DOFA weights: {e}")
                raise e
                
        # Freeze backbone parameters
        for p in self.model.parameters():
            p.requires_grad = False
            
    def forward(self, x: torch.Tensor, wave_list: List[float]) -> torch.Tensor:
        """
        Extract features from input tensor x conditioned on wavelengths list.
        Args:
            x: Input tensor of shape (B, C, H, W).
            wave_list: Central wavelengths in micrometers. Shape (C,).
        Returns:
            Pooled feature representation of shape (B, 768).
        """
        has_trainable = any(p.requires_grad for p in self.model.parameters())
        if self.training and has_trainable:
            features = self.model.forward_features(x, wave_list)
        else:
            prev_mode = self.model.training
            self.model.eval()
            with torch.no_grad():
                features = self.model.forward_features(x, wave_list)
            self.model.train(prev_mode)
        return features


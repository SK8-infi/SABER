from typing import Any
import torch
import numpy as np

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

class DummyTransform:
    """Fallback spatial transform pipeline for offline testing without albumentations."""
    def __call__(self, image=None, **kwargs):
        if image is None:
            return {"image": None}
        if isinstance(image, torch.Tensor):
            return {"image": image}
        img_t = torch.tensor(image, dtype=torch.float32)
        if img_t.ndim == 3 and img_t.shape[-1] in [1, 2, 4, 12, 14]:
            img_t = img_t.permute(2, 0, 1)
        return {"image": img_t}

def get_transforms(image_size: int = 224, is_train: bool = True) -> Any:
    """
    Get spatial transform pipelines.
    Supports multi-channel remote sensing images.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        return DummyTransform()

    if is_train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.85, 1.15), translate_percent=(-0.05, 0.05), rotate=(-15, 15),
                p=0.5
            ),


            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.7, 1.0), ratio=(0.85, 1.15), p=0.5
            ),
            A.GaussNoise(p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.4
            ),
            A.ChannelDropout(channel_drop_range=(1, 1), p=0.1),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            ToTensorV2()
        ])

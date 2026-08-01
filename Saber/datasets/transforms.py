import os
from typing import Any, List, Dict, Tuple
import torch
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

try:
    import cv2
    cv2.setNumThreads(0)
except ImportError:
    pass

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

class MultiCropTransform:
    """
    Multi-Crop Self-Supervised Data Augmentation:
    Generates 2 Global Crops (224x224, scale 0.4-1.0) + 4 Local Sub-Crops (96x96, scale 0.15-0.4).
    """
    def __init__(self, global_size: int = 224, local_size: int = 96, num_global: int = 2, num_local: int = 4) -> None:
        self.num_global = num_global
        self.num_local = num_local
        
        if ALBUMENTATIONS_AVAILABLE:
            self.global_transform = A.Compose([
                A.RandomResizedCrop(size=(global_size, global_size), scale=(0.4, 1.0), p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
                ToTensorV2()
            ])
            self.local_transform = A.Compose([
                A.RandomResizedCrop(size=(local_size, local_size), scale=(0.15, 0.4), p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2()
            ])
        else:
            self.global_transform = DummyTransform()
            self.local_transform = DummyTransform()

    def __call__(self, image=None, **kwargs):
        crops = []
        for _ in range(self.num_global):
            res = self.global_transform(image=image)
            img = res.get("image", res)
            crops.append(img)
        for _ in range(self.num_local):
            res = self.local_transform(image=image)
            img = res.get("image", res)
            crops.append(img)
        return {"crops": crops}

def get_transforms(image_size: int = 224, is_train: bool = True, multi_crop: bool = False) -> Any:
    """
    Get spatial transform pipelines.
    Supports multi-channel remote sensing images.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        return DummyTransform()

    if is_train:
        if multi_crop:
            return MultiCropTransform(global_size=image_size, local_size=96, num_global=2, num_local=4)
            
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.7, 1.0), ratio=(0.85, 1.15), p=0.5
            ),
            A.GaussNoise(p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.4
            ),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            ToTensorV2()
        ])

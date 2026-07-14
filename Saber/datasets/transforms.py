from typing import Any
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(image_size: int = 224, is_train: bool = True) -> Any:
    """
    Get albumentations transform pipelines.
    Supports multi-channel remote sensing images by applying spatial transforms.
    
    Args:
        image_size: Target height and width for the images.
        is_train: If True, returns a pipeline with augmentations. Otherwise, val/test transforms.
        
    Returns:
        An Albumentations Compose object.
    """
    if is_train:
        return A.Compose([
            A.Resize(image_size, image_size),
            # Geometric augmentations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.15, rotate_limit=15,
                border_mode=0, p=0.5
            ),
            # Multi-scale views via random crop + resize
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.7, 1.0), ratio=(0.85, 1.15), p=0.5
            ),
            # Radiometric / spectral augmentations (safe for multi-channel RS imagery)
            A.GaussNoise(p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.4
            ),
            # Channel dropout: randomly zero one band (simulates band failure)
            A.ChannelDropout(channel_drop_range=(1, 1), p=0.1),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            ToTensorV2()
        ])

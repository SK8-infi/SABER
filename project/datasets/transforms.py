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
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            ToTensorV2()
        ])

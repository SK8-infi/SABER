import os
import torch
import numpy as np
from typing import Dict, Any, Optional
from project.datasets.base_dataset import BaseDataset

class BEN14KDataset(BaseDataset):
    """
    Dataset class for BEN-14K remote sensing images.
    Supports Sentinel-1 (2 channels) and Sentinel-2 (12 channels).
    """
    def __init__(
        self,
        data_dir: str,
        use_synthetic: bool = False,
        size: int = 1000,
        image_size: int = 224,
        transform: Optional[Any] = None,
        modality: str = "s2",  # "s1", "s2", or "both"
        num_classes: int = 19,
        is_train: bool = True
    ) -> None:
        self.modality = modality.lower()
        self.num_classes = num_classes
        self.is_train = is_train
        
        if self.modality == "s1":
            self.num_channels = 2
        elif self.modality == "s2":
            self.num_channels = 12
        else:
            self.num_channels = 14
            
        super().__init__(
            data_dir=data_dir,
            use_synthetic=use_synthetic,
            size=size,
            image_size=image_size,
            transform=transform
        )
        
        self.image_paths = []
        self.labels = []
        
        # If not forced synthetic, try to locate files
        if not self.use_synthetic and os.path.exists(data_dir):
            self._scan_dataset()
            
        # Fall back to synthetic mode if no images were found
        if len(self.image_paths) == 0:
            self.use_synthetic = True
            
    def _scan_dataset(self) -> None:
        """Scan the data directory for image files."""
        # Simple scan logic
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(root, file))
                    # Assign a random mock multi-label for classification task simulations
                    dummy_label = np.random.randint(0, 2, size=self.num_classes).astype(np.float32)
                    self.labels.append(dummy_label)
                    
                    if len(self.image_paths) >= self.size:
                        break
            if len(self.image_paths) >= self.size:
                break

    def __len__(self) -> int:
        return self.size if self.use_synthetic else len(self.image_paths)

    def get_real_item(self, idx: int) -> Dict[str, Any]:
        path = self.image_paths[idx]
        img = None
        
        # Try importing specialized geospatial libraries
        try:
            import tifffile as tiff
            img = tiff.imread(path)
        except ImportError:
            try:
                import rasterio
                with rasterio.open(path) as src:
                    img = src.read()  # Shape: (C, H, W)
                    img = np.moveaxis(img, 0, -1)  # Shape: (H, W, C)
            except ImportError:
                # Fallback to standard PIL image loader
                from PIL import Image
                img = np.array(Image.open(path))

        if img is None:
            raise ValueError(f"Could not load image file from: {path}")

        # Handle dimension shapes
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        elif len(img.shape) == 3 and img.shape[0] == self.num_channels:
            img = np.moveaxis(img, 0, -1)

        # Slice or pad channels to conform with required channels
        if img.shape[-1] != self.num_channels:
            if img.shape[-1] > self.num_channels:
                img = img[..., :self.num_channels]
            else:
                pad_width = ((0, 0), (0, 0), (0, self.num_channels - img.shape[-1]))
                img = np.pad(img, pad_width, mode="constant")

        label = self.labels[idx]

        if self.is_train:
            if self.transform:
                img_tensor1 = self.transform(image=img.astype(np.float32))["image"]
                img_tensor2 = self.transform(image=img.astype(np.float32))["image"]
            else:
                img_tensor1 = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
                img_tensor2 = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
            return {
                "image1": img_tensor1.float(),
                "image2": img_tensor2.float(),
                "label": torch.tensor(label, dtype=torch.float32),
                "name": os.path.basename(path)
            }
        else:
            if self.transform:
                img_tensor = self.transform(image=img.astype(np.float32))["image"]
            else:
                img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
            return {
                "image": img_tensor.float(),
                "label": torch.tensor(label, dtype=torch.float32),
                "name": os.path.basename(path)
            }

    def get_synthetic_item(self, idx: int) -> Dict[str, Any]:
        # Generate synthetic normal distributed noise
        img = np.random.randn(self.image_size, self.image_size, self.num_channels).astype(np.float32)
        
        # Generate multi-hot binary label
        label = np.zeros(self.num_classes, dtype=np.float32)
        active_indices = np.random.choice(self.num_classes, size=np.random.randint(1, 4), replace=False)
        label[active_indices] = 1.0

        if self.is_train:
            if self.transform:
                img_tensor1 = self.transform(image=img)["image"]
                img_tensor2 = self.transform(image=img)["image"]
            else:
                img_tensor1 = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
                img_tensor2 = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
            return {
                "image1": img_tensor1,
                "image2": img_tensor2,
                "label": torch.tensor(label, dtype=torch.float32),
                "name": f"BEN14K_synthetic_{idx}.png"
            }
        else:
            if self.transform:
                img_tensor = self.transform(image=img)["image"]
            else:
                img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
            return {
                "image": img_tensor,
                "label": torch.tensor(label, dtype=torch.float32),
                "name": f"BEN14K_synthetic_{idx}.png"
            }

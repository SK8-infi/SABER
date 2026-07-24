import os
import torch
import numpy as np
import logging
from typing import Dict, Any, Optional
from Saber.datasets.base_dataset import BaseDataset

logger = logging.getLogger("saber")

DSRSID_CLASSES = [
    "aquafarm",
    "cloud",
    "forest",
    "high building",
    "low building",
    "farm land",
    "river",
    "water"
]

class DSRSIDDataset(BaseDataset):
    """
    Dataset class for DSRSID Gaofen-1 satellite remote sensing image pairs.
    Loads real Panchromatic (PAN, 1 channel, 256x256) and Multispectral (MS, 4 channels, 64x64)
    images directly from the DSRSID.mat HDF5 file.
    """
    def __init__(
        self,
        data_dir: str,
        use_synthetic: bool = False,
        size: int = 1000,
        image_size: int = 224,
        transform: Optional[Any] = None,
        modality: str = "ms",  # "pan", "ms", or "both"
        num_classes: int = 8,
        is_train: bool = True,
        split: str = "train"  # "train", "val", "test", or "all"
    ) -> None:
        self.modality = modality.lower()
        self.num_classes = num_classes
        self.is_train = is_train
        self.split = split.lower()

        if self.modality == "pan":
            self.num_channels = 1
        elif self.modality == "ms":
            self.num_channels = 4
        else:
            self.num_channels = 5
            
        super().__init__(
            data_dir=data_dir,
            use_synthetic=use_synthetic,
            size=size,
            image_size=image_size,
            transform=transform
        )

        self.mat_path = None
        self.f_handle = None  # Lazy-opened HDF5 file handle

        if self.use_synthetic:
            total_n = self.size
            train_end = int(0.70 * total_n)
            val_end = int(0.80 * total_n)
            if self.split == "train":
                self.size = train_end
            elif self.split == "val":
                self.size = val_end - train_end
            elif self.split == "test":
                self.size = total_n - val_end

        if not self.use_synthetic:
            # Detect where DSRSID.mat is located
            if os.path.isfile(data_dir) and data_dir.endswith(".mat"):
                self.mat_path = data_dir
            elif os.path.isdir(data_dir):
                target_file = os.path.join(data_dir, "DSRSID.mat")
                if os.path.exists(target_file):
                    self.mat_path = target_file
            
            # Fallback scan locations
            if self.mat_path is None or not os.path.exists(self.mat_path):
                fallbacks = [
                    "c:/Github/SABER/Datasets/DSRSID/DSRSID-001.mat",
                    os.path.join(os.path.expanduser("~"), "Downloads", "DSRSID.mat"),
                    "data/DSRSID.mat"
                ]
                for path in fallbacks:
                    if os.path.exists(path):
                        self.mat_path = path
                        break

            if self.mat_path is None:
                logger.warning(
                    f"DSRSID.mat not found in data_dir '{data_dir}' or fallbacks. "
                    "Falling back to synthetic data."
                )
                self.use_synthetic = True
            else:
                # Open temporarily to read dataset size and build stratified index map
                import h5py
                try:
                    with h5py.File(self.mat_path, "r") as f:
                        self.total_samples = f["MUL_IMAGES"].shape[0]
                        all_labels = f["LAND_COVER_TYPES"][0, :].astype(int)

                    self.size = min(self.size, self.total_samples)

                    # Build stratified indices: sample equally from each class
                    unique_classes = np.unique(all_labels)
                    num_classes_found = len(unique_classes)
                    per_class = max(1, self.size // num_classes_found)
                    rng = np.random.RandomState(42)
                    selected = []
                    for cls in unique_classes:
                        cls_indices = np.where(all_labels == cls)[0]
                        n_take = min(per_class, len(cls_indices))
                        chosen = rng.choice(cls_indices, size=n_take, replace=False)
                        selected.append(chosen)
                    all_stratified = np.sort(np.concatenate(selected))[:self.size]
                    
                    # Split partitioning (Seed 42): 70% Train | 10% Val | 20% Test
                    total_n = len(all_stratified)
                    train_end = int(0.70 * total_n)
                    val_end = int(0.80 * total_n)
                    
                    if self.split == "train":
                        self.sample_indices = all_stratified[:train_end]
                    elif self.split == "val":
                        self.sample_indices = all_stratified[train_end:val_end]
                    elif self.split == "test":
                        self.sample_indices = all_stratified[val_end:]
                    else: # "all"
                        self.sample_indices = all_stratified

                    self.size = len(self.sample_indices)

                    logger.info(f"Connected to DSRSID.mat at '{self.mat_path}'. Using {self.size} samples [{self.split.upper()} SPLIT].")
                except Exception as e:
                    logger.error(f"Error reading DSRSID.mat: {e}. Falling back to synthetic mode.")
                    self.use_synthetic = True

    def __len__(self) -> int:
        return self.size

    def get_real_item(self, idx: int) -> Dict[str, Any]:
        # Lazy open HDF5 file handle (avoids open file handle copies across PyTorch worker processes)
        if self.f_handle is None:
            import h5py
            self.f_handle = h5py.File(self.mat_path, "r")

        # Map sequential idx to the stratified sample index
        real_idx = int(self.sample_indices[idx]) if hasattr(self, 'sample_indices') else idx
        # 1. Load label (DSRSID labels are float64, 1.0 to 8.0)
        # Shift 1-indexed [1, 8] label values to 0-indexed [0, 7] labels for PyTorch compatibility
        label_raw = float(self.f_handle["LAND_COVER_TYPES"][0, real_idx])
        label = int(max(0.0, label_raw - 1.0))

        # 2. Load image array
        if self.modality == "pan":
            # PAN images shape is (1, 256, 256)
            img = np.array(self.f_handle["PAN_IMAGES"][real_idx], dtype=np.uint8).astype(np.float32)
            # Z-score normalize PAN
            img = (img - 73.59) / 22.78
            # Rearrange to HWC (256, 256, 1) for Albumentations augmentations
            img = np.moveaxis(img, 0, -1)
        elif self.modality == "both":
            # PAN images shape is (1, 256, 256)
            img_pan = np.array(self.f_handle["PAN_IMAGES"][real_idx], dtype=np.uint8).astype(np.float32)
            img_pan = (img_pan - 73.59) / 22.78
            img_pan = np.moveaxis(img_pan, 0, -1)
            
            # MS images shape is (4, 64, 64)
            img_ms = np.array(self.f_handle["MUL_IMAGES"][real_idx], dtype=np.uint8).astype(np.float32)
            ms_mean = np.array([113.24, 111.79, 84.83, 53.06], dtype=np.float32).reshape(4, 1, 1)
            ms_std = np.array([22.96, 27.69, 26.16, 21.43], dtype=np.float32).reshape(4, 1, 1)
            img_ms = (img_ms - ms_mean) / ms_std
            img_ms = np.moveaxis(img_ms, 0, -1)
            
            # Resize MS image (64x64x4) to match PAN spatial dimension (256x256) using cv2 for massive speedup
            import cv2
            img_ms_resized = cv2.resize(img_ms, (img_pan.shape[1], img_pan.shape[0]), interpolation=cv2.INTER_LINEAR)
            img = np.concatenate([img_pan, img_ms_resized], axis=-1)
        else:
            # MS images shape is (4, 64, 64)
            img = np.array(self.f_handle["MUL_IMAGES"][real_idx], dtype=np.uint8).astype(np.float32)
            ms_mean = np.array([113.24, 111.79, 84.83, 53.06], dtype=np.float32).reshape(4, 1, 1)
            ms_std = np.array([22.96, 27.69, 26.16, 21.43], dtype=np.float32).reshape(4, 1, 1)
            img = (img - ms_mean) / ms_std
            # Rearrange to HWC (64, 64, 4) for Albumentations augmentations
            img = np.moveaxis(img, 0, -1)

        name = f"DSRSID_{self.modality}_{real_idx}.png"

        # Apply spatial/color transforms
        if self.is_train:
            if self.transform:
                # Apply twice for dual-view predictive loss targets
                img_tensor1 = self.transform(image=img.astype(np.float32))["image"]
                img_tensor2 = self.transform(image=img.astype(np.float32))["image"]
            else:
                img_tensor1 = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
                img_tensor2 = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
            out = {
                "image1": img_tensor1,
                "image2": img_tensor2,
                "label": torch.tensor(label, dtype=torch.long),
                "name": name
            }
            if self.modality == "both":
                out["image1_s1"] = img_tensor1[:1]
                out["image1_s2"] = img_tensor1[1:]
                out["image2_s1"] = img_tensor2[:1]
                out["image2_s2"] = img_tensor2[1:]
            return out
        else:
            if self.transform:
                img_tensor = self.transform(image=img.astype(np.float32))["image"]
            else:
                img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
            out = {
                "image": img_tensor,
                "label": torch.tensor(label, dtype=torch.long),
                "name": name
            }
            if self.modality == "both":
                out["image_s1"] = img_tensor[:1]
                out["image_s2"] = img_tensor[1:]
            return out

    def get_synthetic_item(self, idx: int) -> Dict[str, Any]:
        # Fallback generator
        img = np.random.randn(self.image_size, self.image_size, self.num_channels).astype(np.float32)
        label = np.random.randint(0, self.num_classes)

        if self.is_train:
            if self.transform:
                img_tensor1 = self.transform(image=img)["image"]
                img_tensor2 = self.transform(image=img)["image"]
            else:
                img_tensor1 = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
                img_tensor2 = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
            out = {
                "image1": img_tensor1,
                "image2": img_tensor2,
                "label": torch.tensor(label, dtype=torch.long),
                "name": f"DSRSID_synthetic_{idx}.png"
            }
            if self.modality == "both":
                out["image1_s1"] = img_tensor1[:1]
                out["image1_s2"] = img_tensor1[1:]
                out["image2_s1"] = img_tensor2[:1]
                out["image2_s2"] = img_tensor2[1:]
            return out
        else:
            if self.transform:
                img_tensor = self.transform(image=img)["image"]
            else:
                img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
            out = {
                "image": img_tensor,
                "label": torch.tensor(label, dtype=torch.long),
                "name": f"DSRSID_synthetic_{idx}.png"
            }
            if self.modality == "both":
                out["image_s1"] = img_tensor[:1]
                out["image_s2"] = img_tensor[1:]
            return out

    def __del__(self) -> None:
        # Gracefully release HDF5 handle on object destruction
        if hasattr(self, "f_handle") and self.f_handle is not None:
            try:
                self.f_handle.close()
            except Exception:
                pass

import os
import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, List
from Saber.datasets.base_dataset import BaseDataset

logger = logging.getLogger("saber")

# Official BigEarthNet 19-class simplified nomenclature
BIGEARTHNET_19_CLASSES = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland",
    "Moors and heathland",
    "Transitional woodland/shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Salt marshes",
    "Water bodies",
    "Coastal wetlands"
]

# Mapping dictionary from raw CORINE classes to the 19 target classes
CORINE_TO_19_MAP = {
    # 1. Urban fabric
    "Continuous urban fabric": "Urban fabric",
    "Discontinuous urban fabric": "Urban fabric",
    "Construction sites": "Urban fabric",
    "Land without current use": "Urban fabric",
    "Green urban areas": "Urban fabric",
    "Sport and leisure facilities": "Urban fabric",
    "Road and rail networks and associated land": "Urban fabric",
    "Port areas": "Urban fabric",
    "Airports": "Urban fabric",
    "Dump sites": "Urban fabric",
    "Mineral extraction sites": "Urban fabric",
    
    # 2. Industrial or commercial units
    "Industrial or commercial units": "Industrial or commercial units",
    
    # 3. Arable land
    "Non-irrigated arable land": "Arable land",
    "Permanently irrigated land": "Arable land",
    "Rice fields": "Arable land",
    
    # 4. Permanent crops
    "Vineyards": "Permanent crops",
    "Fruit trees and berry plantations": "Permanent crops",
    "Olive groves": "Permanent crops",
    
    # 5. Pastures
    "Pastures": "Pastures",
    
    # 6. Complex cultivation patterns
    "Complex cultivation patterns": "Complex cultivation patterns",
    
    # 7. Land principally occupied by agriculture...
    "Land principally occupied by agriculture, with significant areas of natural vegetation": "Land principally occupied by agriculture, with significant areas of natural vegetation",
    
    # 8. Agro-forestry areas
    "Agro-forestry areas": "Agro-forestry areas",
    
    # 9. Broad-leaved forest
    "Broad-leaved forest": "Broad-leaved forest",
    
    # 10. Coniferous forest
    "Coniferous forest": "Coniferous forest",
    
    # 11. Mixed forest
    "Mixed forest": "Mixed forest",
    
    # 12. Natural grassland
    "Natural grassland": "Natural grassland",
    "Sclerophyllous vegetation": "Natural grassland",
    
    # 13. Moors and heathland
    "Moors and heathland": "Moors and heathland",
    
    # 14. Transitional woodland/shrub
    "Transitional woodland/shrub": "Transitional woodland/shrub",
    
    # 15. Beaches, dunes, sands
    "Beaches, dunes, sands": "Beaches, dunes, sands",
    "Bare rock": "Beaches, dunes, sands",
    "Sparsely vegetated areas": "Beaches, dunes, sands",
    "Burnt areas": "Beaches, dunes, sands",
    "Glaciers and perpetual snow": "Beaches, dunes, sands",
    
    # 16. Inland wetlands
    "Inland marshes": "Inland wetlands",
    "Peat bogs": "Inland wetlands",
    
    # 17. Salt marshes
    "Salt marshes": "Salt marshes",
    "Salines": "Salt marshes",
    
    # 18. Water bodies
    "Water courses": "Water bodies",
    "Water bodies": "Water bodies",
    "Coastal lagoons": "Water bodies",
    "Estuaries": "Water bodies",
    "Sea and ocean": "Water bodies",
    
    # 19. Coastal wetlands
    "Intertidal flats": "Coastal wetlands"
}

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
        is_train: bool = True,
        split: str = "train"  # "train", "val", "test", or "all"
    ) -> None:
        self.modality = modality.lower()
        self.num_classes = num_classes
        self.is_train = is_train
        self.split = split.lower()
        
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
        
        self.csv_path = None
        self.df = None

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

        # Build class-to-index mapping
        self.class_to_idx = {name: idx for idx, name in enumerate(BIGEARTHNET_19_CLASSES)}

        if not self.use_synthetic:
            # Locate master labels CSV file
            if os.path.isdir(data_dir):
                target_csv = os.path.join(data_dir, "benv1_14k_dataset_master_labels.csv")
                if os.path.exists(target_csv):
                    self.csv_path = target_csv
            
            # Fallback scan locations
            if self.csv_path is None or not os.path.exists(self.csv_path):
                fallbacks = [
                    "c:/Github/SABER/Datasets/benv1_14k/benv1_14k_dataset_master_labels.csv",
                    os.path.join(os.path.expanduser("~"), "Downloads", "benv1_14k", "benv1_14k_dataset_master_labels.csv"),
                    "data/benv1_14k/benv1_14k_dataset_master_labels.csv"
                ]
                for path in fallbacks:
                    if os.path.exists(path):
                        self.csv_path = path
                        break

            if self.csv_path is None:
                logger.warning(
                    f"benv1_14k_dataset_master_labels.csv not found in data_dir '{data_dir}' or fallbacks. "
                    "Falling back to synthetic data."
                )
                self.use_synthetic = True
            else:
                try:
                    full_df = pd.read_csv(self.csv_path)
                    total_n = len(full_df)
                    
                    # Deterministic split partitioning (Seed 42)
                    # 70% Train (10,382) | 10% Val (1,483) | 20% Test (2,967)
                    rng = np.random.RandomState(42)
                    shuffled_idx = rng.permutation(total_n)
                    
                    train_end = int(0.70 * total_n)
                    val_end = int(0.80 * total_n)
                    
                    if self.split == "train":
                        split_idx = shuffled_idx[:train_end]
                    elif self.split == "val":
                        split_idx = shuffled_idx[train_end:val_end]
                    elif self.split == "test":
                        split_idx = shuffled_idx[val_end:]
                    else: # "all"
                        split_idx = shuffled_idx
                        
                    self.df = full_df.iloc[split_idx].reset_index(drop=True)
                    self.size = len(self.df)
                    self.ben14k_root = os.path.dirname(self.csv_path)
                    logger.info(f"Loaded BEN-14K [{self.split.upper()} SPLIT] metadata CSV from '{self.csv_path}'. Using {self.size} samples.")
                except Exception as e:
                    logger.error(f"Error loading BEN-14K metadata CSV: {e}. Falling back to synthetic.")
                    self.use_synthetic = True

    def __len__(self) -> int:
        return self.size

    def _parse_multi_label(self, raw_labels_str: str) -> np.ndarray:
        """Parses raw CORINE classes to the 19 simplified classes multi-hot array."""
        label_vec = np.zeros(self.num_classes, dtype=np.float32)
        if pd.isna(raw_labels_str) or not raw_labels_str:
            return label_vec
            
        raw_labels = raw_labels_str.split("|")
        for rl in raw_labels:
            rl = rl.strip()
            # Map CORINE class to 19simplified target class name
            target_class = CORINE_TO_19_MAP.get(rl)
            if target_class and target_class in self.class_to_idx:
                idx = self.class_to_idx[target_class]
                label_vec[idx] = 1.0
        return label_vec

    def get_real_item(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        s2_id = row["S2_ID"]
        s1_id = row["S1_ID"]
        
        # Load image bands
        img_s1, img_s2 = None, None
        
        if self.modality == "s1" or self.modality == "both":
            s1_path = os.path.join(self.ben14k_root, "s1", s1_id, f"{s1_id}_all.npy")
            # Loaded shape: (2, 120, 120)
            img_s1 = np.load(s1_path).astype(np.float32)
            
            # S1 DB Clipping & Min-Max Scaling to [0, 1]
            vv_clipped = np.clip(img_s1[0], -20.0, 5.0)
            vh_clipped = np.clip(img_s1[1], -30.0, 0.0)
            vv_norm = (vv_clipped + 20.0) / 25.0
            vh_norm = (vh_clipped + 30.0) / 30.0
            img_s1 = np.stack([vv_norm, vh_norm], axis=0)
            
            # Z-score normalization with legacy stats
            s1_mean = np.array([0.34904295, 0.4175904], dtype=np.float32).reshape(2, 1, 1)
            s1_std = np.array([0.13458131, 0.12477361], dtype=np.float32).reshape(2, 1, 1)
            img_s1 = (img_s1 - s1_mean) / s1_std
            
            # Rearrange to (120, 120, 2) for augmentations
            img_s1 = np.moveaxis(img_s1, 0, -1)
            
        if self.modality == "s2" or self.modality == "both":
            s2_path = os.path.join(self.ben14k_root, "s2", s2_id, f"{s2_id}_all.npy")
            # Loaded shape: (12, 120, 120)
            img_s2 = np.load(s2_path).astype(np.float32)
            
            # S2 Scaling to [0, 1]
            img_s2 = img_s2 / 10000.0
            
            # Z-score normalization with legacy stats
            s2_mean = np.array([0.03488038, 0.04391864, 0.06977729, 0.06504002, 0.11127272, 0.2385335, 0.2864792, 0.29709122, 0.31154051, 0.30906704, 0.22802109, 0.14070274], dtype=np.float32).reshape(12, 1, 1)
            s2_std = np.array([0.01923979, 0.02354799, 0.02809117, 0.04333137, 0.0391201, 0.05621961, 0.07425146, 0.08028981, 0.07793317, 0.07158578, 0.06703733, 0.0692741], dtype=np.float32).reshape(12, 1, 1)
            img_s2 = (img_s2 - s2_mean) / s2_std
            
            # Rearrange to (120, 120, 12) for augmentations
            img_s2 = np.moveaxis(img_s2, 0, -1)

        # Assemble image channels
        if self.modality == "s1":
            img = img_s1
            name = f"{s1_id}.png"
        elif self.modality == "s2":
            img = img_s2
            name = f"{s2_id}.png"
        else:  # both
            img = np.concatenate([img_s1, img_s2], axis=-1)  # shape: (120, 120, 14)
            name = f"{s2_id}_paired.png"

        # Parse 19-class label
        label = self._parse_multi_label(row["S2_Labels"])

        # Apply augmentations
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
                "label": torch.tensor(label, dtype=torch.float32),
                "name": name
            }
            if self.modality == "both":
                out["image1_s1"] = img_tensor1[:2]
                out["image1_s2"] = img_tensor1[2:]
                out["image2_s1"] = img_tensor2[:2]
                out["image2_s2"] = img_tensor2[2:]
            return out
        else:
            if self.transform:
                img_tensor = self.transform(image=img)["image"]
            else:
                img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
            out = {
                "image": img_tensor,
                "label": torch.tensor(label, dtype=torch.float32),
                "name": name
            }
            if self.modality == "both":
                out["image_s1"] = img_tensor[:2]
                out["image_s2"] = img_tensor[2:]
            return out

    def get_synthetic_item(self, idx: int) -> Dict[str, Any]:
        # Generate synthetic normal noise
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
            out = {
                "image1": img_tensor1,
                "image2": img_tensor2,
                "label": torch.tensor(label, dtype=torch.float32),
                "name": f"BEN14K_synthetic_{idx}.png"
            }
            if self.modality == "both":
                out["image1_s1"] = img_tensor1[:2]
                out["image1_s2"] = img_tensor1[2:]
                out["image2_s1"] = img_tensor2[:2]
                out["image2_s2"] = img_tensor2[2:]
            return out
        else:
            if self.transform:
                img_tensor = self.transform(image=img)["image"]
            else:
                img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
            out = {
                "image": img_tensor,
                "label": torch.tensor(label, dtype=torch.float32),
                "name": f"BEN14K_synthetic_{idx}.png"
            }
            if self.modality == "both":
                out["image_s1"] = img_tensor[:2]
                out["image_s2"] = img_tensor[2:]
            return out

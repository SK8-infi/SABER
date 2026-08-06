import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

class BaseDataset(Dataset, ABC):
    """
    Abstract base dataset defining the standard interface for remote sensing datasets.
    Supports a synthetic mode fallback if the real data is not available.
    """
    def __init__(
        self,
        data_dir: str,
        use_synthetic: bool = False,
        size: int = 1000,
        image_size: int = 224,
        transform: Any = None
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.use_synthetic = use_synthetic
        self.size = size
        self.image_size = image_size
        self.transform = transform

    @abstractmethod
    def __len__(self) -> int:
        """Return the size of the dataset."""
        pass

    @abstractmethod
    def get_real_item(self, idx: int) -> Dict[str, Any]:
        """Load and return a real sample from the dataset."""
        pass

    @abstractmethod
    def get_synthetic_item(self, idx: int) -> Dict[str, Any]:
        """Generate and return a synthetic sample with the correct dimensions and format."""
        pass

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample. Loads real item unless use_synthetic is explicitly set to True.
        """
        if self.use_synthetic:
            return self.get_synthetic_item(idx)
        return self.get_real_item(idx)

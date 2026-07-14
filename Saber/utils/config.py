import yaml
from typing import Any, Dict

class DotDict(dict):
    """
    A dictionary subclass that allows dot notation access to nested keys.
    For example: config.model.backbone_name
    """
    def __getattr__(self, name: str) -> Any:
        try:
            val = self[name]
            if isinstance(val, dict) and not isinstance(val, DotDict):
                val = DotDict(val)
                self[name] = val
            return val
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{name}'")

def load_config(config_path: str) -> DotDict:
    """
    Load a YAML configuration file and return a DotDict object.
    Automatically scales batch size and learning rate on high-VRAM GPUs (e.g., T4/15GB).
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        A DotDict object representing the configuration.
    """
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = DotDict(config_dict)
    
    # Auto-scale batch size and learning rate on high-VRAM GPUs
    try:
        import torch
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            # If GPU has > 12 GB VRAM (e.g., T4, L4, A100)
            if total_vram > 12.0:
                # Default batch size in config is 16; auto-upgrade to 64
                if "dataset" in config and config.dataset.get("batch_size", 16) == 16:
                    config.dataset.batch_size = 64
                # Scale learning rate proportionally (0.001 -> 0.002)
                if "train" in config and config.train.get("learning_rate", 0.001) == 0.001:
                    config.train.learning_rate = 0.002
    except Exception:
        pass
        
    return config


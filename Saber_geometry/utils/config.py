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
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        A DotDict object representing the configuration.
    """
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return DotDict(config_dict)

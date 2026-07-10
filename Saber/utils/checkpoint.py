import os
import torch
from typing import Any, Dict

def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str = "checkpoint.pth",
    is_best: bool = False
) -> None:
    """
    Save the model training state.
    
    Args:
        state: Dictionary containing state_dict, optimizer, epoch, etc.
        checkpoint_dir: Directory path to save the checkpoint.
        filename: Name of the checkpoint file.
        is_best: If True, also save as 'model_best.pth'.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, "model_best.pth")
        torch.save(state, best_filepath)

def load_checkpoint(filepath: str, map_location: str = "cpu") -> Dict[str, Any]:
    """
    Load a saved model checkpoint.
    
    Args:
        filepath: Path to the checkpoint file.
        map_location: Target device mapping for loaded weights.
        
    Returns:
        The loaded checkpoint dictionary.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at '{filepath}'")
    try:
        return torch.load(filepath, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(filepath, map_location=map_location)

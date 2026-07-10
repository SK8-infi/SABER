import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

def visualize_attention_map(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    save_path: str
) -> None:
    """
    Hooks the final self-attention block of the ViT backbone to capture and plot
    the self-attention weights overlaid on the input image.
    
    Args:
        model: The REJEPA model.
        image_tensor: Single input image of shape (C, H, W) or (1, C, H, W).
        save_path: Location where the resulting plot will be saved.
    """
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)  # Shape -> (1, C, H, W)

    captured_attn = []

    def hook_fn(module, input_args, output):
        captured_attn.append(output.detach().cpu())

    # Navigate to the timm backbone block structure
    backbone_model = getattr(model, "backbone", model)
    vit_model = getattr(backbone_model, "model", backbone_model)

    hook_handle = None
    try:
        if hasattr(vit_model, "blocks") and len(vit_model.blocks) > 0:
            last_block = vit_model.blocks[-1]
            if hasattr(last_block, "attn") and hasattr(last_block.attn, "attn_drop"):
                # attn_drop is a dropout layer containing the calculated attention softmax matrix
                hook_handle = last_block.attn.attn_drop.register_forward_hook(hook_fn)
    except Exception:
        pass

    # Run forward pass to trigger hook
    model.eval()
    with torch.no_grad():
        _ = model(image_tensor.to(next(model.parameters()).device))

    if hook_handle is not None:
        hook_handle.remove()

    # Prep the original image for RGB display
    img_np = image_tensor[0].cpu().numpy()
    if img_np.shape[0] in [1, 2, 4, 12]:
        img_np = np.moveaxis(img_np, 0, -1)

    h, w, c = img_np.shape
    if c > 3:
        rgb = img_np[..., :3]
    elif c == 1:
        rgb = np.repeat(img_np, 3, axis=-1)
    elif c == 2:
        pad = np.zeros((h, w, 1), dtype=img_np.dtype)
        rgb = np.concatenate([img_np, pad], axis=-1)
    else:
        rgb = img_np

    # Rescale to [0, 1]
    img_min, img_max = rgb.min(), rgb.max()
    if img_max - img_min > 0:
        rgb = (rgb - img_min) / (img_max - img_min)

    plt.figure(figsize=(10, 5), dpi=150)

    # Plot input image
    plt.subplot(1, 2, 1)
    plt.imshow(rgb)
    plt.title("Original Remote Sensing Input")
    plt.axis("off")

    # Plot Attention overlay
    plt.subplot(1, 2, 2)
    if len(captured_attn) > 0:
        # attention matrix shape: (1, num_heads, num_tokens, num_tokens)
        attn = captured_attn[0][0]  # shape: (num_heads, num_tokens, num_tokens)
        attn_avg = attn.mean(dim=0)  # average across heads: (num_tokens, num_tokens)
        
        # Get attention from CLS token (index 0) to all patch tokens (index 1 to end)
        cls_attn = attn_avg[0, 1:]  # shape: (num_tokens - 1,)
        
        num_patches = len(cls_attn)
        grid_size = int(np.sqrt(num_patches))
        
        if grid_size * grid_size == num_patches:
            cls_attn_grid = cls_attn.reshape(grid_size, grid_size).numpy()
            
            # Upsample back to match original image size
            cls_attn_tensor = torch.tensor(cls_attn_grid).unsqueeze(0).unsqueeze(0)
            cls_attn_upsampled = F.interpolate(
                cls_attn_tensor,
                size=(image_tensor.shape[2], image_tensor.shape[3]),
                mode="bilinear",
                align_corners=False
            ).numpy()[0, 0]

            plt.imshow(rgb)
            plt.imshow(cls_attn_upsampled, cmap="jet", alpha=0.55)
            plt.title("ViT Attention Map Overlay")
        else:
            plt.imshow(rgb)
            plt.title("Attention Overlay (Dimension Mismatch)")
    else:
        # Fallback if hooking failed (e.g. no attn_drop layer)
        # Compute a spatial activation map based on feature variances across the grid
        plt.imshow(rgb)
        plt.title("Attention Overlay (No Weights Hooked)")

    plt.axis("off")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

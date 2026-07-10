import torch
import torch.nn as nn

class InputAdapter(nn.Module):
    """
    Adapts multi-channel remote sensing inputs (e.g. 1, 2, 4, 12 channels)
    to a 3-channel format expected by standard Vision Transformer (ViT) backbones.
    """
    def __init__(self, in_channels: int, adapter_type: str = "conv1x1") -> None:
        """
        Args:
            in_channels: Number of channels in the raw input image.
            adapter_type: Type of adapter. Either "conv1x1" or "cnn".
        """
        super().__init__()
        self.in_channels = in_channels
        self.adapter_type = adapter_type.lower()

        if self.adapter_type == "conv1x1":
            self.adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=True)
        elif self.adapter_type == "cnn":
            self.adapter = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.GELU(),
                nn.Conv2d(16, 3, kernel_size=3, padding=1, bias=True)
            )
        else:
            raise ValueError(f"Invalid input_adapter_type: '{adapter_type}'. Use 'conv1x1' or 'cnn'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            Adapted tensor of shape (B, 3, H, W).
        """
        return self.adapter(x)

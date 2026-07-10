import torch
import torch.nn as nn

class PredictionLoss(nn.Module):
    """
    Computes the Mean Squared Error (L2 loss) between the predicted
    representations and the target representations.
    """
    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_pred: Predicted embedding of shape (B, D).
            z_target: Target embedding of shape (B, D).
            
        Returns:
            MSE loss scalar.
        """
        return self.loss_fn(z_pred, z_target)

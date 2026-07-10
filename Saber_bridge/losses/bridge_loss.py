import torch
import torch.nn as nn

class CFMLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred_v: torch.Tensor, logvar: torch.Tensor, z_s1: torch.Tensor, z_s2: torch.Tensor) -> torch.Tensor:
        """
        Calculates rectified flow-matching loss with heteroscedastic uncertainty regression.
        
        Args:
            pred_v: Predicted velocity field, shape [B, D]
            logvar: Predicted log-variance, shape [B, D]
            z_s1: Source latent representation, shape [B, D]
            z_s2: Target latent representation, shape [B, D]
            
        Returns:
            Negative log-likelihood loss scalar
        """
        # Target velocity: straight path from z_s1 to z_s2
        target_v = z_s2 - z_s1
        
        # Heteroscedastic regression loss
        diff = pred_v - target_v
        sq_error = diff ** 2
        
        precision = torch.exp(-logvar)
        loss = 0.5 * (precision * sq_error + logvar)
        
        return loss.mean()

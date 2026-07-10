import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class VICRegLoss(nn.Module):
    """
    VICReg (Variance-Invariance-Covariance Regularization) Loss.
    Ensures representations are invariant to augmentations, maintain high variance,
    and are decorrelated across feature channels.
    """
    def __init__(
        self,
        invariance_weight: float = 25.0,
        variance_weight: float = 25.0,
        covariance_weight: float = 1.0,
        epsilon: float = 1e-4
    ) -> None:
        """
        Args:
            invariance_weight: Scale factor for the Invariance loss.
            variance_weight: Scale factor for the Variance hinge loss.
            covariance_weight: Scale factor for the Covariance decorrelation loss.
            epsilon: Small constant for standard deviation stability.
        """
        super().__init__()
        self.invariance_weight = invariance_weight
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.epsilon = epsilon

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            z1: Projection embeddings from view 1 of shape (B, D).
            z2: Projection embeddings from view 2 of shape (B, D).
            
        Returns:
            Dictionary containing combined loss and sub-component losses.
        """
        batch_size, num_features = z1.shape
        if batch_size <= 1:
            # Handle edge case where variance/covariance cannot be calculated
            zero_loss = torch.tensor(0.0, device=z1.device)
            return {
                "loss": zero_loss,
                "invariance_loss": zero_loss,
                "variance_loss": zero_loss,
                "covariance_loss": zero_loss
            }

        # 1. Invariance Loss (Mean Squared Error)
        inv_loss = F.mse_loss(z1, z2)

        # 2. Variance Loss (Hinge loss targeting standard deviation close to 1.0)
        std_z1 = torch.sqrt(z1.var(dim=0) + self.epsilon)
        std_z2 = torch.sqrt(z2.var(dim=0) + self.epsilon)
        
        var_loss = (
            torch.mean(F.relu(1.0 - std_z1)) + 
            torch.mean(F.relu(1.0 - std_z2))
        ) / 2.0

        # 3. Covariance Loss (Decorrelation of off-diagonal elements in the covariance matrix)
        z1_centered = z1 - z1.mean(dim=0)
        z2_centered = z2 - z2.mean(dim=0)
        
        cov_z1 = (z1_centered.T @ z1_centered) / (batch_size - 1)
        cov_z2 = (z2_centered.T @ z2_centered) / (batch_size - 1)
        
        # Mask off-diagonal entries
        diag_mask = ~torch.eye(num_features, device=z1.device, dtype=torch.bool)
        
        cov_loss = (
            cov_z1[diag_mask].pow(2).sum() + 
            cov_z2[diag_mask].pow(2).sum()
        ) / (2.0 * num_features)

        # Weighted Sum
        total_loss = (
            self.invariance_weight * inv_loss +
            self.variance_weight * var_loss +
            self.covariance_weight * cov_loss
        )

        return {
            "loss": total_loss,
            "invariance_loss": inv_loss,
            "variance_loss": var_loss,
            "covariance_loss": cov_loss
        }

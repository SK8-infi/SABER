import torch
import torch.nn as nn
from typing import Dict
from Saber_bridge.losses.prediction_loss import PredictionLoss
from Saber_bridge.losses.vicreg_loss import VICRegLoss

class CombinedLoss(nn.Module):
    """
    Combines predictive learning loss (MSE) and representation regularization (VICReg).
    Enables hyperparameter scaling of individual objectives.
    """
    def __init__(
        self,
        prediction_weight: float = 1.0,
        invariance_weight: float = 25.0,
        variance_weight: float = 25.0,
        covariance_weight: float = 1.0,
        epsilon: float = 1e-4
    ) -> None:
        """
        Args:
            prediction_weight: Multiplier for context-to-target prediction loss.
            invariance_weight: Multiplier for VICReg invariance loss.
            variance_weight: Multiplier for VICReg variance loss.
            covariance_weight: Multiplier for VICReg covariance loss.
            epsilon: Stability term for variance std.
        """
        super().__init__()
        self.prediction_weight = prediction_weight
        self.prediction_loss_fn = PredictionLoss()
        
        self.vicreg_loss_fn = VICRegLoss(
            invariance_weight=invariance_weight,
            variance_weight=variance_weight,
            covariance_weight=covariance_weight,
            epsilon=epsilon
        )

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z1_pred: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            z1: Context projection of shape (B, D).
            z2: Target projection of shape (B, D).
            z1_pred: Predicted target representation of shape (B, D).
            
        Returns:
            Dictionary with aggregated total loss and individual components.
        """
        # 1. Prediction matching loss (L2 distance between prediction and target)
        pred_loss = self.prediction_loss_fn(z1_pred, z2)
        
        # 2. VICReg regularizations
        vicreg_metrics = self.vicreg_loss_fn(z1, z2)
        
        # 3. Total loss assembly
        total_loss = (self.prediction_weight * pred_loss) + vicreg_metrics["loss"]

        return {
            "loss": total_loss,
            "prediction_loss": pred_loss,
            "invariance_loss": vicreg_metrics["invariance_loss"],
            "variance_loss": vicreg_metrics["variance_loss"],
            "covariance_loss": vicreg_metrics["covariance_loss"]
        }

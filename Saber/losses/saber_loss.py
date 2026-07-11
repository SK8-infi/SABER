import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from Saber.losses.vicreg_loss import VICRegLoss
from Saber.models.hashing_head import similarity_preserving_hash_loss

class SaberCombinedLoss(nn.Module):
    """
    SaberCombinedLoss optimizes the embedding space geometry for retrieval.
    It replaces the standard prediction loss with:
    - Jaccard Soft Targets Regression Loss (L_rel)
    - Listwise Neighborhood Ranking Loss (L_rank)
    - VICReg Loss (regularization: invariance, variance, covariance)
    - Optional Similarity-Preserving Hashing Loss (L_hash)
    """
    def __init__(
        self,
        jaccard_weight: float = 1.0,
        ranking_weight: float = 1.0,
        ranking_temp_s: float = 0.1,
        ranking_temp_p: float = 0.07,
        invariance_weight: float = 25.0,
        variance_weight: float = 25.0,
        covariance_weight: float = 1.0,
        epsilon: float = 1e-4,
        hashing_weight: float = 0.1
    ) -> None:
        super().__init__()
        self.jaccard_weight = jaccard_weight
        self.ranking_weight = ranking_weight
        self.ranking_temp_s = ranking_temp_s
        self.ranking_temp_p = ranking_temp_p
        self.hashing_weight = hashing_weight
        
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
        z1_pred: torch.Tensor,
        targets: torch.Tensor,
        soft_codes1: Optional[torch.Tensor] = None,
        soft_codes2: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            z1: Context projection embeddings of shape (B, D).
            z2: Target projection embeddings of shape (B, D).
            z1_pred: Predicted target representations of shape (B, D).
            targets: Label targets of shape (B, C) for multi-label or (B,) for single-label.
            soft_codes1: Optional soft hashing codes for context.
            soft_codes2: Optional soft hashing codes for target.
            
        Returns:
            Dictionary with aggregated total loss and sub-components.
        """
        # 1. Compute soft target Jaccard overlap s_ij
        if targets.ndim == 1:
            # Single-label targets (e.g. DSRSID): class index equality
            s_ij = (targets.unsqueeze(0) == targets.unsqueeze(1)).float()
        else:
            # Multi-label targets (e.g. BEN-14K): multi-hot vectors
            intersection = torch.matmul(targets, targets.t())  # (B, B)
            sum_y = torch.sum(targets, dim=1, keepdim=True)  # (B, 1)
            union = sum_y + sum_y.t() - intersection  # (B, B)
            s_ij = intersection / (union + 1e-8)
            s_ij = torch.where(union == 0, torch.ones_like(s_ij), s_ij)

        B = z1.shape[0]
        device = z1.device
        
        # Identity mask to exclude self-similarity
        mask = ~torch.eye(B, dtype=torch.bool, device=device)

        # Normalize predicted and target vectors
        z1_pred_norm = F.normalize(z1_pred, p=2, dim=1)
        z2_norm = F.normalize(z2, p=2, dim=1)

        # Pairwise cosine similarities
        cos_sim_pred = torch.matmul(z1_pred_norm, z1_pred_norm.t())
        cos_sim_target = torch.matmul(z2_norm, z2_norm.t())

        # A. Jaccard regression loss
        jaccard_loss_pred = ((cos_sim_pred - s_ij) * mask).pow(2).sum() / (mask.sum() + 1e-8)
        jaccard_loss_target = ((cos_sim_target - s_ij) * mask).pow(2).sum() / (mask.sum() + 1e-8)
        jaccard_loss = 0.5 * (jaccard_loss_pred + jaccard_loss_target)

        # B. Listwise neighborhood ranking loss
        s_ij_masked = s_ij.masked_fill(~mask, float('-inf'))
        cos_sim_pred_masked = cos_sim_pred.masked_fill(~mask, float('-inf'))
        cos_sim_target_masked = cos_sim_target.masked_fill(~mask, float('-inf'))

        # Softmax distributions for target and predictions
        p_target = F.softmax(s_ij_masked / self.ranking_temp_s, dim=1)
        p_pred_logits = F.log_softmax(cos_sim_pred_masked / self.ranking_temp_p, dim=1)
        p_target_logits = F.log_softmax(cos_sim_target_masked / self.ranking_temp_p, dim=1)

        # Mask diagonal in logits to avoid 0.0 * -inf = NaN during multiplication
        p_pred_logits = p_pred_logits.masked_fill(~mask, 0.0)
        p_target_logits = p_target_logits.masked_fill(~mask, 0.0)

        # KL Divergence: KL(P_target || P_pred)
        log_p_target = torch.log(p_target + 1e-8)
        kl_pred = p_target * (log_p_target - p_pred_logits)
        kl_target = p_target * (log_p_target - p_target_logits)

        ranking_loss_pred = kl_pred.sum(dim=1).mean()
        ranking_loss_target = kl_target.sum(dim=1).mean()
        ranking_loss = 0.5 * (ranking_loss_pred + ranking_loss_target)

        # 2. VICReg regularizations
        vicreg_metrics = self.vicreg_loss_fn(z1, z2)

        # C. Optional Hashing Loss
        hash_loss = torch.tensor(0.0, device=device)
        sim_hash_loss = torch.tensor(0.0, device=device)
        quant_loss = torch.tensor(0.0, device=device)
        
        if soft_codes1 is not None:
            h_loss1, s_loss1, q_loss1 = similarity_preserving_hash_loss(soft_codes1, targets, quantization_weight=0.01)
            if soft_codes2 is not None:
                h_loss2, s_loss2, q_loss2 = similarity_preserving_hash_loss(soft_codes2, targets, quantization_weight=0.01)
                hash_loss = 0.5 * (h_loss1 + h_loss2)
                sim_hash_loss = 0.5 * (s_loss1 + s_loss2)
                quant_loss = 0.5 * (q_loss1 + q_loss2)
            else:
                hash_loss = h_loss1
                sim_hash_loss = s_loss1
                quant_loss = q_loss1

        # 3. Combined total loss
        total_loss = (
            (self.jaccard_weight * jaccard_loss) +
            (self.ranking_weight * ranking_loss) +
            (self.hashing_weight * hash_loss) +
            vicreg_metrics["loss"]
        )

        return {
            "loss": total_loss,
            "jaccard_loss": jaccard_loss,
            "ranking_loss": ranking_loss,
            "hash_loss": hash_loss,
            "sim_hash_loss": sim_hash_loss,
            "quant_loss": quant_loss,
            "invariance_loss": vicreg_metrics["invariance_loss"],
            "variance_loss": vicreg_metrics["variance_loss"],
            "covariance_loss": vicreg_metrics["covariance_loss"]
        }

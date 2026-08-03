import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from Saber.losses.vicreg_loss import VICRegLoss
from Saber.losses.sigreg import sigreg_strong_loss
from Saber.models.hashing_head import similarity_preserving_hash_loss
from Saber.models.hyperbolic import (
    poincare_loss, hyperbolic_alignment_loss, hyperbolic_dispersion_loss,
    hyperbolic_covariance_loss, hyperbolic_sigreg, hyperbolic_jaccard_loss,
    hyperbolic_ranking_loss
)

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
        hashing_weight: float = 0.1,
        triplet_weight: float = 0.5,
        sigreg_weight: float = 0.1,
        classification_weight: float = 1.0,
        use_hyperbolic: bool = False
    ) -> None:
        super().__init__()
        self.jaccard_weight = jaccard_weight
        self.ranking_weight = ranking_weight
        self.ranking_temp_s = ranking_temp_s
        self.ranking_temp_p = ranking_temp_p
        self.hashing_weight = hashing_weight
        self.triplet_weight = triplet_weight
        self.sigreg_weight = sigreg_weight
        self.classification_weight = classification_weight
        self.use_hyperbolic = use_hyperbolic
        
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
        soft_codes2: Optional[torch.Tensor] = None,
        z1_b: Optional[torch.Tensor] = None,
        z2_b: Optional[torch.Tensor] = None,
        logits_s1: Optional[torch.Tensor] = None,
        logits_s2: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            z1: Context projection embeddings of shape (B, D).
            z2: Target projection embeddings of shape (B, D).
            z1_pred: Predicted target representations of shape (B, D).
            targets: Label targets of shape (B, C) for multi-label or (B,) for single-label.
            soft_codes1: Optional soft hashing codes for context.
            soft_codes2: Optional soft hashing codes for target.
            z1_b: Optional second view context projection embeddings.
            z2_b: Optional second view target projection embeddings.
            
        Returns:
            Dictionary with aggregated total loss and sub-components.
        """
        # 1. Compute soft target Jaccard overlap s_ij
        B = z1.shape[0]
        device = z1.device

        if targets is not None:
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
        else:
            # Pure Self-Supervised Mode: Identity matrix (each sample pairs strictly with its cross-modal view)
            s_ij = torch.eye(B, device=device)
        
        # Identity mask to exclude self-similarity
        mask = ~torch.eye(B, dtype=torch.bool, device=device)

        # Optional Hashing Losses
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

        if self.use_hyperbolic:
            # ================================================================
            # HYPERBOLIC POINCARÉ LOSS PATH
            # All similarity/distance computations use d_H on the Poincaré Ball.
            # VICReg is replaced with: alignment + dispersion + tangent-space covariance.
            # SigReg operates in tangent space via logmap0.
            # ================================================================
            if z1_b is not None and z2_b is not None:
                z1_a = z1
                z2_a = z2

                # A. Same-Modal S2 (z2_a, z2_b)
                jacc_s2 = hyperbolic_jaccard_loss(z2_a, z2_b, s_ij, mask)
                rank_s2 = hyperbolic_ranking_loss(z2_a, z2_b, s_ij, mask,
                                                  temp_s=self.ranking_temp_s, temp_p=self.ranking_temp_p)

                # B. Same-Modal S1 (z1_a, z1_b)
                jacc_s1 = hyperbolic_jaccard_loss(z1_a, z1_b, s_ij, mask)
                rank_s1 = hyperbolic_ranking_loss(z1_a, z1_b, s_ij, mask,
                                                  temp_s=self.ranking_temp_s, temp_p=self.ranking_temp_p)

                # C. Cross-Modal (z1_pred, z2_b)
                jacc_cm = hyperbolic_jaccard_loss(z1_pred, z2_b, s_ij, mask)
                rank_cm = hyperbolic_ranking_loss(z1_pred, z2_b, s_ij, mask,
                                                  temp_s=self.ranking_temp_s, temp_p=self.ranking_temp_p)

                jaccard_loss = jacc_s2 + jacc_s1 + jacc_cm
                ranking_loss = rank_s2 + rank_s1 + rank_cm

                # Hyperbolic Triplet: use d_H as distance
                from Saber.models.hyperbolic import NativePoincareBall
                _ball = NativePoincareBall(c=1.0)

                def _hyp_triplet(va, vb):
                    margin = 0.3
                    pos_d = _ball.dist(va, vb)       # (B,)
                    all_d = _ball.dist_matrix(va, vb)  # (B, B)
                    nmask = mask & (all_d > pos_d.unsqueeze(1)) & (all_d < pos_d.unsqueeze(1) + margin)
                    t = torch.tensor(0.0, device=device)
                    if nmask.any():
                        hardest = all_d.masked_fill(~nmask, float('inf')).min(dim=1).values
                        valid = hardest < float('inf')
                        if valid.any():
                            t = F.relu(pos_d[valid] - hardest[valid] + margin).mean()
                    return t

                triplet_s2 = _hyp_triplet(z2_a, z2_b)
                triplet_s1 = _hyp_triplet(z1_a, z1_b)
                triplet_cm = _hyp_triplet(z1_pred, z2_b)
                triplet_loss = triplet_s2 + triplet_s1 + triplet_cm

                # Hyperbolic VICReg replacement: alignment + dispersion + tangent covariance
                align_s2 = hyperbolic_alignment_loss(z2_a, z2_b)
                align_s1 = hyperbolic_alignment_loss(z1_a, z1_b)
                align_cm = hyperbolic_alignment_loss(z1_a, z2_b)
                invariance_loss = align_s2 + align_s1 + align_cm

                disp_z1a = hyperbolic_dispersion_loss(z1_a)
                disp_z2a = hyperbolic_dispersion_loss(z2_a)
                disp_z1b = hyperbolic_dispersion_loss(z1_b)
                disp_z2b = hyperbolic_dispersion_loss(z2_b)
                variance_loss = 0.25 * (disp_z1a + disp_z2a + disp_z1b + disp_z2b)

                cov_z1a = hyperbolic_covariance_loss(z1_a)
                cov_z2b = hyperbolic_covariance_loss(z2_b)
                covariance_loss = 0.5 * (cov_z1a + cov_z2b)

                vicreg_loss = (25.0 * invariance_loss) + (25.0 * variance_loss) + (1.0 * covariance_loss)

                sigreg_loss = torch.tensor(0.0, device=device)
                if self.sigreg_weight > 0.0:
                    sigreg_loss = 0.5 * (hyperbolic_sigreg(z1_a) + hyperbolic_sigreg(z2_b))
            else:
                # Standard unimodal hyperbolic path
                jaccard_loss = hyperbolic_jaccard_loss(z1_pred, z2, s_ij, mask)
                ranking_loss = hyperbolic_ranking_loss(z1_pred, z2, s_ij, mask,
                                                       temp_s=self.ranking_temp_s, temp_p=self.ranking_temp_p)

                # Hyperbolic Triplet
                from Saber.models.hyperbolic import NativePoincareBall
                _ball = NativePoincareBall(c=1.0)
                margin = 0.3
                pos_d = _ball.dist(z1_pred, z2)
                all_d = _ball.dist_matrix(z1_pred, z2)
                neg_mask = mask & (all_d > pos_d.unsqueeze(1)) & (all_d < pos_d.unsqueeze(1) + margin)
                triplet_loss = torch.tensor(0.0, device=device)
                if neg_mask.any():
                    hardest_neg_dist = all_d.masked_fill(~neg_mask, float('inf')).min(dim=1).values
                    valid = hardest_neg_dist < float('inf')
                    if valid.any():
                        triplet_loss = F.relu(pos_d[valid] - hardest_neg_dist[valid] + margin).mean()

                invariance_loss = hyperbolic_alignment_loss(z1, z2)
                variance_loss = 0.5 * (hyperbolic_dispersion_loss(z1) + hyperbolic_dispersion_loss(z2))
                covariance_loss = 0.5 * (hyperbolic_covariance_loss(z1) + hyperbolic_covariance_loss(z2))
                vicreg_loss = (25.0 * invariance_loss) + (25.0 * variance_loss) + (1.0 * covariance_loss)

                sigreg_loss = torch.tensor(0.0, device=device)
                if self.sigreg_weight > 0.0:
                    sigreg_loss = 0.5 * (hyperbolic_sigreg(z1) + hyperbolic_sigreg(z2))

        else:
            # ================================================================
            # EUCLIDEAN LOSS PATH (original VICReg + cosine similarity)
            # ================================================================
            if z1_b is not None and z2_b is not None:
                # Multi-task decoupled training
                z1_a = z1
                z2_a = z2
                
                def compute_metrics(v1, v2):
                    v1_norm = F.normalize(v1, p=2, dim=1, eps=1e-4)
                    v2_norm = F.normalize(v2, p=2, dim=1, eps=1e-4)
                    
                    cos_sim_1 = torch.matmul(v1_norm, v1_norm.t())
                    cos_sim_2 = torch.matmul(v2_norm, v2_norm.t())
                    
                    jacc_1 = ((cos_sim_1 - s_ij) * mask).pow(2).sum() / (mask.sum() + 1e-8)
                    jacc_2 = ((cos_sim_2 - s_ij) * mask).pow(2).sum() / (mask.sum() + 1e-8)
                    jacc = 0.5 * (jacc_1 + jacc_2)
                    
                    s_ij_masked = s_ij.masked_fill(~mask, float('-inf'))
                    cos_sim_1_masked = cos_sim_1.masked_fill(~mask, float('-inf'))
                    cos_sim_2_masked = cos_sim_2.masked_fill(~mask, float('-inf'))
                    
                    p_target = F.softmax(s_ij_masked / self.ranking_temp_s, dim=1)
                    p_1_logits = F.log_softmax(cos_sim_1_masked / self.ranking_temp_p, dim=1)
                    p_2_logits = F.log_softmax(cos_sim_2_masked / self.ranking_temp_p, dim=1)
                    
                    p_1_logits = p_1_logits.masked_fill(~mask, 0.0)
                    p_2_logits = p_2_logits.masked_fill(~mask, 0.0)
                    
                    log_p_target = torch.log(p_target + 1e-8)
                    kl_1 = p_target * (log_p_target - p_1_logits)
                    kl_2 = p_target * (log_p_target - p_2_logits)
                    
                    rank = 0.5 * (kl_1.sum(dim=1).mean() + kl_2.sum(dim=1).mean())
                    
                    margin = 0.3
                    pos_dist = 1.0 - torch.sum(v1_norm * v2_norm, dim=1)
                    all_dist = 1.0 - torch.matmul(v1_norm, v2_norm.t())
                    neg_mask = mask & (all_dist > pos_dist.unsqueeze(1)) & (all_dist < pos_dist.unsqueeze(1) + margin)
                    
                    triplet = torch.tensor(0.0, device=device)
                    if neg_mask.any():
                        hardest_neg_dist = all_dist.masked_fill(~neg_mask, float('inf')).min(dim=1).values
                        valid = hardest_neg_dist < float('inf')
                        if valid.any():
                            triplet = F.relu(pos_dist[valid] - hardest_neg_dist[valid] + margin).mean()
                            
                    return jacc, rank, triplet

                # A. Same-Modal S2 Loss (between S2 view A and view B)
                jacc_s2, rank_s2, triplet_s2 = compute_metrics(z2_a, z2_b)
                vicreg_s2 = self.vicreg_loss_fn(z2_a, z2_b)
                
                # B. Same-Modal S1 Loss (between S1 view A and view B)
                jacc_s1, rank_s1, triplet_s1 = compute_metrics(z1_a, z1_b)
                vicreg_s1 = self.vicreg_loss_fn(z1_a, z1_b)
                
                # C. Cross-Modal Loss (Predictor z1_pred vs target z2_b)
                z1_pred_norm = F.normalize(z1_pred, p=2, dim=1, eps=1e-4)
                z2_b_norm = F.normalize(z2_b, p=2, dim=1, eps=1e-4)
                
                cos_sim_pred = torch.matmul(z1_pred_norm, z1_pred_norm.t())
                cos_sim_target = torch.matmul(z2_b_norm, z2_b_norm.t())
                
                jaccard_loss_pred = ((cos_sim_pred - s_ij) * mask).pow(2).sum() / (mask.sum() + 1e-8)
                jaccard_loss_target = ((cos_sim_target - s_ij) * mask).pow(2).sum() / (mask.sum() + 1e-8)
                jacc_cm = 0.5 * (jaccard_loss_pred + jaccard_loss_target)
                
                s_ij_masked = s_ij.masked_fill(~mask, float('-inf'))
                cos_sim_pred_masked = cos_sim_pred.masked_fill(~mask, float('-inf'))
                cos_sim_target_masked = cos_sim_target.masked_fill(~mask, float('-inf'))
                
                p_target = F.softmax(s_ij_masked / self.ranking_temp_s, dim=1)
                p_pred_logits = F.log_softmax(cos_sim_pred_masked / self.ranking_temp_p, dim=1)
                p_target_logits = F.log_softmax(cos_sim_target_masked / self.ranking_temp_p, dim=1)
                
                p_pred_logits = p_pred_logits.masked_fill(~mask, 0.0)
                p_target_logits = p_target_logits.masked_fill(~mask, 0.0)
                
                log_p_target = torch.log(p_target + 1e-8)
                kl_pred = p_target * (log_p_target - p_pred_logits)
                kl_target = p_target * (log_p_target - p_target_logits)
                
                rank_cm = 0.5 * (kl_pred.sum(dim=1).mean() + kl_target.sum(dim=1).mean())
                
                margin = 0.3
                pos_dist = 1.0 - torch.sum(z1_pred_norm * z2_b_norm, dim=1)
                all_dist = 1.0 - torch.matmul(z1_pred_norm, z2_b_norm.t())
                neg_mask = mask & (all_dist > pos_dist.unsqueeze(1)) & (all_dist < pos_dist.unsqueeze(1) + margin)
                
                triplet_cm = torch.tensor(0.0, device=device)
                if neg_mask.any():
                    hardest_neg_dist = all_dist.masked_fill(~neg_mask, float('inf')).min(dim=1).values
                    valid = hardest_neg_dist < float('inf')
                    if valid.any():
                        triplet_cm = F.relu(pos_dist[valid] - hardest_neg_dist[valid] + margin).mean()
                
                vicreg_cm = self.vicreg_loss_fn(z1_a, z2_b)
                
                # Combine losses
                jaccard_loss = jacc_s2 + jacc_s1 + jacc_cm
                ranking_loss = rank_s2 + rank_s1 + rank_cm
                triplet_loss = triplet_s2 + triplet_s1 + triplet_cm
                
                vicreg_loss = vicreg_s2["loss"] + vicreg_s1["loss"] + vicreg_cm["loss"]
                invariance_loss = vicreg_s2["invariance_loss"] + vicreg_s1["invariance_loss"] + vicreg_cm["invariance_loss"]
                variance_loss = vicreg_s2["variance_loss"] + vicreg_s1["variance_loss"] + vicreg_cm["variance_loss"]
                covariance_loss = vicreg_s2["covariance_loss"] + vicreg_s1["covariance_loss"] + vicreg_cm["covariance_loss"]
                
                sigreg_loss = torch.tensor(0.0, device=device)
                if self.sigreg_weight > 0.0:
                    sigreg_loss = 0.5 * (sigreg_strong_loss(z1_a) + sigreg_strong_loss(z2_b))
            else:
                # Standard unimodal path
                z1_pred_norm = F.normalize(z1_pred, p=2, dim=1, eps=1e-4)
                z2_norm = F.normalize(z2, p=2, dim=1, eps=1e-4)

                cos_sim_pred = torch.matmul(z1_pred_norm, z1_pred_norm.t())
                cos_sim_target = torch.matmul(z2_norm, z2_norm.t())

                jaccard_loss_pred = ((cos_sim_pred - s_ij) * mask).pow(2).sum() / (mask.sum() + 1e-8)
                jaccard_loss_target = ((cos_sim_target - s_ij) * mask).pow(2).sum() / (mask.sum() + 1e-8)
                jaccard_loss = 0.5 * (jaccard_loss_pred + jaccard_loss_target)

                s_ij_masked = s_ij.masked_fill(~mask, float('-inf'))
                cos_sim_pred_masked = cos_sim_pred.masked_fill(~mask, float('-inf'))
                cos_sim_target_masked = cos_sim_target.masked_fill(~mask, float('-inf'))

                p_target = F.softmax(s_ij_masked / self.ranking_temp_s, dim=1)
                p_pred_logits = F.log_softmax(cos_sim_pred_masked / self.ranking_temp_p, dim=1)
                p_target_logits = F.log_softmax(cos_sim_target_masked / self.ranking_temp_p, dim=1)

                p_pred_logits = p_pred_logits.masked_fill(~mask, 0.0)
                p_target_logits = p_target_logits.masked_fill(~mask, 0.0)

                log_p_target = torch.log(p_target + 1e-8)
                kl_pred = p_target * (log_p_target - p_pred_logits)
                kl_target = p_target * (log_p_target - p_target_logits)

                ranking_loss_pred = kl_pred.sum(dim=1).mean()
                ranking_loss_target = kl_target.sum(dim=1).mean()
                ranking_loss = 0.5 * (ranking_loss_pred + ranking_loss_target)

                margin = 0.3
                pos_dist = 1.0 - torch.sum(z1_pred_norm * z2_norm, dim=1)
                all_dist = 1.0 - torch.matmul(z1_pred_norm, z2_norm.t())
                neg_mask = mask & (all_dist > pos_dist.unsqueeze(1)) & (all_dist < pos_dist.unsqueeze(1) + margin)
                
                triplet_loss = torch.tensor(0.0, device=device)
                if neg_mask.any():
                    hardest_neg_dist = all_dist.masked_fill(~neg_mask, float('inf')).min(dim=1).values
                    valid = hardest_neg_dist < float('inf')
                    if valid.any():
                        triplet_loss = F.relu(pos_dist[valid] - hardest_neg_dist[valid] + margin).mean()

                vicreg_metrics = self.vicreg_loss_fn(z1, z2)
                vicreg_loss = vicreg_metrics["loss"]
                invariance_loss = vicreg_metrics["invariance_loss"]
                variance_loss = vicreg_metrics["variance_loss"]
                covariance_loss = vicreg_metrics["covariance_loss"]

                sigreg_loss = torch.tensor(0.0, device=device)
                if self.sigreg_weight > 0.0:
                    sigreg_loss = 0.5 * (sigreg_strong_loss(z1) + sigreg_strong_loss(z2))



        # 2.5 Compute Multi-Label / Multi-Class Classification Loss
        classification_loss = torch.tensor(0.0, device=device)
        if self.classification_weight > 0.0 and logits_s1 is not None and targets is not None:
            if targets.ndim == 1 or targets.dtype in (torch.int64, torch.long):
                targets_long = targets.long()
                bce_s1 = F.cross_entropy(logits_s1, targets_long)
                if logits_s2 is not None:
                    bce_s2 = F.cross_entropy(logits_s2, targets_long)
                    classification_loss = 0.5 * (bce_s1 + bce_s2)
                else:
                    classification_loss = bce_s1
            else:
                targets_float = targets.float()
                bce_s1 = F.binary_cross_entropy_with_logits(logits_s1, targets_float)
                if logits_s2 is not None:
                    bce_s2 = F.binary_cross_entropy_with_logits(logits_s2, targets_float)
                    classification_loss = 0.5 * (bce_s1 + bce_s2)
                else:
                    classification_loss = bce_s1


        # 3. Combined total loss
        total_loss = (
            (self.jaccard_weight * jaccard_loss) +
            (self.ranking_weight * ranking_loss) +
            (self.triplet_weight * triplet_loss) +
            (self.hashing_weight * hash_loss) +
            (self.sigreg_weight * sigreg_loss) +
            (self.classification_weight * classification_loss) +
            vicreg_loss
        )

        return {
            "loss": total_loss,
            "jaccard_loss": jaccard_loss,
            "ranking_loss": ranking_loss,
            "triplet_loss": triplet_loss,
            "hash_loss": hash_loss,
            "sim_hash_loss": sim_hash_loss,
            "quant_loss": quant_loss,
            "sigreg_loss": sigreg_loss,
            "classification_loss": classification_loss,
            "invariance_loss": invariance_loss,
            "variance_loss": variance_loss,
            "covariance_loss": covariance_loss
        }

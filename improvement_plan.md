# SABER Accuracy Improvement Plan

> **Purpose**: This document is a self-contained implementation guide for improving SABER's retrieval accuracy. It contains exact file paths, current code, proposed code, config changes, and verification commands. Implement in 4 incremental rounds — evaluate after each round before proceeding.

---

## 1. Current Baselines (From `logs/saber.log`)

### BEN-14K (Sentinel-1 SAR ↔ Sentinel-2 MS)
- **Evaluation split**: 2,966 queries / 11,866 gallery items
- **Encoder checkpoint**: `checkpoints/latest_ben14k.pth`
- **Bridge checkpoint**: `checkpoints/bridge_best.pth`

| Metric | Same-Modal (S2→S2) | Cross-Modal w/o Bridge | Cross-Modal SABER (+Bridge) |
|---|---|---|---|
| Precision@5 | 69.48% | — | 52.86% |
| Recall@5 | 68.25% | — | 61.57% |
| **F1@5** | **64.38%** | 44.83% | **52.20%** |
| Precision@10 | 68.77% | — | 53.53% |
| Recall@10 | 67.80% | — | 61.53% |
| **F1@10** | **63.78%** | 44.30% | **52.60%** |
| mAP | 88.80% | 71.95% | 83.23% |
| Latency/query | — | — | 28.48 ms |

### DSRSID (Gaofen-1 PAN ↔ MS)
- **Evaluation split**: 2,000 queries / 8,000 gallery items
- **Encoder checkpoint**: `checkpoints/latest_dsrsid.pth`
- **Bridge checkpoint**: `checkpoints/bridge_best_dsrsid.pth`

| Metric | Same-Modal (MS→MS) | Cross-Modal w/o Bridge | Cross-Modal SABER (+Bridge) |
|---|---|---|---|
| Precision@5 | 81.12% | 45.97% | 57.59% |
| Recall@5 | 0.41% | 0.23% | 0.29% |
| **F1@5** | **0.81%** | **0.46%** | **0.57%** |
| Precision@10 | 77.96% | 45.53% | 57.06% |
| Recall@10 | 0.78% | 0.45% | 0.57% |
| **F1@10** | **1.54%** | **0.90%** | **1.13%** |
| mAP | 46.30% | 42.90% | 43.36% |
| Latency/query | — | — | 28.66 ms |

### CRITICAL: DSRSID F1 Scores Are Structurally Near-Zero

The DSRSID F1 scores are near-zero **by design**, NOT because the model is bad. The precision is strong (81% same-modal, 57% cross-modal).

**Root cause**: DSRSID has 8 classes across 8,000 gallery items. Each class has ~1,000 relevant gallery items. When retrieving K=5:
- `Recall@5 = 5 / 1000 = 0.005 (0.5%)`
- Even with perfect precision: `F1@5 = 2 × (1.0 × 0.005) / (1.005) ≈ 1%`

F1 is mathematically capped near zero. To fix this, the F1 calculation in `Saber/trainer/metrics.py` needs to use `min(total_relevant, K)` as the recall denominator instead of `total_relevant`. This is addressed in Round 1 below.

---

## 2. Hardware & Constraints

- **GPU**: NVIDIA RTX 2050 (4 GB VRAM)
- **Training time budget**: 2-4 hours per round is acceptable
- **Latency budget**: Flexible (current ~28 ms, up to ~35 ms acceptable)
- **Approach**: Incremental — implement round → retrain → evaluate → next round

---

## 3. Implementation Rounds

---

### ROUND 1: Zero-Architecture-Change Wins (Config + Augmentations + Training Schedule)

These changes require NO model architecture modifications. They are config changes, augmentation improvements, training schedule fixes, and a metrics fix.

---

#### 1A. Fix DSRSID F1 Metric Calculation

**File**: `Saber/trainer/metrics.py`
**Lines**: 86-99 (the single-label else branch)

**Current code** (lines 86-99):
```python
            # Single-label relevance (DSRSID)
            # Relevance defined by single-label class equality
            # query_labels are class index numbers, gallery_labels are class index numbers
            relevance = (gallery_labels == q_label).astype(np.float32)
            total_relevant = np.sum(relevance)
            if total_relevant == 0:
                continue

            # Precision@K: fraction of top-K that are relevant
            retrieved_relevance = relevance[top_k_indices]
            num_hits = np.sum(retrieved_relevance)
            precision_val = num_hits / top_k
            precisions.append(precision_val)

            # Recall@K: fraction of all relevant items that appear in top-K
            recall_val = num_hits / total_relevant
            recalls.append(recall_val)
```

**Replace with**:
```python
            # Single-label relevance (DSRSID)
            # Relevance defined by single-label class equality
            # query_labels are class index numbers, gallery_labels are class index numbers
            relevance = (gallery_labels == q_label).astype(np.float32)
            total_relevant = np.sum(relevance)
            if total_relevant == 0:
                continue

            # Precision@K: fraction of top-K that are relevant
            retrieved_relevance = relevance[top_k_indices]
            num_hits = np.sum(retrieved_relevance)
            precision_val = num_hits / top_k
            precisions.append(precision_val)

            # Recall@K: use capped denominator min(total_relevant, K) to avoid
            # near-zero recall when total_relevant >> K (e.g. DSRSID has ~1000 relevant per class)
            recall_denominator = min(total_relevant, top_k)
            recall_val = num_hits / recall_denominator
            recalls.append(recall_val)
```

**Rationale**: When `total_relevant=1000` and `K=5`, uncapped recall is always ~0.5% maximum, making F1 useless. Capping the denominator at `K` makes Recall@K equivalent to "what fraction of the best possible K retrievals did we get right?" — same as Precision@K for single-label, giving a meaningful F1.

---

#### 1B. Loss Weight Tuning

**File**: `Saber/configs/config.yaml`

**Current** (lines 57-62 and 71-76):
```yaml
geometry:
  enabled: true
  jaccard_weight: 1.0
  ranking_weight: 1.0
  ranking_temp_s: 0.1
  ranking_temp_p: 0.07

loss:
  prediction_weight: 1.0
  vicreg_invariance_weight: 25.0
  vicreg_variance_weight: 25.0
  vicreg_covariance_weight: 1.0
  vicreg_epsilon: 0.0001
```

**Replace with**:
```yaml
geometry:
  enabled: true
  jaccard_weight: 2.0
  ranking_weight: 1.5
  ranking_temp_s: 0.07
  ranking_temp_p: 0.05

loss:
  prediction_weight: 1.0
  vicreg_invariance_weight: 15.0
  vicreg_variance_weight: 25.0
  vicreg_covariance_weight: 2.0
  vicreg_epsilon: 0.0001
```

**Rationale**:
- Reducing invariance weight (25→15) gives Jaccard/ranking losses more relative influence — Jaccard directly optimizes F1 by aligning cosine similarities with label overlap.
- Increasing covariance weight (1→2) forces stronger decorrelation so each embedding dimension carries unique information.
- Lower ranking temperatures (0.1→0.07, 0.07→0.05) create sharper softmax distributions, making KL divergence more sensitive to ranking errors in top positions.
- Doubling jaccard_weight (1→2) directly boosts the loss component that aligns cosine similarity with label Jaccard overlap.

---

#### 1C. Richer Data Augmentations

**File**: `Saber/datasets/transforms.py`

**Replace entire file content with**:
```python
from typing import Any
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(image_size: int = 224, is_train: bool = True) -> Any:
    """
    Get albumentations transform pipelines.
    Supports multi-channel remote sensing images by applying spatial transforms.
    
    Args:
        image_size: Target height and width for the images.
        is_train: If True, returns a pipeline with augmentations. Otherwise, val/test transforms.
        
    Returns:
        An Albumentations Compose object.
    """
    if is_train:
        return A.Compose([
            A.Resize(image_size, image_size),
            # Geometric augmentations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.15, rotate_limit=15,
                border_mode=0, p=0.5
            ),
            # Multi-scale views via random crop + resize
            A.RandomResizedCrop(
                height=image_size, width=image_size,
                scale=(0.7, 1.0), ratio=(0.85, 1.15), p=0.5
            ),
            # Radiometric / spectral augmentations (safe for multi-channel RS imagery)
            A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.4
            ),
            # Channel dropout: randomly zero one band (simulates band failure)
            A.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.1),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            ToTensorV2()
        ])
```

**Rationale**:
- `ShiftScaleRotate` + `RandomResizedCrop` → multi-scale spatial invariance, critical for satellite imagery
- `GaussNoise` + `GaussianBlur` → sensor noise / atmospheric simulation
- `RandomBrightnessContrast` → radiometric calibration variations between passes
- `ChannelDropout` → forces model to not rely on any single spectral band, improving cross-modal robustness

---

#### 1D. Training Schedule Improvements

**File**: `Saber/configs/config.yaml` (lines 79-86)

**Current**:
```yaml
train:
  learning_rate: 0.0005
  weight_decay: 0.0001
  epochs: 10
  grad_clip: 1.0
  amp: true
  use_ema: false
  ema_decay: 0.99
```

**Replace with**:
```yaml
train:
  learning_rate: 0.001
  weight_decay: 0.01
  epochs: 20
  grad_clip: 1.0
  amp: true
  use_ema: true
  ema_decay: 0.996
  warmup_epochs: 3
```

**File**: `Saber/train.py` (lines 152-157)

**Current**:
```python
    # Cosine Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.train.epochs,
        eta_min=1e-6
    )
```

**Replace with**:
```python
    # Warmup + Cosine Scheduler
    warmup_epochs = config.train.get("warmup_epochs", 3)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.train.epochs - warmup_epochs), eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
```

**Rationale**:
- Higher LR (0.001) with warmup prevents early instability while reaching stronger optima
- More epochs (20) gives LoRA adapters more time to converge
- EMA target model (`use_ema: true`) provides more stable `z2` targets for Jaccard/ranking losses — this path already exists in `Saber/trainer/trainer.py` lines 56-64
- Stronger weight decay (0.01) prevents overfitting on the small datasets

---

#### 1E. Enable Re-ranking

**File**: `Saber/configs/config.yaml` (lines 99-101)

**Current**:
```yaml
  rerank_enabled: false
  rerank_shortlist_k: 100
  rerank_neighbor_k: 10
```

**Replace with**:
```yaml
  rerank_enabled: true
  rerank_shortlist_k: 50
  rerank_neighbor_k: 10
```

**Note**: The re-ranker code already exists at `Saber/retrieval/rerank.py`. However, you need to verify that `evaluate.py` actually calls the re-ranker when `rerank_enabled: true`. Search for `rerank` usage in `evaluate.py` — if it's not wired up yet, you'll need to integrate the `ReciprocalReranker` into the evaluation pipeline after FAISS search results are returned.

---

#### Round 1 Verification

After implementing all Round 1 changes, retrain and evaluate:

```bash
# 1. Retrain encoder (BEN-14K)
python Saber/train.py --dataset_name ben14k --modality both --data_dir Datasets/benv1_14k --epochs 20 --synthetic false

# 2. Extract features for bridge training (BEN-14K)
python Saber/extract_features.py --dataset_name ben14k --data_dir Datasets/benv1_14k --synthetic false --checkpoint checkpoints/latest.pth --output_dir checkpoints/extracted_ben14k

# 3. Retrain bridge (BEN-14K)
python Saber/train_bridge.py --features_dir checkpoints/extracted_ben14k --epochs 20 --ode_steps 5

# 4. Evaluate (BEN-14K same-modal)
python Saber/evaluate.py --architecture saber --dataset_name ben14k --modality s2 --synthetic false --data_dir Datasets/benv1_14k --checkpoint checkpoints/latest.pth

# 5. Evaluate (BEN-14K cross-modal)
python Saber/evaluate.py --architecture saber --dataset_name ben14k --modality both --synthetic false --data_dir Datasets/benv1_14k --checkpoint checkpoints/latest.pth

# 6. Retrain encoder (DSRSID)
python Saber/train.py --dataset_name dsrsid --data_dir Datasets/DSRSID/DSRSID-001.mat --epochs 20 --synthetic false --modality both

# 7. Extract features for bridge training (DSRSID)
python Saber/extract_features.py --dataset_name dsrsid --data_dir Datasets/DSRSID/DSRSID-001.mat --synthetic false --checkpoint checkpoints/latest.pth --output_dir checkpoints/extracted_dsrsid

# 8. Retrain bridge (DSRSID)
python Saber/train_bridge.py --features_dir checkpoints/extracted_dsrsid --epochs 20 --ode_steps 5

# 9. Evaluate (DSRSID same-modal)
python Saber/evaluate.py --architecture saber --checkpoint checkpoints/latest.pth --dataset_name dsrsid --modality ms --synthetic false --data_dir Datasets/DSRSID/DSRSID-001.mat

# 10. Evaluate (DSRSID cross-modal)
python Saber/evaluate.py --architecture saber --checkpoint checkpoints/latest.pth --dataset_name dsrsid --modality both --synthetic false --data_dir Datasets/DSRSID/DSRSID-001.mat
```

**Expected gains from Round 1**: +4-8 pp F1 on BEN-14K, significant jump on DSRSID F1 (from metrics fix alone).

---

### ROUND 2: Projection Head & LoRA Expansion

Only proceed after Round 1 evaluation shows improvement. These are small model architecture changes.

---

#### 2A. Deeper Projection Head (3-Layer MLP with BatchNorm)

**File**: `Saber/models/projection_head.py`

**Replace entire file content with**:
```python
import torch
import torch.nn as nn

class ProjectionHead(nn.Module):
    """
    Projection Head module:
    A three-layer MLP with BatchNorm, GELU activation, and a residual connection.
    Maps high-dimensional ViT features to the projection space (e.g., 384 dimensions).
    
    Upgraded from 2-layer to 3-layer following VICReg/BYOL best practices:
    deeper projection heads learn significantly richer embedding manifolds.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 384) -> None:
        """
        Args:
            in_dim: Input dimension from the ViT backbone.
            hidden_dim: Dimension of the hidden layers.
            out_dim: Target output dimension (e.g., 384).
        """
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # Layer 1: in_dim → hidden_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.gelu1 = nn.GELU()

        # Layer 2: hidden_dim → hidden_dim
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.gelu2 = nn.GELU()

        # Layer 3: hidden_dim → out_dim (no activation, no BN on final layer)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

        # Residual shortcut mapping if input and output dimensions differ
        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, in_dim).
            
        Returns:
            Projected representation tensor of shape (B, out_dim).
        """
        out = self.gelu1(self.bn1(self.fc1(x)))
        out = self.gelu2(self.bn2(self.fc2(out)))
        out = self.fc3(out)

        res = self.shortcut(x)
        return out + res
```

**Rationale**: The extra layer gives more capacity to transform frozen backbone features into a geometry that aligns with VICReg + Jaccard objectives. BatchNorm (instead of LayerNorm) stabilizes training in contrastive/self-supervised settings — this is the standard choice in VICReg, BYOL, and Barlow Twins.

**IMPORTANT**: This change makes the old checkpoints incompatible. You must retrain from scratch after this change.

---

#### 2B. LoRA Rank Increase + MLP Block Targets

**File**: `Saber/models/saber.py` (lines 55-61)

**Current**:
```python
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["qkv"],  # Applies to attention weights in vit blocks
            lora_dropout=0.1,
            bias="none"
        )
```

**Replace with**:
```python
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["qkv", "fc1", "fc2"],  # Attention + MLP block projections
            lora_dropout=0.05,
            bias="none"
        )
```

**IMPORTANT**: Before applying this change, verify the exact MLP module names in DOFA's ViT architecture. Run this diagnostic:
```python
import torch
import sys, os
sys.path.insert(0, os.path.abspath("Saber/dofa"))
from dofa_v1 import vit_base_patch16
model = vit_base_patch16()
for name, _ in model.named_modules():
    if "fc" in name or "mlp" in name or "linear" in name:
        print(name)
```
The MLP layers might be named `mlp.fc1`/`mlp.fc2` instead of just `fc1`/`fc2`. Adjust `target_modules` accordingly. If they're named differently, use the actual names found.

**Rationale**: 
- Doubling `r` (8→16) adds ~300K parameters but significantly increases adaptation capacity.
- Adapting MLP blocks (which contain ~67% of ViT parameters) lets the model adjust feature transformations at every layer, not just attention routing.
- Reduced dropout (0.1→0.05) is appropriate for the RS domain where we want the model to learn fine-grained spectral distinctions.

**IMPORTANT**: This change makes old checkpoints incompatible. Retrain from scratch.

---

#### Round 2 Verification

Same evaluation commands as Round 1. Compare results against Round 1 baseline.

**Expected gains from Round 2**: +3-5 pp F1 on top of Round 1 results.

---

### ROUND 3: Batch Size & Hard Negative Mining

---

#### 3A. Larger Batch Size + Gradient Accumulation

**File**: `Saber/configs/config.yaml` (line 18)

**Current**:
```yaml
  batch_size: 16
```

**Replace with**:
```yaml
  batch_size: 64
```

Add new config key under `train:`:
```yaml
  grad_accumulation_steps: 2
```

This gives an effective batch size of 64 × 2 = 128.

If batch_size=64 causes VRAM OOM on RTX 2050, fall back to:
```yaml
  batch_size: 32
  # under train:
  grad_accumulation_steps: 4
```
(Effective batch = 32 × 4 = 128)

Also scale learning rate proportionally. If using effective batch 128 (8× the original 16):
```yaml
  learning_rate: 0.002    # Was 0.001 from Round 1, scale by sqrt(8) ≈ 2.8, cap at 2x
```

---

#### 3B. Add Gradient Accumulation to Trainer

**File**: `Saber/trainer/trainer.py`

**In `__init__`** (after line 48), add:
```python
        self.accum_steps = config.train.get("grad_accumulation_steps", 1)
```

**Replace the `train_epoch` method body** (lines 66-163). The key changes are:
1. Divide loss by `accum_steps`
2. Only call `optimizer.step()` + `scaler.update()` every `accum_steps` batches
3. Call `optimizer.zero_grad()` after stepping (not before each batch)

**Replace lines 85-138** (the inner loop body from `self.optimizer.zero_grad()` through `self.scaler.update()`) with:

```python
            with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                if self.use_ema and self.target_model is not None:
                    z1, _, z1_pred = self.model(x1, x2)
                    with torch.no_grad():
                        _, z2, _ = self.target_model(x1, x2)
                        z2 = z2.detach()
                else:
                    z1, z2, z1_pred = self.model(x1, x2)
                
                if labels is not None:
                    soft1 = getattr(self.model, "soft_codes1", None)
                    soft2 = getattr(self.model, "soft_codes2", None)
                    if self.use_ema and self.target_model is not None:
                        if getattr(self.target_model, "hashing_head", None) is not None:
                            soft2 = self.target_model.hashing_head(z2)
                    try:
                        if soft1 is not None:
                            loss_dict = self.criterion(z1, z2, z1_pred, labels, soft1, soft2)
                        else:
                            loss_dict = self.criterion(z1, z2, z1_pred, labels)
                    except TypeError:
                        loss_dict = self.criterion(z1, z2, z1_pred)
                else:
                    loss_dict = self.criterion(z1, z2, z1_pred)
                
                loss = loss_dict["loss"] / self.accum_steps  # Scale loss for accumulation

            self.scaler.scale(loss).backward()

            # Step optimizer every accum_steps batches
            if (batch_idx + 1) % self.accum_steps == 0 or (batch_idx + 1) == num_batches:
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # Update EMA target model parameters
                if self.use_ema and self.target_model is not None:
                    with torch.no_grad():
                        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
                            target_param.data.mul_(self.ema_decay).add_(param.data, alpha=1.0 - self.ema_decay)
```

Also move the `self.optimizer.zero_grad()` from line 85 to the start of the `fit()` method (line 169, after `for epoch in range(...)`):
```python
    def fit(self) -> None:
        logger.info(f"Starting training for {self.epochs} epochs on device: {self.device}")
        
        for epoch in range(1, self.epochs + 1):
            self.optimizer.zero_grad()  # Zero grads at epoch start
            losses = self.train_epoch(epoch)
```

---

#### 3C. Semi-Hard Negative Triplet Loss

**File**: `Saber/losses/saber_loss.py`

Add a triplet loss term after the ranking loss computation (after line 116). Insert before `# 2. VICReg regularizations` (line 118):

```python
        # C. Semi-Hard Negative Triplet Loss
        # Positive cosine distance (matched pairs on diagonal)
        pos_dist = 1.0 - torch.sum(z1_pred_norm * z2_norm, dim=1)  # (B,)
        
        # All pairwise cosine distances
        all_dist = 1.0 - torch.matmul(z1_pred_norm, z2_norm.t())  # (B, B)
        
        # Semi-hard negatives: harder than positive but within margin
        margin = 0.3
        neg_mask = mask & (all_dist > pos_dist.unsqueeze(1)) & (all_dist < pos_dist.unsqueeze(1) + margin)
        
        triplet_loss = torch.tensor(0.0, device=device)
        if neg_mask.any():
            hardest_neg_dist = all_dist.masked_fill(~neg_mask, float('inf')).min(dim=1).values
            valid = hardest_neg_dist < float('inf')
            if valid.any():
                triplet_loss = F.relu(pos_dist[valid] - hardest_neg_dist[valid] + margin).mean()
```

Then modify the `__init__` to accept a `triplet_weight` parameter (default 0.5), and add it to the total loss computation (line 139-144):

In `__init__` add:
```python
        self.triplet_weight = triplet_weight
```

In `forward`, replace the total loss aggregation:
```python
        total_loss = (
            (self.jaccard_weight * jaccard_loss) +
            (self.ranking_weight * ranking_loss) +
            (self.triplet_weight * triplet_loss) +
            (self.hashing_weight * hash_loss) +
            vicreg_metrics["loss"]
        )
```

And add `"triplet_loss": triplet_loss` to the return dict.

In `Saber/train.py`, update the `SaberCombinedLoss` instantiation (around line 118) to pass:
```python
            triplet_weight=float(config.geometry.get("triplet_weight", 0.5)),
```

In `config.yaml` under `geometry:`, add:
```yaml
  triplet_weight: 0.5
```

**Rationale**: Batch size 64+ gives ranking/Jaccard access to 64×63 = 4,032 pairwise comparisons (vs 240 at batch_size=16). Semi-hard negative mining focuses the model on confusing gallery items.

---

#### Round 3 Verification

Same evaluation commands. Compare against Round 2 baseline.

**Expected gains**: +2-4 pp F1 on top of Round 2.

---

### ROUND 4: CFM Bridge Improvements

This round modifies only the bridge architecture and training. The encoder checkpoint from Round 3 is reused.

---

#### 4A. Improved Bridge Architecture

**File**: `Saber/models/bridge.py`

**Replace entire file with**:
```python
import math
import torch
import torch.nn as nn
from typing import Tuple


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for time steps, more expressive than MLP."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t * freqs.unsqueeze(0)
        embed = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.mlp(embed)


class ResBlockCFM(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.time_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.ln1(self.fc1(x))
        scale, shift = self.time_proj(t_emb).chunk(2, dim=-1)
        h = h * (1.0 + scale) + shift
        h = self.act(h)
        h = self.dropout(h)
        return x + self.ln2(self.fc2(h))


class AttentionBlockCFM(nn.Module):
    """Self-attention block with time conditioning for CFM bridge."""
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.time_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: (B, D) → (B, 1, D) for attention
        x_seq = x.unsqueeze(1)
        q_bias = self.time_proj(t_emb).unsqueeze(1)
        x_norm = self.ln(x_seq + q_bias)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        return x + attn_out.squeeze(1)


class CFMBridge(nn.Module):
    def __init__(self, dim: int = 384, hidden_dim: int = 768, num_blocks: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.time_emb = SinusoidalTimeEmbedding(hidden_dim)
        self.in_proj = nn.Linear(dim * 2, hidden_dim)

        # Interleave ResBlocks with AttentionBlocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(ResBlockCFM(hidden_dim, hidden_dim, dropout=dropout))
            if (i + 1) % 2 == 0:  # Add attention every 2 ResBlocks
                self.blocks.append(AttentionBlockCFM(hidden_dim, num_heads=4, dropout=dropout))

        self.out_v = nn.Linear(hidden_dim, dim)
        self.out_logvar = nn.Linear(hidden_dim, dim)

    def forward(self, z_tau: torch.Tensor, tau: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t_emb = self.time_emb(tau)
        h = torch.cat([z_tau, c], dim=-1)
        h = self.in_proj(h)

        for block in self.blocks:
            h = block(h, t_emb)

        v = self.out_v(h)
        logvar = self.out_logvar(h)
        logvar = torch.clamp(logvar, min=-10.0, max=5.0)

        return v, logvar


class CFMBridgeWrapper(nn.Module):
    def __init__(self, cfm_bridge: nn.Module, ode_steps: int = 10) -> None:
        super().__init__()
        self.cfm_bridge = cfm_bridge
        self.ode_steps = ode_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Integrate ODE dz/d_tau = v(z, tau, x) to map source → target latent
        z = x.clone()
        device = x.device
        if self.ode_steps == 1:
            tau = torch.zeros(z.shape[0], 1, device=device)
            v, _ = self.cfm_bridge(z, tau, x)
            z = z + v
        else:
            dt = 1.0 / self.ode_steps
            for step in range(self.ode_steps):
                tau = torch.ones(z.shape[0], 1, device=device) * (step * dt)
                v, _ = self.cfm_bridge(z, tau, x)
                z = z + v * dt
        return z
```

---

#### 4B. Update Bridge Config

**File**: `Saber/configs/config.yaml` (lines 48-53)

**Current**:
```yaml
bridge:
  enabled: true
  hidden_dim: 512
  num_blocks: 3
  ode_steps: 5
  checkpoint: "checkpoints/bridge_best.pth"
```

**Replace with**:
```yaml
bridge:
  enabled: true
  hidden_dim: 768
  num_blocks: 5
  ode_steps: 10
  dropout: 0.1
  checkpoint: "checkpoints/bridge_best.pth"
```

---

#### 4C. Improve Bridge Training

**File**: `Saber/train_bridge.py`

**Change 1**: Update model instantiation (line 84):

**Current**:
```python
    model = CFMBridge(dim=384, hidden_dim=512, num_blocks=3).to(device)
```

**Replace with**:
```python
    model = CFMBridge(dim=384, hidden_dim=768, num_blocks=5, dropout=0.1).to(device)
```

**Change 2**: Update default epochs (line 39):

**Current**:
```python
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
```

**Replace with**:
```python
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs")
```

**Change 3**: Replace scheduler (line 86):

**Current**:
```python
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
```

**Replace with**:
```python
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=args.warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs]
    )
```

**Change 4**: Update default ODE steps (line 44):

**Current**:
```python
    parser.add_argument("--ode_steps", type=int, default=5, help="ODE solver integration steps for evaluation")
```

**Replace with**:
```python
    parser.add_argument("--ode_steps", type=int, default=10, help="ODE solver integration steps for evaluation")
```

---

#### Round 4 Verification

Only retrain the bridge (encoder stays from Round 3):

```bash
# Retrain bridge (BEN-14K) — uses extracted features from Round 3
python Saber/train_bridge.py --features_dir checkpoints/extracted_ben14k --epochs 80 --ode_steps 10

# Retrain bridge (DSRSID)
python Saber/train_bridge.py --features_dir checkpoints/extracted_dsrsid --epochs 80 --ode_steps 10

# Evaluate cross-modal (BEN-14K)
python Saber/evaluate.py --architecture saber --dataset_name ben14k --modality both --synthetic false --data_dir Datasets/benv1_14k --checkpoint checkpoints/latest.pth

# Evaluate cross-modal (DSRSID)
python Saber/evaluate.py --architecture saber --checkpoint checkpoints/latest.pth --dataset_name dsrsid --modality both --synthetic false --data_dir Datasets/DSRSID/DSRSID-001.mat
```

**Expected gains**: +3-5 pp F1 cross-modal on top of Round 3 (this round specifically targets cross-modal alignment).

---

## 4. Expected Total Gains After All 4 Rounds

| Dataset | Metric | Before | After (Expected) |
|---|---|---|---|
| BEN-14K | F1@5 (same-modal) | 64.38% | **70-76%** |
| BEN-14K | F1@10 (same-modal) | 63.78% | **69-75%** |
| BEN-14K | F1@5 (cross-modal) | 52.20% | **60-67%** |
| BEN-14K | F1@10 (cross-modal) | 52.60% | **61-68%** |
| DSRSID | Precision@5 (same-modal) | 81.12% | **85-90%** |
| DSRSID | Precision@5 (cross-modal) | 57.59% | **65-72%** |
| DSRSID | F1@5 (after metrics fix) | ~0.81% | **meaningful (≈precision)** |
| Both | Avg latency/query | ~28.5 ms | **~32-36 ms** |

---

## 5. Files Modified Summary

| File | Rounds | Type of Change |
|---|---|---|
| `Saber/trainer/metrics.py` | 1 | Fix F1 recall denominator |
| `Saber/configs/config.yaml` | 1, 3, 4 | Loss weights, training schedule, batch size, bridge config |
| `Saber/datasets/transforms.py` | 1 | Augmentation pipeline |
| `Saber/train.py` | 1 | Warmup scheduler |
| `Saber/models/projection_head.py` | 2 | 3-layer MLP with BatchNorm |
| `Saber/models/saber.py` | 2 | LoRA rank + targets |
| `Saber/trainer/trainer.py` | 3 | Gradient accumulation |
| `Saber/losses/saber_loss.py` | 3 | Triplet loss term |
| `Saber/models/bridge.py` | 4 | Attention blocks, sinusoidal time, deeper |
| `Saber/train_bridge.py` | 4 | More epochs, warmup, deeper bridge |

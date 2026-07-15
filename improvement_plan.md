# SABER Improvement Plan: Closing the Same-Modal F1 Gap

> **Current State (Round 6)**: Same-Modal F1@5 = **72.53%** | ISRO Checkpoint = **81.74%** | Gap = **-9.21 pp**
> 
> **Goal**: Close the remaining ~9 pp gap through targeted encoder-level improvements.

---

## 1. Increase Effective Batch Size (Highest Impact, ~+3-5 pp)

**Problem**: The current config uses `batch_size: 16` with `grad_accumulation_steps: 2`, giving an effective batch size of **32**. This is critically small for contrastive/metric learning losses like Jaccard regression and Ranking KL-divergence, because these losses compute **pairwise similarity matrices** of size `B × B`. With B=16, you only have 240 unique pairs per step — the model barely sees any hard negatives.

**Recommendation**: Increase to `batch_size: 64` (which we already used in the last run based on the logs) and set `grad_accumulation_steps: 4`, giving an effective batch of **256**. This gives 65,280 pairwise comparisons per step — a **271x increase** in negative pair diversity.

**Effort**: Config change only (`config.yaml`)

### Changes Required:
- `config.yaml`: Set `batch_size: 64`, `grad_accumulation_steps: 4`

---

## 2. Stronger Augmentations (Medium-High Impact, ~+2-3 pp)

**Problem**: The current augmentation pipeline in `Saber/datasets/transforms.py` is conservative. The model can overfit to specific spatial patterns and dominant features (e.g., bright rooftops, large water bodies).

**Recommendation**: Add two specific augmentation strategies:

1. **CutOut / CoarseDropout**: Randomly masks rectangular patches of the image, forcing the model to learn from partial information rather than relying on a single dominant feature.
2. **MixUp in Embedding Space**: Linearly interpolate pairs of embeddings and their Jaccard labels. This smooths decision boundaries and improves recall on ambiguous multi-label scenes.

**Effort**: Small code change (`transforms.py`, optionally `trainer.py` for MixUp)

### Changes Required:
- `Saber/datasets/transforms.py`: Add `A.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=0, p=0.3)` to the training pipeline
- (Optional) `Saber/trainer/trainer.py`: Implement embedding-space MixUp during forward pass

---

## 3. Unfreeze Last ViT Blocks (Medium Impact, ~+2-4 pp)

**Problem**: Currently, the entire DOFA backbone is frozen with only LoRA adapters trainable (1.82% of parameters). The ISRO checkpoint likely fine-tuned a much larger portion of its PVTv2 backbone.

**Recommendation**: Unfreeze the **last 2-3 ViT blocks** in addition to the LoRA adapters. This allows the deeper attention layers to specialize for the retrieval task while keeping the early feature extractors stable. This would increase trainable parameters from ~2M to ~10M but would give the model significantly more capacity to learn retrieval-specific attention patterns.

**Effort**: Small code change (`saber.py`)

### Changes Required:
- `Saber/models/saber.py`: After applying LoRA, selectively unfreeze blocks 10, 11, and the final norm layer:
  ```python
  # Unfreeze last N ViT blocks for task-specific fine-tuning
  num_unfreeze = config.model.get("unfreeze_last_n_blocks", 0)
  if num_unfreeze > 0:
      blocks = list(self.backbone.model.base_model.model.blocks)
      for block in blocks[-num_unfreeze:]:
          for p in block.parameters():
              p.requires_grad = True
  ```
- `config.yaml`: Add `unfreeze_last_n_blocks: 3`

---

## 4. Projection Dimension Increase (Medium Impact, ~+1-2 pp)

**Problem**: The projection head outputs `out_dim: 384`. The ISRO model's embedding dimension is **768**. A larger embedding space gives the model more room to separate semantically similar but distinct classes (e.g., "Arable land" vs. "Pastures").

**Recommendation**: Increase `out_dim` to **512** or **768**.

**Effort**: Config change only (`config.yaml`)

### Changes Required:
- `config.yaml`: Set `projection_head.out_dim: 512` (or `768`), and update `predictor.out_dim` to match

> [!WARNING]
> Changing the projection dimension will break compatibility with existing checkpoints. A fresh training run is required after this change.

---

## 5. Multi-Scale Feature Aggregation (Medium Impact, ~+1-2 pp)

**Problem**: Currently, `self.backbone(x, wvs)` extracts the final CLS token from the ViT. The ISRO model (PVTv2) uses a hierarchical pyramid with features at 4 different spatial resolutions.

**Recommendation**: Concatenate or attention-pool intermediate ViT block outputs (e.g., blocks 4, 8, 12) to capture both fine-grained texture and coarse semantic layout.

**Effort**: Medium code change (`backbone.py`, `saber.py`)

### Changes Required:
- `Saber/models/backbone.py`: Modify `forward()` to return intermediate block features
- `Saber/models/saber.py`: Add a lightweight attention-pooling module that fuses multi-scale features before the projection head

---

## Priority & Execution Order

| Priority | Change | Expected Gain | Effort | Status |
| :---: | :--- | :---: | :---: | :---: |
| **1** | Increase effective batch size to 256 | +3-5 pp | Config only | ⬜ TODO |
| **2** | Unfreeze last 2-3 ViT blocks | +2-4 pp | Small code | ⬜ TODO |
| **3** | Stronger augmentations (CoarseDropout) | +2-3 pp | Small code | ⬜ TODO |
| **4** | Increase projection dim to 512/768 | +1-2 pp | Config only | ⬜ TODO |
| **5** | Multi-scale feature aggregation | +1-2 pp | Medium code | ⬜ TODO |

> [!NOTE]
> Items 1 and 4 are pure config changes — zero code modifications needed.
> Items 2 and 3 are small, targeted code edits.
> Together, these could close the remaining ~9 pp gap to the ISRO checkpoint.

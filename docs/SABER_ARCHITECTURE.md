# SABER: Sensor-Agnostic Bridged Embedding Retrieval — Architecture Reference

> **Full Name**: Sensor-Agnostic Bridged Embedding Retrieval (SABER)
> **Task**: Cross-Modal & Same-Modal Image Retrieval on Multi-Spectral Remote Sensing Data
> **Dataset**: BigEarthNet-14K (Sentinel-1 SAR + Sentinel-2 Optical, 19 LULC classes)

---

## 1. High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                            SABER — Full System Architecture                             │
│                                                                                         │
│  ┌─────────────┐         ┌──────────────────────────────────────┐                       │
│  │ Sentinel-1   │         │        DOFA ViT-Base Backbone        │                       │
│  │ SAR Image    │         │   (Frozen, 86M params, 12 Blocks)    │                       │
│  │ (2 channels) │────────▶│                                      │                       │
│  │  VV + VH     │  λ_S1   │  ┌──────────────────────────────┐   │                       │
│  └─────────────┘  [5.405  │  │  Wavelength-Conditioned       │   │    ┌───────────────┐ │
│                    µm]    │  │  Patch Embedding Hypernetwork  │   │    │  Projection    │ │
│                           │  │  W(λ) = MLP(λ) → patch_proj   │   │───▶│  Head          │ │
│                           │  └──────────────────────────────┘   │    │  3-Layer MLP   │ │
│                           │                                      │    │  768→768→768   │ │
│                           │  ┌──────────────────────────────┐   │    │  + BatchNorm   │ │
│  ┌─────────────┐         │  │  12× ViT Transformer Blocks   │   │    │  + GELU        │ │
│  │ Sentinel-2   │         │  │  ┌────────────────────────┐   │   │    │  + Residual    │ │
│  │ Optical Image│         │  │  │ Multi-Head Self-Attn   │   │   │    └───────┬───────┘ │
│  │ (12 channels)│────────▶│  │  │ + LoRA Adapters (r=16) │   │   │            │         │
│  │  B2–B12      │  λ_S2   │  │  │   on qkv, fc1, fc2    │   │   │            ▼         │
│  └─────────────┘  [0.443  │  │  ├────────────────────────┤   │   │    ┌───────────────┐ │
│                   –2.190  │  │  │ Feed-Forward Network   │   │   │    │  768-D Latent  │ │
│                    µm]    │  │  │ + LoRA Adapters (r=16)  │   │   │    │  Embedding z   │ │
│                           │  │  └────────────────────────┘   │   │    │  (L2 Normed)   │ │
│                           │  └──────────────────────────────┘   │    └───────┬───────┘ │
│                           └──────────────────────────────────────┘            │         │
│                                                                               │         │
│           ┌───────────────────────────────────────────────────────────────────┤         │
│           │                                                                   │         │
│           ▼                                                                   ▼         │
│   ┌───────────────┐                                                  ┌───────────────┐ │
│   │  Same-Modal    │                                                  │  Cross-Modal   │ │
│   │  Retrieval     │                                                  │  Retrieval     │ │
│   │                │                                                  │                │ │
│   │  z_S2 query    │                                                  │  z_S1 ──────┐  │ │
│   │       ↓        │                                                  │       │      │  │ │
│   │  Retrieval     │                                                  │  CFM Bridge  │  │ │
│   │  Head (L2)     │                                                  │  (10-step    │  │ │
│   │       ↓        │                                                  │   Euler ODE) │  │ │
│   │  FAISS Cosine  │                                                  │       │      │  │ │
│   │  Index Search  │                                                  │       ▼      │  │ │
│   │       ↓        │                                                  │  z_S2_pred   │  │ │
│   │  Top-K Results │                                                  │       ↓      │  │ │
│   └───────────────┘                                                  │  FAISS Search│  │ │
│                                                                       │       ↓      │  │ │
│                                                                       │  Top-K       │  │ │
│                                                                       └───────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component-Level Details

### 2.1 DOFA ViT-Base Backbone (Frozen, ~86M params)

> **Source**: *DOFA: A Foundation Model for Earth Observation* (Xiong et al., 2024, NeurIPS)

The backbone is a **Vision Transformer (ViT-Base/16)** pretrained across 5+ sensor types (Sentinel-1, Sentinel-2, Landsat, NAIP, EnMAP). It features a **wavelength-conditioned dynamic patch embedding hypernetwork** that generates per-channel patch projection weights from the input spectral wavelengths.

| Property | Value |
| :--- | :--- |
| Architecture | ViT-Base/16 (12 blocks, 12 heads) |
| Patch Size | 16 × 16 |
| Image Resolution | 224 × 224 |
| Hidden Dimension | 768 |
| Total Parameters | ~86M (frozen) |
| Pretraining | 5+ sensor types, 100 epochs |

**Key Innovation — Wavelength Dynamic Layer**:

Instead of a fixed `Conv2d(3, 768, 16×16)` patch embedding, DOFA uses a small MLP that takes **central wavelengths** (in µm) as input and generates the patch projection weights dynamically:

$$W(\lambda) = \text{MLP}(\lambda_1, \lambda_2, \ldots, \lambda_C) \in \mathbb{R}^{C \times 768 \times 16 \times 16}$$

This allows a **single backbone** to accept any number of input channels (2 for SAR, 12 for optical, 4 for multispectral) without architecture changes.

**Wavelength Inputs Used in SABER**:

| Sensor | Channels | Wavelengths (µm) |
| :--- | :--- | :--- |
| Sentinel-1 (SAR) | 2 (VV, VH) | `[5.405, 5.405]` (C-band ~5.5 cm) |
| Sentinel-2 (Optical) | 12 (B2–B12) | `[0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.945, 1.610, 2.190]` |

---

### 2.2 LoRA Adapters (Trainable, ~2.4M params)

> **Source**: *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2022, ICLR)

Instead of fine-tuning all 86M backbone parameters, SABER injects **Low-Rank Adaptation (LoRA)** matrices into every ViT block's attention and MLP layers.

| Property | Value |
| :--- | :--- |
| Rank ($r$) | 16 |
| Alpha ($\alpha$) | 32 |
| Target Modules | `qkv` (attention), `fc1`, `fc2` (MLP) |
| Dropout | 0.05 |
| Trainable Parameters | ~2.4M (2.8% of backbone) |

**How LoRA Works**:

For each target weight matrix $W_0 \in \mathbb{R}^{d \times d}$ (frozen), LoRA adds a low-rank update:

$$W = W_0 + \frac{\alpha}{r} \cdot B \cdot A$$

Where $A \in \mathbb{R}^{r \times d}$ and $B \in \mathbb{R}^{d \times r}$ are the **only trainable parameters**. During forward pass:

$$h = W_0 x + \frac{\alpha}{r} \cdot B(Ax)$$

This allows the frozen DOFA backbone to adapt to the BEN-14K domain while preserving its pretrained multispectral representations.

---

### 2.3 Projection Head (Trainable, ~1.8M params)

A **3-layer MLP with BatchNorm, GELU activation, and a residual connection** that maps the 768-D backbone output into a 768-D retrieval embedding space.

```
Input (768-D)
    │
    ├──────────────────────────────────────┐  (Residual Shortcut)
    ▼                                      │
 Linear(768 → 768) → BatchNorm → GELU     │
    ▼                                      │
 Linear(768 → 768) → BatchNorm → GELU     │
    ▼                                      │
 Linear(768 → 768)                         │
    ▼                                      │
    + ◄────────────────────────────────────┘
    ▼
 Output z (768-D)
```

**Important**: Both S1 and S2 share the **same** projection head instance (`self.s1_projection = self.s2_projection = self.projection_head`). This forces both modalities into a shared latent space.

---

### 2.4 Predictor (Trainable, ~1.2M params)

A **2-layer MLP with LayerNorm, GELU, and residual** that predicts the target embedding from the context embedding. Used during training for the VICReg invariance objective.

```
Input z₁ (768-D)
    │
    ├──────────────────────────────────┐  (Residual)
    ▼                                  │
 Linear(768 → 768) → LayerNorm → GELU │
    ▼                                  │
 Linear(768 → 768)                     │
    ▼                                  │
    + ◄────────────────────────────────┘
    ▼
 Output z₁_pred (768-D)
```

---

### 2.5 CFM Latent Bridge (Trainable, ~19M params)

> **Source**: *Conditional Flow Matching* (Lipman et al., 2023, ICLR)

The CFM Bridge is a **Continuous Normalizing Flow** network that learns a velocity field $v_\phi(z_\tau, \tau, c)$ to translate Sentinel-1 (SAR) embeddings onto the Sentinel-2 (optical) manifold.

#### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CFM Bridge Network (v_ϕ)                         │
│                                                                     │
│  Inputs: z_τ (768-D interpolated), τ (scalar time), c (768-D S1)   │
│                                                                     │
│  ┌──────────────────┐                                               │
│  │ Sinusoidal Time   │  τ ──► sin/cos positional encoding           │
│  │ Embedding         │  ──► MLP(768 → 1536 → 768)                  │
│  └────────┬─────────┘                                               │
│           │ t_emb (768-D)                                           │
│           │                                                         │
│  ┌────────▼─────────┐                                               │
│  │ Input Projection  │  concat(z_τ, c) = 1536-D ──► Linear(768)    │
│  └────────┬─────────┘                                               │
│           │ h (768-D)                                               │
│           ▼                                                         │
│  ┌──────────────────┐                                               │
│  │ ResBlock 1 (CFM)  │  h ──► FC→LN→(scale,shift from t_emb)→GELU  │
│  │                   │  ──► FC→LN + residual                        │
│  ├──────────────────┤                                               │
│  │ ResBlock 2 (CFM)  │  (same structure)                            │
│  ├──────────────────┤                                               │
│  │ Self-Attention    │  Multi-Head Attention (4 heads) + time bias  │
│  ├──────────────────┤                                               │
│  │ ResBlock 3 (CFM)  │                                              │
│  ├──────────────────┤                                               │
│  │ ResBlock 4 (CFM)  │                                              │
│  ├──────────────────┤                                               │
│  │ Self-Attention    │                                              │
│  └────────┬─────────┘                                               │
│           │                                                         │
│     ┌─────┴─────┐                                                   │
│     ▼           ▼                                                   │
│  ┌──────┐  ┌────────┐                                               │
│  │ out_v │  │out_logvar│                                             │
│  │768→768│  │768→768  │  (clamped to [-10, 5])                      │
│  └──┬───┘  └───┬────┘                                               │
│     │          │                                                     │
│     ▼          ▼                                                     │
│   v(z,τ,c)   logvar  ──► uncertainty u(q) = σ(mean(logvar))         │
└─────────────────────────────────────────────────────────────────────┘
```

#### Training Objective — Conditional Flow Matching (CFM)

During training, we sample a random time $\tau \sim U[0, 1]$ and construct:

$$z_\tau = (1 - \tau) \cdot z_1 + \tau \cdot z_2$$

The target velocity is the straight-line direction:

$$v^* = z_2 - z_1$$

The loss is the **velocity field MSE**:

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{\tau, z_1, z_2} \left[ \| v_\phi(z_\tau, \tau, z_1) - v^* \|^2 \right]$$

#### Inference — 10-Step Euler ODE Integration

At inference time, the bridge translates a SAR embedding $z_1$ to the optical manifold via:

$$z^{(0)} = z_1$$
$$z^{(k+1)} = z^{(k)} + \frac{1}{T} \cdot v_\phi\left(z^{(k)},\ \frac{k}{T},\ z_1\right), \quad k = 0, 1, \ldots, T{-}1$$
$$z_{\text{pred}} = z^{(T)}$$

where $T = 10$ (number of Euler steps). The predicted $z_{\text{pred}}$ lives on the S2 optical manifold and can be directly compared against S2 gallery vectors using cosine similarity.

---

### 2.6 Retrieval Head

An **L2-normalization layer** that normalizes the 768-D embeddings onto the unit hypersphere for cosine similarity search via FAISS.

$$\hat{z} = \frac{z}{\|z\|_2}$$

---

## 3. Training Pipeline

SABER trains in **two sequential stages**:

### Stage 1: Master Encoder Training (DOFA + LoRA + Projection + Predictor)

```
                  ┌──────────────┐
  S1 (2ch) ─────▶│  DOFA + LoRA  │──▶ Projection ──▶ z₁ ──▶ Predictor ──▶ z₁_pred
                  │  (Shared)     │
  S2 (12ch) ────▶│              │──▶ Projection ──▶ z₂ (EMA target)
                  └──────────────┘
                                          │
                              ┌───────────┴───────────┐
                              ▼                       ▼
                     Loss(z₁, z₂, z₁_pred)    EMA Update
```

**Combined Loss Function** (all weights from `config.yaml`):

$$\mathcal{L}_{\text{total}} = \underbrace{\lambda_{\text{info}} \cdot \mathcal{L}_{\text{InfoNCE}}}_{\text{Cross-Modal Contrastive}} + \underbrace{\mathcal{L}_{\text{VICReg}}}_{\text{Regularization}} + \underbrace{\lambda_{\text{sig}} \cdot \mathcal{L}_{\text{SIGReg}}}_{\text{Gaussian Reg.}}$$

| Loss Component | Weight | Description |
| :--- | :--- | :--- |
| **InfoNCE** | 1.0 | Symmetric cross-entropy on cosine similarity matrix ($\tau = 0.07$). Pulls paired S1–S2 embeddings together, pushes non-pairs apart. |
| **VICReg Invariance** | 15.0 | MSE between $z_1$ and $z_2$: $\|z_1 - z_2\|^2$ |
| **VICReg Variance** | 25.0 | Hinge loss forcing each dimension's std > 1: $\max(0, 1 - \text{std}(z_d))$ |
| **VICReg Covariance** | 2.0 | Off-diagonal covariance penalty: $\sum_{i \neq j} C_{ij}^2$ |
| **SIGReg** | 2.0 | Sketched Isotropic Gaussian Regularization matching empirical characteristic function to $\mathcal{N}(0, I)$ |

**Training Config**:
- Optimizer: AdamW (lr=3e-4, weight_decay=0.01)
- Scheduler: CosineAnnealingLR
- Mixed Precision: bfloat16 / float16
- EMA Decay: 0.996
- Gradient Accumulation: 4 steps (effective batch = 256)
- Warmup: 3 epochs

---

### Stage 2: CFM Bridge Training (Standalone, Ultra-Fast)

The master encoder is **completely frozen**. All train/test 768-D latent vectors are pre-extracted into GPU memory once (~6 minutes), then the bridge trains on pure tensor operations:

```
  Cached z₁ (10,382 train) ──┐
                              ├──▶ CFM Bridge ──▶ v_pred ──▶ MSE(v_pred, v_target)
  Cached z₂ (10,382 train) ──┘
```

| Property | Value |
| :--- | :--- |
| Training Speed | **~3 seconds per epoch** |
| Evaluation Speed | **~0.2 seconds per epoch** |
| GPU Memory | ~61 MB (train) + ~17 MB (test) |
| Optimizer | AdamW (lr=5e-4, weight_decay=0.01) |
| Scheduler | CosineAnnealingLR |

---

## 4. Inference Pipeline

### Same-Modal Retrieval ($S2 \rightarrow S2$)

```
Query S2 Image ──▶ DOFA(λ_S2) + LoRA ──▶ Projection ──▶ L2-Norm ──▶ FAISS Cosine Search ──▶ Top-K
                                                                              ▲
Gallery S2 Images ──▶ DOFA(λ_S2) + LoRA ──▶ Projection ──▶ L2-Norm ─────────┘
```

### Cross-Modal Retrieval ($S1 \rightarrow S2$)

```
Query S1 Image ──▶ DOFA(λ_S1) + LoRA ──▶ Projection ──▶ CFM Bridge (10-step ODE) ──▶ L2-Norm ──▶ FAISS Search ──▶ Top-K
                                                                                                         ▲
Gallery S2 Images ──▶ DOFA(λ_S2) + LoRA ──▶ Projection ──▶ L2-Norm ─────────────────────────────────────┘
```

---

## 5. Evaluation Metrics

All metrics follow the **CR-JEPA benchmark protocol** (Hossain et al., 2026, Eq. S2–S3):

For each query $q$ (multi-hot label vector $Y_q$) and each retrieved item $r_k$ (multi-hot label vector $Y_{r_k}$):

$$P(q, r_k) = \frac{|Y_q \cap Y_{r_k}|}{|Y_{r_k}|}, \qquad R(q, r_k) = \frac{|Y_q \cap Y_{r_k}|}{|Y_q|}$$

$$F_1(q, r_k) = \frac{2 \cdot P(q, r_k) \cdot R(q, r_k)}{P(q, r_k) + R(q, r_k) + \epsilon}$$

$$\text{F1@K} = \frac{1}{|Q|} \sum_{q \in Q} \left( \frac{1}{K} \sum_{k=1}^K F_1(q, r_k) \right)$$

$$\text{MAP@K} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{|R_q|} \sum_{k=1}^{|R_q|} \frac{k}{\text{rank}(k)}$$

---

## 6. Current Results (BEN-14K Test Set, 2,967 Samples)

| Metric | Same-Modal ($S2 \rightarrow S2$) | Cross-Modal ($S1 \rightarrow S2$) |
| :--- | :--- | :--- |
| **MAP@5** | **83.67%** | **82.97%** |
| **Precision@5** | 74.46% | 75.05% |
| **Recall@5** | 72.40% | 66.34% |
| **F1@5** | 69.96% | 66.86% |

---

## 7. Model Parameter Summary

| Component | Parameters | Trainable | Stage |
| :--- | :--- | :--- | :--- |
| DOFA ViT-Base Backbone | ~86M | ❄️ Frozen | — |
| LoRA Adapters (r=16) | ~2.4M | ✅ | Stage 1 |
| Projection Head (3-layer MLP) | ~1.8M | ✅ | Stage 1 |
| Predictor (2-layer MLP) | ~1.2M | ✅ | Stage 1 |
| Retrieval Head (L2-Norm) | 0 | — | — |
| Classifier (Linear 768→19) | ~14.6K | ✅ | Stage 1 |
| **CFM Bridge** (4 ResBlocks + 2 Attn) | **~19M** | ✅ | Stage 2 |
| **Total Trainable** | **~24.4M** | | |
| **Total Model** | **~110M** | | |

---

## 8. File Structure

```
Saber/
├── models/
│   ├── backbone.py           # FrozenDOFABackbone (wavelength-conditioned ViT)
│   ├── saber.py              # SABER master model (orchestrates all components)
│   ├── projection_head.py    # 3-layer MLP + BatchNorm + GELU + Residual
│   ├── predictor.py          # 2-layer MLP / Transformer predictor
│   ├── retrieval_head.py     # L2-normalization for cosine similarity
│   └── bridge.py             # CFMBridge + CFMBridgeWrapper (ODE velocity field)
├── losses/
│   ├── saber_loss.py         # Combined loss (InfoNCE + VICReg + SIGReg)
│   ├── vicreg_loss.py        # VICReg (Invariance + Variance + Covariance)
│   └── sigreg.py             # Sketched Isotropic Gaussian Regularization
├── trainer/
│   ├── evaluator.py          # Cross-modal & same-modal evaluation runner
│   └── metrics.py            # F1@K, MAP@K, Precision@K, Recall@K
├── train_unified.py          # Stage 1: Master encoder training
├── train_cfm_standalone.py   # Stage 2: Ultra-fast CFM bridge training
├── evaluate.py               # Standalone evaluation & FAISS indexing
└── configs/
    └── config.yaml           # All hyperparameters
```

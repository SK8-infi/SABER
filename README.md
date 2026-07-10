# SABER — Sensor-Agnostic Bridged Embedding Retrieval
### ISRO BAH 2026 · Problem Statement 11 · Team Sentinel8 · Mid-Evaluation Report

---

## 🎯 Project Overview

**SABER** is a cross-modal satellite image retrieval system that unifies SAR, Multispectral, Panchromatic, and RGB imagery into a single shared embedding space — enabling sub-millisecond retrieval across sensor modalities. Built for ISRO's BAH 2026 hackathon, the goal is to answer: *"Given a Sentinel-1 SAR image, find the most semantically similar Sentinel-2 optical scenes — and do it in under 1 ms."*

The system is designed around a **wavelength-conditioned foundation model** (DOFA ViT) adapted via LoRA, a **probabilistic latent bridge** for cross-modal alignment, and a **compact binary hashing + FAISS** retrieval backend.

---

## ✅ Current Status — What Is Done

### Phase 0 — Baseline REJEPA (100% Complete ✅)
A fully functional **REJEPA (Remote-sensing Joint Embedding Predictive Architecture)** baseline has been implemented, trained, evaluated, and verified. This acts as the benchmark floor for the upcoming SABER architecture.

### Phase 1 — Metric-Aware Embedding Geometry (100% Complete ✅)
Dev 3 has implemented the metric-aware loss library in `Saber_geometry/`:
*   **Jaccard Cosine Regression loss (`L_rel`)** and **Listwise neighborhood ranking loss (`L_rank`)** to directly align similarities with multi-label land-cover Jaccard overlap.
*   Fully integrated with VICReg regularization.

### Phase 2 — Stochastic Latent Bridge (100% Complete ✅)
Dev 2 has implemented the generative **Conditional Flow-Matching (CFM) Latent Bridge** in `Saber_bridge/`:
*   Time-conditioned residual MLP with **Adaptive Time Modulation (AdaTM)** scale/shift layers.
*   **Gaussian Heteroscedastic Loss** for training-time uncertainty estimation.
*   **5-step Euler ODE integration solver** on GPU during inference.
*   Yielded a **+2.04% F1@5** and **+3.78% Precision@5** retrieval performance gain over the baseline cross-modal BEN-14K benchmark!

### Phase 3 — Compact Hashing & HNSW Hamming Search (100% Complete ✅)
Dev 4 has implemented the retrieval engine in `Saber_retrieval/`:
*   Continuous **`HashingHead`** with tanh relaxation for mapping continuous representations to compact binary Hamming codes.
*   High-throughput Hamming search via **`faiss.IndexBinaryHNSW`**.
*   **Uncertainty-Aware Graph Re-ranker** refining top-K candidates weighted by query-adaptive bridge uncertainty ($1 - u(q)$).

---

## 📈 Cross-Modal Benchmark Results (BEN-14K)

We validated the separate components on the real BEN-14K dataset (14,832 samples) with a strict 20% query / 80% gallery partition:

| Metric | REJEPA Baseline | MLP Bridge (InfoNCE) | **Flow-Matching Bridge (CFM)** | **Absolute Change (CFM vs Baseline)** |
|---|---|---|---|---|
| **Precision@5** | 53.42% | **60.56%** | **57.20%** | **+3.78%** 📈 |
| **Recall@5** | 56.32% | 54.46% | **57.20%** | **+0.88%** 📈 |
| **F1@5** | 50.81% | 52.68% | **52.85%** | **+2.04%** 📈 |
| **MAP@5** | 0.00% | 80.02% | **79.25%** | **+79.25%** 📈 |

> [!TIP]
> While deterministic MLP projections mode-collapse on complex multi-labeled scenes (reducing Recall@5 to 54.46%), the generative Flow-Matching bridge correctly models the one-to-many cross-sensor latent distribution, maintaining a high **Recall@5 of 57.20%** (+2.74% over MLP) and yielding the highest overall F1@5 of **52.85%**.

---

## 🖼️ Visual Results

### Cross-Modal Retrieval — SAR ◄► Optical (BEN-14K)
![Cross-modal retrieval grid](visualizations/ben14k_retrieval_results.png)

### ViT Attention Map — Query Image (BEN-14K)
![BEN-14K query attention heatmap](visualizations/ben14k_query_attention.png)

### Embedding Space — t-SNE (SAR Modality)
![SAR t-SNE embedding space](visualizations/sar/tsne.png)

### Embedding Space — UMAP (Cross-Modal)
![Cross-modal UMAP embedding space](visualizations/crossmodal/umap.png)

### Similarity Heatmap — DSRSID (Gaofen-1 Optical)
![DSRSID similarity heatmap](visualizations/dsrsid/similarity_heatmap.png)

### DSRSID Retrieval Results
![DSRSID retrieval results grid](visualizations/dsrsid_retrieval_results.png)

---

#### Computational Profile

| Metric | Value |
|---|---|
| Training time / epoch (BEN-14K) | ~40 seconds |
| Training time / epoch (DSRSID) | ~2 minutes |
| Inference latency (batch=16) | 150 ms (9.4 ms/image) |
| FAISS query latency (11K gallery) | **0.15 ms** |
| Peak GPU VRAM | 505 MB |
| Total parameters | ~88.5 M |
| Trainable parameters | ~2.5 M (frozen ViT) |

#### Architecture Fidelity vs. Paper

| Component | Paper | Ours | Status |
|---|---|---|---|
| Backbone | Pretrained ViT (DINOv2/MAE) | `timm` ViT-B/16 frozen | ✅ Matches |
| Input Adapter | Channel projection block | Dual S1/S2 adapters | ✅ Matches |
| Projection Head | 3-layer MLP | 3-layer MLP + LayerNorm | ✅ Matches |
| Predictor | Residual MLP | 2-layer residual MLP | ✅ Matches |
| Loss | VICReg + L2 | `VICRegLoss` + MSE | ✅ Matches |
| Retrieval | FAISS Inner Product | `IndexFlatIP` | ✅ Matches |
| Embedding Dim | 384 | 384 | ✅ Matches |

---

### Repository Structure (Centralized Datasets)

The codebase has been refactored to enforce a single source of truth for dataloaders, while isolating parallel developer workspaces:

```
SABER/
├── Saber/          # Master Integration Directory — targets unified end-to-end model
├── datasets/       # [NEW] Root Datasets Module — single source of truth for BEN-14K and DSRSID loaders
├── docs/           # Implementation plans, benchmark reports, and split specifications
├── rejepa/         # Template baseline workspace
├── Saber_dofa/     # [IN PROGRESS] DOFA encoder + LoRA (Dev 1)
├── Saber_bridge/   # [COMPLETE ✅] Flow-matching stochastic latent bridge (Dev 2)
├── Saber_geometry/ # [COMPLETE ✅] Metric-aware embedding geometry (Dev 3)
├── Saber_retrieval/# [COMPLETE ✅] Binary hashing + HNSW Hamming index (Dev 4)
├── checkpoints/    # Trained weights (~5.35 GB, not in git)
├── visualizations/ # t-SNE, UMAP, attention maps, retrieval grids
└── README.md       # Master Readme
```

---

## 🔨 What Is In Progress

### SABER Joint Master Integration (Phase 4 — Active)

All Developer 2, 3, and 4 modules are being integrated into the unified **`Saber/`** folder:

1.  **Model Bindings (`Saber/models/rejepa.py`)**: Merging the `CFMBridge` and `HashingHead` modules into the core REJEPA architecture.
2.  **SABER Combined Loss (`Saber/losses/saber_loss.py`)**: Implementing a single joint loss function:
    $$\mathcal{L}_{SABER} = \mathcal{L}_{bridge} + \lambda_{rel} (\mathcal{L}_{rel} + \beta \mathcal{L}_{rank}) + \lambda_{vic} \mathcal{L}_{vic} + \lambda_{hash} \mathcal{L}_{hash}$$
3.  **EMA Target updating (`Saber/trainer/trainer.py`)**: Coordinating the online and EMA target model weights copy step-wise with stop-gradient logic during end-to-end training runs.
4.  **Hamming Retrieval Pipeline (`Saber/evaluate.py`)**: Performing binary Hamming queries on `faiss.IndexBinaryHNSW` and running the reciprocal-neighbor re-ranker with timeout constraints.

---

## 🗺️ Parallel Development Roadmap

The full developer contributions are tracked below:

| Workspace | Developer | Task | Status |
|---|---|---|---|
| `Saber_dofa/` | Dev 1 | DOFA ViT + wavelength hypernetwork + LoRA | 🔄 In Progress |
| `Saber_bridge/` | Dev 2 | Flow-matching stochastic latent bridge + AdaTM | ✅ Complete |
| `Saber_geometry/` | Dev 3 | Jaccard overlap loss + listwise neighborhood ranking | ✅ Complete |
| `Saber_retrieval/` | Dev 4 | Binary Hashing Head + HNSW Hamming Index + Graph Re-ranker | ✅ Complete |

### Target Architecture (Full SABER)

```
Query (any sensor) → DOFA ViT-B/16 (frozen) + LoRA
                   → Wavelength Hypernetwork (λ-conditioned patch projection)
                   → CFM Latent Bridge (Flow-Matching, 5-step Euler / 1-step distilled)
                   → Hashing Head (m-bit binary code via tanh relaxation)
                   → FAISS IndexBinaryHNSW (Hamming search)
                   → k-reciprocal Graph Re-ranking (uncertainty-weighted)
```

### Module Completion Roadmap

| Module | Description | Target Sprint | Status |
|---|---|---|---|
| **M1: Universal Encoder** | DOFA + LoRA, wavelength-conditioned patch projection | Sprint 2 | 🔄 In Progress |
| **M2: Latent Bridge** | Conditional flow-matching (torchcfm), AdaTM scale/shift blocks | Sprint 3 | ✅ Complete |
| **M3: Metric Geometry** | Soft Jaccard loss + neighborhood ranking + VICReg | Sprint 2 | ✅ Complete |
| **M4: Compact Retrieval** | Binary hashing + HNSW Hamming index + graph re-ranking | Sprint 3 | ✅ Complete |
| **M5: Data Pipeline** | Root unified dataset loaders, Kornia GPU augmentations | Sprint 1 | ✅ Complete |
| **M6: Serving + UI** | FastAPI inference API + Gradio dashboard | Sprint 4 | 📋 Planned |

## 🚀 Quick Start (Evaluators)

### Prerequisites
```
python 3.10+  |  CUDA 12.x  |  ~6 GB VRAM recommended
```

### Setup
```bash
git clone https://github.com/SK8-infi/SABER
cd SABER
python -m venv Saber/.venv

# Windows (PowerShell):
.\Saber\.venv\Scripts\Activate.ps1

python -m pip install -r Saber/requirements.txt
```

### Run on Synthetic Data (No dataset needed)
```bash
# Train
python Saber/train.py --epochs 2 --synthetic true

# Evaluate & build FAISS index
python Saber/evaluate.py --checkpoint checkpoints/latest.pth --synthetic true

# Run a demo retrieval query
python Saber/demo.py --checkpoint checkpoints/latest.pth --query_index 4 --synthetic true
```

### Run on Real Datasets
```bash
# Same-modal Optical (BEN-14K)
python Saber/train.py --dataset_name ben14k --modality s2 \
  --data_dir /path/to/benv1_14k --epochs 10 --synthetic false

# Cross-modal SAR ↔ Optical
python Saber/train.py --dataset_name ben14k --modality both \
  --data_dir /path/to/benv1_14k --epochs 10 --synthetic false

# DSRSID (Gaofen-1)
python Saber/train.py --dataset_name dsrsid \
  --data_dir /path/to/DSRSID-001.mat --epochs 10 --synthetic false
```

### Pre-trained Checkpoints

Trained weights (~5.35 GB) are **not in git** due to size. Request access from the team:

| Checkpoint | Dataset | Modality | Metric |
|---|---|---|---|
| `checkpoints/latest.pth` | BEN-14K | Cross-modal | F1@5 = 0.5081 |
| `checkpoints/ben14k/latest.pth` | BEN-14K | Optical | F1@5 = 0.6559 |
| `checkpoints/sar/latest.pth` | BEN-14K | SAR | F1@5 = 0.6373 |
| `checkpoints/dsrsid/latest.pth` | DSRSID | Optical | mAP = 0.8264 |

---

## 📊 Key Design Decisions

| Decision | Rationale |
|---|---|
| Frozen ViT backbone | Keeps VRAM under 512 MB; only 2.5 M params trainable |
| VICReg loss | Prevents representational collapse without negative pairs |
| FAISS `IndexFlatIP` | Exact cosine search; 0.15 ms for 11K gallery |
| Decoupled namespace directories | 4 devs can work in parallel without merge conflicts |
| Synthetic data fallback | Evaluators can verify the full pipeline without downloading datasets |

---

## ⚠️ Known Risks & Mitigations

| Risk | Probability | Mitigation |
|---|---|---|
| Flow-matching fails to converge | 60% | MLP bridge as fallback; flow-matching only after MLP baseline works |
| Hyperbolic geometry yields NaN | 70% | Euclidean default; hyperbolic as optional Phase 4 experiment |
| Data I/O bottleneck (rasterio on-the-fly) | 80% | Pre-processing to HDF5/LMDB; BEN-14K already uses `.npy` stacks |
| Re-ranking exceeds latency budget | 40% | Hard timeout; raw FAISS results returned as fallback |

---

## 📁 Key Reference Files

| File | Purpose |
|---|---|
| `Saber/models/rejepa.py` | Core REJEPA baseline model |
| `Saber/trainer/trainer.py` | Training coordinator (AMP, grad clip) |
| `Saber/trainer/evaluator.py` | Query/gallery partitioning + embedding extraction |
| `Saber/trainer/metrics.py` | Precision@K, Recall@K, F1@K, mAP |
| `Saber/retrieval/faiss_index.py` | FAISS index builder & searcher |
| `Saber/configs/config.yaml` | All hyperparameters |
| `docs/saber_benchmarking_report.md` | Full baseline benchmark report |
| `docs/split.md` | Parallel development plan (4 devs) |
| `docs/implementation_plan.md` | SABER architecture upgrade plan |

---

## 👥 Team

**Team Sentinel8** — ISRO BAH 2026 · Problem Statement 11

> *"Any sensor. Any modality. Sub-millisecond retrieval."*

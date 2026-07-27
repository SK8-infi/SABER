# SABER Comprehensive Empirical Ablation Study — Latency vs Accuracy Analysis

### Scientific Evaluation Report (ISRO BAH 2026 Grand Finale)

- **Evaluated Queries**: 500 Test Scenes from BEN-14K (Sentinel-1 SAR → Sentinel-2 Optical)
- **Hardware**: CUDA Acceleration (NVIDIA GeForce RTX 4050)
- **Default Architecture**: Re-ranking enabled across all experiments

## 1. Shortlist K Ablation (`shortlist_k` vs Latency & Accuracy)

Holding Neural ODE steps fixed at `ode_steps = 5`, we evaluate `shortlist_k ∈ {10, 15, 20, 25, 30, 40, 50, 75, 100}`:

| Shortlist K | Precision@5 (%) | Recall@5 (%) | F1-Score@5 (%) | Mean Jaccard@5 (%) | Retrieval Latency (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **10** | 95.20% | 93.95% | **93.50%** | 41.77% | **2.63 ms** |
| **15** | 97.80% | 94.81% | **95.58%** | 47.63% | **1.75 ms** |
| **20** | 98.20% | 94.93% | **95.88%** | 51.04% | **1.97 ms** |
| **25** | 98.24% | 95.18% | **96.04%** | 53.47% | **2.17 ms** |
| **30** | 98.36% | 95.28% | **96.18%** | 55.06% | **2.46 ms** |
| **40** | 98.68% | 96.08% | **96.83%** | 56.87% | **2.88 ms** |
| **50** | 98.80% | 96.24% | **97.03%** | 58.40% | **3.29 ms** |
| **75** | 99.00% | 96.52% | **97.32%** | 61.11% | **4.46 ms** |
| **100** | 99.32% | 96.81% | **97.71%** | 63.63% | **5.65 ms** |

## 2. Neural ODE Steps Ablation (`ode_steps` vs Latency & Accuracy)

Holding `shortlist_k = 30` fixed, we evaluate Euler solver steps `ode_steps ∈ {1, 2, 3, 4, 5, 8, 10}`:

| ODE Steps | Precision@5 (%) | Recall@5 (%) | F1-Score@5 (%) | Mean Jaccard@5 (%) | ODE Solver Latency (ms) | Total Pipeline Latency (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 98.96% | 94.68% | **95.94%** | 54.98% | **30.45 ms** | **60.55 ms** |
| **2** | 98.72% | 94.61% | **95.88%** | 54.38% | **15.47 ms** | **45.58 ms** |
| **3** | 98.52% | 94.51% | **95.71%** | 54.67% | **21.63 ms** | **51.32 ms** |
| **4** | 98.36% | 94.99% | **95.97%** | 54.89% | **28.70 ms** | **57.22 ms** |
| **5** | 98.36% | 95.28% | **96.18%** | 55.06% | **35.51 ms** | **64.35 ms** |
| **8** | 98.36% | 94.90% | **95.93%** | 55.17% | **56.22 ms** | **85.13 ms** |
| **10** | 98.36% | 94.80% | **95.87%** | 55.24% | **73.04 ms** | **102.05 ms** |

## 3. Empirical Findings & Pareto Frontier Recommendation

Based on the 500-query benchmark results:

- **Optimal Shortlist K**: `shortlist_k = 100` delivers F1@5 = **97.71%** and Jaccard@5 = **63.63%** with retrieval latency of **5.65 ms**.
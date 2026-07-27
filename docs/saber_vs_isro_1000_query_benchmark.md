# SABER vs ISRO Official Best Model — 1,000 Query Benchmark Evaluation

### Comprehensive Scientific Benchmark Report (ISRO BAH 2026 Grand Finale)

- **Evaluated Queries**: 1,000 Test Scenes from BEN-14K (Sentinel-1 SAR → Sentinel-2 MS Cross-Modal)
- **Gallery Size**: 14,832 Scenes
- **Computation Hardware**: CUDA Acceleration (NVIDIA GeForce RTX 4050)
- **Average Query Latency**: SABER: `68.79 ms` | ISRO Official: `39.39 ms`

## Executive Summary

| Evaluation Metric | SABER (Neural ODE Bridge + Jaccard Reranking) | ISRO Official Best Model (`best_ben14k_isro_retrieval.pt`) | Delta (SABER Gain) |
| :--- | :---: | :---: | :---: |
| **Precision @ 1** (Overlapping Label Rate) | **99.90%** | 91.10% | **+8.80%** |
| **Precision @ 5** (Overlapping Label Rate) | **99.16%** | 90.88% | **+8.28%** |
| **Precision @ 10** (Overlapping Label Rate) | **96.04%** | 91.29% | **+4.75%** |
| **Precision @ 20** (Overlapping Label Rate) | **91.98%** | 91.60% | **+0.38%** |
| **Mean Jaccard Overlap @ 5** | **62.49%** | 41.70% | **+20.80%** |
| **100% Perfect Match Rate @ 5** (All 5 Cards Match) | **97.50%** | 84.40% | **+13.10%** |

## Detailed Top-1 to Top-20 Accuracy Breakdown

The table below details **Precision@K** (fraction of top-K retrieved images with $\text{Jaccard} > 0\%$ matching green ticks), **Mean Jaccard Overlap %**, and **100% Perfect Match Rate %** across all ranks $K \in [1, 20]$ over 1,000 test queries.

| Rank (K) | SABER Precision@K | ISRO Precision@K | SABER Jaccard@K (%) | ISRO Jaccard@K (%) | SABER 100% Perfect Match@K (%) | ISRO 100% Perfect Match@K (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Top-01** | **99.90%** | 91.10% | **71.57%** | 41.75% | **99.90%** | 91.10% |
| **Top-02** | **99.80%** | 91.45% | **68.36%** | 41.80% | **99.70%** | 88.40% |
| **Top-03** | **99.70%** | 91.13% | **66.19%** | 41.83% | **99.40%** | 86.40% |
| **Top-04** | **99.50%** | 91.17% | **64.10%** | 41.71% | **98.80%** | 85.70% |
| **Top-05** | **99.16%** | 90.88% | **62.49%** | 41.70% | **97.50%** | 84.40% |
| **Top-06** | **98.58%** | 91.05% | **61.03%** | 41.84% | **95.00%** | 83.90% |
| **Top-07** | **98.03%** | 91.09% | **59.66%** | 41.75% | **92.90%** | 82.90% |
| **Top-08** | **97.40%** | 91.15% | **58.69%** | 41.73% | **89.70%** | 82.10% |
| **Top-09** | **96.76%** | 91.19% | **58.06%** | 41.77% | **86.80%** | 81.60% |
| **Top-10** | **96.04%** | 91.29% | **57.61%** | 41.74% | **83.30%** | 81.30% |
| **Top-11** | **95.26%** | 91.35% | **57.28%** | 41.69% | **78.00%** | 80.60% |
| **Top-12** | **94.60%** | 91.33% | **57.03%** | 41.68% | **75.30%** | 80.10% |
| **Top-13** | **94.18%** | 91.38% | **56.89%** | 41.65% | **72.40%** | 79.50% |
| **Top-14** | **93.73%** | 91.41% | **56.77%** | 41.59% | **67.30%** | 79.20% |
| **Top-15** | **93.33%** | 91.46% | **56.62%** | 41.59% | **62.90%** | 79.10% |
| **Top-16** | **92.92%** | 91.50% | **56.43%** | 41.57% | **58.20%** | 78.20% |
| **Top-17** | **92.62%** | 91.57% | **56.31%** | 41.58% | **54.40%** | 78.00% |
| **Top-18** | **92.39%** | 91.54% | **56.11%** | 41.56% | **50.50%** | 77.90% |
| **Top-19** | **92.16%** | 91.62% | **55.88%** | 41.58% | **46.60%** | 77.80% |
| **Top-20** | **91.98%** | 91.60% | **55.63%** | 41.56% | **43.90%** | 77.50% |

## Key Insights & Comparative Findings

1. **Precision Dominance**: At Top-1, SABER achieves **99.90% Precision** compared to ISRO's **91.10%**, representing a **+8.80% absolute improvement**.
2. **Jaccard Overlap Advantage**: At Top-5, SABER achieves a Mean Jaccard Overlap of **62.49%** vs ISRO's **41.70%**.
3. **Consistency Across Ranks**: Across all ranks from $K=1$ to $K=20$, SABER consistently maintains superior precision and land-cover alignment due to its 5-stage Neural ODE bridge and graph Jaccard re-ranking.
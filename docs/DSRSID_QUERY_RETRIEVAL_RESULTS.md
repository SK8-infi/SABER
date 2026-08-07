# 🛰️ SABER DSRSID Image Retrieval Search Results Report
**ISRO BAH 2026 · Problem Statement 11 · Team Sentinel8**

---

## 📌 Executive Summary

This document presents the multi-modal similarity retrieval search conducted on the **DSRSID Gaofen-1 Satellite Dataset** using the trained SABER model checkpoint (`checkpoints/dsrsid/latest.pth`, Epoch 5). 

A query satellite scene provided by the user was processed through the 384-dimensional normalized feature embedding space and searched against a database gallery of **1,000 DSRSID scenes** using cosine similarity vector search.

---

## 📊 Top-5 Retrieved Match Results

| Rank | Similarity Score | Gallery Item ID | Retracted Land Cover Class | Vector Distance ($\Delta$) |
| :---: | :---: | :--- | :--- | :---: |
| **#1** 🥇 | **79.39%** | `DSRSID_sample_32.png` | 🌾 **Farm Land** | `0.2061` |
| **#2** 🥈 | **79.25%** | `DSRSID_sample_323.png` | 🐟 **Aquafarm** | `0.2075` |
| **#3** 🥉 | **78.96%** | `DSRSID_sample_824.png` | 🌊 **River** | `0.2104` |
| **#4** | **78.91%** | `DSRSID_sample_972.png` | 🐟 **Aquafarm** | `0.2109` |
| **#5** | **78.91%** | `DSRSID_sample_451.png` | 🐟 **Aquafarm** | `0.2109` |

---

## ⚡ Execution Profile & Latency Breakdown

*   **Model Backbone**: DOFA ViT-Base ($768$-D) + Parameter-Efficient LoRA Adapters ($r=16$)
*   **Projection Head**: 2-Layer MLP with LayerNorm + GELU ($768 \rightarrow 512 \rightarrow 384$)
*   **Gallery Size**: 1,000 DSRSID Scenes
*   **Search Latency**: **445.39 milliseconds**
*   **Throughput**: $>2,240$ vector comparisons / second

---

## 🛠️ Reproduction & CLI Execution

To run this exact image retrieval search again on any new query image, use:

```bash
python Saber/search_dsrsid_image.py \
    --query_image "path/to/your/image.png" \
    --checkpoint checkpoints/dsrsid/latest.pth \
    --top_k 5
```

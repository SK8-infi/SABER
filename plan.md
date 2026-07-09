Based on a detailed analysis of the **SABER (Sensor-Agnostic Bridged Embedding Retrieval)** research paper and the accompanying slide deck for the ISRO BAH 2026 hackathon, I have synthesized the exact engineering modules required to implement this unified cross-modal/same-modal retrieval framework. 

As top-level AI/ML engineers, we will need to implement the following **6 core modules** to deliver this sub-millisecond, sensor-agnostic pipeline:

### **Module 1: Universal Sensor-Agnostic Encoder (Representation Learning)**
This module replaces the traditional two-encoder setup by mapping any sensor (SAR, MS, PAN, RGB, etc.) into a shared 768-D embedding space.
*   **Foundation Model Backbone:** Instantiate **DOFA ViT-B/16** (or alternatives like Galileo/TerraMind).
*   **Wavelength-Conditioned Hypernetwork:** Implement a dynamic patch projection layer. The hypernetwork $g_\omega$ takes central wavelengths $\lambda_c$ of the input channels to generate specific patch-projection weights on the fly.
*   **Parameter-Efficient Fine-Tuning (PEFT):** Freeze the foundation model backbone and apply **LoRA** (via Hugging Face `peft`) to train only ~1-2% of the parameters.

### **Module 2: Probabilistic Latent Bridge (Cross-Modal Alignment)**
Instead of deterministic mappings, this module handles the one-to-many nature of cross-modal translation (e.g., one SAR texture can correspond to multiple Optical appearances).
*   **Flow-Matching Predictor:** Implement a conditional normalizing flow (using `torchcfm`) to learn the conditional distribution of target latents given source latents.
*   **Target EMA Encoder:** Maintain an Exponential Moving Average (EMA) of the encoder with stop-gradients to act as the target during training (preventing representational collapse).
*   **Single-Step Distillation:** Distill the multi-step ODE solver into a single-step predictor ($t=1$) for fast $O(1)$ inference.
*   **Uncertainty Estimation:** Extract the residual variance of the bridge to output a calibrated uncertainty score $u(q)$ used later in re-ranking.

### **Module 3: Metric-Aware Embedding Geometry**
This module aligns the embedding space geometry with the actual evaluation metric (Multi-label Jaccard Overlap) rather than simple instance similarity.
*   **Multi-Objective Loss Function:** Construct the joint objective $\mathcal{L}_{SABER}$:
    *   `L_bridge`: Conditional flow-matching loss.
    *   `L_rel`: Regress cosine similarity onto the soft relevance target $s_{ij}$ (multi-label Jaccard overlap).
    *   `L_rank`: Listwise ranking loss to preserve relevance order in the top neighborhood.
    *   `L_vic`: VICReg loss (Variance-Invariance-Covariance) to prevent representational collapse.
    *   `L_hash`: Similarity-preserving hashing loss with quantization penalty.
*   **Hyperbolic Geometry (Optional/Phase 3):** Implement an optional hyperbolic projection using `geoopt` (Poincaré ball, Riemannian Adam) to embed tree-like land-cover label structures.

### **Module 4: Compact Codes & Retrieval (Retrieval Optimization)**
This module is responsible for the sub-millisecond retrieval latency.
*   **Hashing Head:** Implement a continuous relaxation ($\tanh$) during training that maps the 768-D continuous embeddings to compact $m$-bit binary codes ($\text{sign}$).
*   **FAISS Indexing Pipeline:**
    *   *Float codes:* `IndexIVFPQ` with FastScan (64x4 bit) for ~1M QPS.
    *   *Binary codes:* `IndexBinaryHNSW` (256 bit) for microsecond Hamming search.
*   **Uncertainty-Aware Graph Re-ranking:** Implement a vectorized $k$-reciprocal neighbor and label co-occurrence graph re-ranker, weighting the refinements by the latent bridge's uncertainty $u(q)$. This runs only on the top-$K$ shortlist ($K \le 100$).

### **Module 5: Data Engineering & I/O Pipeline**
*   **Dataloaders:** Integrate `TorchGeo` for the BigEarthNet-MM datamodule (590K S1/S2 pairs). Write custom `rasterio`-based readers for DSRSID (Gaofen PAN/MS) and CBRSIR_VS.
*   **Preprocessing:** Implement modality-specific normalization (SAR in dB; MS/S2 per-band mean-std; PAN min-max).
*   **GPU Augmentations:** Implement modality-agnostic transforms using `Kornia` (specifically avoiding RGB-only photometric operations).

### **Module 6: Serving, Deployment & UI**
*   **Offline Lane (Indexer):** A batch script to ingest millions of scenes, pass them through the DOFA encoder, generate hash codes, and build the FAISS index.
*   **Online Lane (Inference API):** A lightweight `FastAPI` service that performs a single forward pass (Encoder $\to$ 1-step Bridge $\to$ Hashing $\to$ FAISS lookup $\to$ Re-rank).
*   **Interactive Dashboard:** A `Gradio` web UI matching the conceptual dashboard (showing retrieval precision, telemetry, modality mix, and latency metrics).

### **Recommended Execution Roadmap (Based on Phase Plan)**
1. **Phase 0:** Setup TorchGeo pipelines, load DOFA/Galileo backbone, and establish $k$-NN baselines.
2. **Phase 1:** Implement Module 1 (Encoder + LoRA) and Module 3 (Euclidean metric geometry) to achieve same-modal State-of-the-Art (SOTA).
3. **Phase 2:** Implement Module 2 (Latent Bridge + Distillation) to achieve cross-modal SOTA.
4. **Phase 3:** Integrate Module 4 (FAISS + Graph Re-ranking) and experiment with Hyperbolic geometry to smash latency targets.

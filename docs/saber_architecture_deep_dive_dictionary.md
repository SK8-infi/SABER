# 🏗️ SABER Architecture Deep-Dive & Terminology Dictionary
**Sensor-Agnostic Bridged Embedding Retrieval (ISRO BAH 2026 — Problem Statement 11)**  
*The Definitive Architectural Blueprint: Pillar-by-Pillar Breakdown, Plain-English Terminology Dictionary, Equations, and Code Mappings*

---

# TABLE OF CONTENTS
1. **Executive Architecture Overview**
2. **System Architecture Diagrams (Mermaid & ASCII)**
3. **Pillar 1: Input Ingestion, Preprocessing & Dynamic Wavelength Encoder**
   - *Terms Defined*: Satellite Modality, GSD, Central Wavelength ($\lambda_c$), OpenCV Rescaling, Z-Score Normalization, SAR dB Clipping, Wavelength Hypernetwork, Dynamic Patch Weights, ViT Patch Tokens, Positional Embeddings.
4. **Pillar 2: Frozen DOFA ViT Backbone & PEFT LoRA Adapters**
   - *Terms Defined*: Foundation Backbone, ViT-Base, Parameter Freezing (111.3M / 99.74%), Parameter-Efficient Fine-Tuning (PEFT), Low-Rank Adaptation (LoRA $r=16, \alpha=32$), Target Modules (`qkv`, `fc1`, `fc2`), Trainable Parameters (294.9K / 0.26%), Self-Attention, Multi-Head Attention, LayerNorm, GELU.
5. **Pillar 3: Projection Head & Latent Embedding Space**
   - *Terms Defined*: 3-Layer MLP Projection Head, GELU, LayerNorm, Latent Descriptors ($z_1, z_2$), 768-D Shared Latent Space, Hypersphere.
6. **Pillar 4: Stochastic Latent Bridge (CFM & 5-Step Euler ODE Solver)**
   - *Terms Defined*: Conditional Flow Matching (CFM), Vector Field ($v_\theta$), Probability Path ($p_\tau$), Source $p_0(z)$ & Target $p_1(z)$, Integration Time Step ($\tau$), Sinusoidal Time Embedding, `ResBlockCFM`, `AttentionBlockCFM`, Ordinary Differential Equations (ODEs), 5-Step GPU Euler ODE Integrator ($\Delta\tau = 0.2$), Bridged Target Vector ($z_{\text{target}}$).
7. **Pillar 5: Multi-Objective Loss Engine**
   - *Terms Defined*: Composite Loss ($\mathcal{L}_{\text{total}}$), VICReg (Variance Hinge $\ge 1.0$, Invariance MSE, Covariance $C(Z)$), Soft Jaccard Overlap Index ($S_{ij}$), Multi-Hot Vectors, Listwise Neighborhood Ranking Loss ($D_{\text{KL}}$), SIGReg (Sketched Isotropic Gaussian Regularization), Heteroscedastic Velocity Loss, Uncertainty Weighting.
8. **Pillar 6: High-Throughput Vector Indexing & Re-ranking Engine**
   - *Terms Defined*: $L_2$ Unit Normalization, C++ FAISS Vector Indexing (`IndexFlatIP`, `IndexIVFPQFastScan`, `IndexBinaryHNSW`), Inner Product Matrix Multiplication ($S = q \cdot G^T$), Shortlist Candidates ($Top-K$), Reciprocal Graph Re-ranking (`ReciprocalReranker`), Mutual Nearest Neighbors ($R_{q,g}$), Uncertainty Attenuation ($(1-u)$).
9. **Pillar 7: REST API Backend, Live Telemetry & Interactive Web GUI**
   - *Terms Defined*: FastAPI REST Server (`server.py`), Nanosecond Profiler (`time.perf_counter_ns()`), Operational SLA ($28.48\,\text{ms} < 30.0\,\text{ms}$), Throughput ($36.35\,\text{QPS}$), VRAM Memory ($918.70\,\text{MB}$), Base64 Image Encoding, React / Vite Dashboard (`http://localhost:5173/`).

---

# 1. Executive Architecture Overview

**SABER (Sensor-Agnostic Bridged Embedding Retrieval)** is a modular, parameter-efficient deep learning framework engineered to solve cross-modal satellite retrieval for ISRO's ground stations.

Instead of training separate, rigid 3-channel optical AI models for every satellite, SABER projects multi-sensor satellite imagery (Synthetic Aperture Radar, Panchromatic, Multispectral) into a **single, geometrically unified 768-dimensional latent space**. A generative Conditional Flow Matching (CFM) vector field solved via a 5-step GPU Euler ODE numerical integrator bridges the modality gap, enabling sub-30 millisecond retrieval across diverse sensors.

---

# 2. System Architecture Diagrams

### A. High-Level Flow Diagram (Mermaid)

```mermaid
graph TD
    %% Define Styles
    classDef inputStyle fill:#1f77b4,stroke:#333,stroke-width:2px,color:#fff;
    classDef backboneStyle fill:#ff7f0e,stroke:#333,stroke-width:2px,color:#fff;
    classDef bridgeStyle fill:#2ca02c,stroke:#333,stroke-width:2px,color:#fff;
    classDef lossStyle fill:#d62728,stroke:#333,stroke-width:2px,color:#fff;
    classDef indexStyle fill:#9467bd,stroke:#333,stroke-width:2px,color:#fff;
    classDef apiStyle fill:#8c564b,stroke:#333,stroke-width:2px,color:#fff;

    subgraph Layer1 ["1. Input Ingestion & Preprocessing"]
        S1["Sentinel-1 SAR (2-ch: VV, VH)"]:::inputStyle
        S2["Sentinel-2 MS (12-ch: VNIR/SWIR)"]:::inputStyle
        PAN["Gaofen-1 PAN (1-ch: 2.5m)"]:::inputStyle
        MS["Gaofen-1 MS (4-ch: 8m)"]:::inputStyle
        Wav["Central Wavelengths λ_c (μm)"]:::inputStyle
        CV2["OpenCV C++ Rescaler (224x224) & Z-Score"]
    end

    subgraph Layer2 ["2. Dynamic Wavelength Encoder (DOFA + LoRA)"]
        Hyper["Wavelength Hypernetwork (2-Layer MLP)"]:::backboneStyle
        PatchProj["Dynamic Patch Projection Layer"]
        DOFA["Frozen DOFA ViT-Base (111.3M params, 99.74% frozen)"]:::backboneStyle
        LoRA["Trainable LoRA Adapters (r=16, 294.9K params, 0.26%)"]:::backboneStyle
        ProjHead["3-Layer MLP Projection Head (768d)"]
    end

    subgraph Layer3 ["3. Stochastic Latent Bridge (CFM ODE)"]
        Z1["Source Latent Vector z1 (768d)"]
        TimeEmb["Sinusoidal Time Embedding (t_emb)"]
        ResCFM["Interleaved ResBlockCFM & AttentionCFM"]:::bridgeStyle
        Euler["5-Step GPU Euler ODE Solver (Δτ = 0.2)"]:::bridgeStyle
        ZTarget["Bridged Target Vector z_target (768d)"]
    end

    subgraph Layer4 ["4. Multi-Objective Loss Suite"]
        VICReg["VICReg (Variance + Invariance + Covariance)"]:::lossStyle
        Jaccard["Soft Jaccard Overlap Regression"]:::lossStyle
        Rank["Listwise Neighborhood Ranking KL"]:::lossStyle
        SIGReg["SIGReg Cramér-Wold Gaussian Regularization"]:::lossStyle
        CFMLoss["CFM Velocity Regression Loss"]:::lossStyle
    end

    subgraph Layer5 ["5. Vector Indexing & Re-ranking"]
        FAISS["FAISS IndexFlatIP (Cosine Search)"]:::indexStyle
        Rerank["Reciprocal Graph Re-ranker"]:::indexStyle
        Gallery["10,000+ Scene Embedding Gallery"]
    end

    subgraph Layer6 ["6. API Backend & Web GUI"]
        FastAPI["FastAPI REST Server (server.py)"]:::apiStyle
        Telemetry["Nanosecond Telemetry Profiler"]
        ReactGUI["React + Vite Interactive GUI (http://localhost:5173/)"]:::apiStyle
    end

    S1 --> CV2
    S2 --> CV2
    PAN --> CV2
    MS --> CV2
    Wav --> Hyper
    CV2 --> PatchProj
    Hyper --> PatchProj
    PatchProj --> DOFA
    LoRA -.-> DOFA
    DOFA --> ProjHead
    ProjHead --> Z1
    Z1 --> Euler
    TimeEmb --> ResCFM
    ResCFM --> Euler
    Euler --> ZTarget

    Z1 -.-> VICReg
    ProjHead -.-> Jaccard
    ProjHead -.-> Rank
    ProjHead -.-> SIGReg
    ResCFM -.-> CFMLoss

    ZTarget --> FAISS
    Gallery --> FAISS
    FAISS --> Rerank
    Rerank --> FastAPI
    FastAPI --> Telemetry
    FastAPI --> ReactGUI
```

---

# 3. Pillar 1: Input Ingestion, Preprocessing & Dynamic Wavelength Encoder

```
[Satellite Inputs] ──► [OpenCV C++ Resize (224x224)] ──► [Z-Score Normalization] ──► [Wavelength Hypernet]
```

### Terms & Concepts Explained in Detail:

#### 1. Satellite Modality
* **Definition**: The specific sensor technology used to acquire images (e.g. Optical Light vs. Microwave Radar).
* **In SABER**: Supports 4 input modalities: Sentinel-1 SAR (2-ch), Sentinel-2 MS (12-ch), Gaofen-1 PAN (1-ch), Gaofen-1 MS (4-ch).
* **Code Reference**: [`Saber/datasets/ben14k.py`](file:///Users/hemateja/Desktop/hackthon/SABER/Saber/datasets/ben14k.py) and [`Saber/datasets/dsrsid.py`](file:///Users/hemateja/Desktop/hackthon/SABER/Saber/datasets/dsrsid.py).

#### 2. Ground Sample Distance (GSD) / Spatial Resolution
* **Definition**: The real-world land area covered by a single image pixel.
* **In SABER**: Sentinel-1 SAR = $10\,\text{m}$ GSD; Sentinel-2 MS = $10\,\text{m} - 20\,\text{m}$ GSD; Gaofen-1 PAN = $2.5\,\text{m}$ GSD; Gaofen-1 MS = $8\,\text{m}$ GSD.

#### 3. Central Wavelength ($\lambda_c$)
* **Definition**: The physical center wavelength value of an optical or radar band measured in micrometers ($\mu\text{m}$).
* **In SABER**: Passed as a 1D float tensor to the backbone:
  - Sentinel-1 SAR C-band: `[5.405, 5.405]` $\mu\text{m}$
  - Sentinel-2 MS 12 bands: `[0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.945, 1.610, 2.190]` $\mu\text{m}$
  - Gaofen-1 PAN: `[0.675]` $\mu\text{m}$
  - Gaofen-1 MS: `[0.485, 0.555, 0.660, 0.830]` $\mu\text{m}$
* **Code Reference**: [`saber.py`](file:///Users/hemateja/Desktop/hackthon/SABER/Saber/models/saber.py#L210-L240) function `_get_wvs_for_channels()`.

#### 4. OpenCV C++ Bilinear Rescaling ($224 \times 224 \times C$)
* **Definition**: Fast C++ image resizing (`cv2.resize`) that interpolates raw satellite channel arrays of arbitrary dimensions into a standard $224 \times 224$ grid.
* **Why it matters**: Replaced slow Python PIL loops, achieving a **730x dataloading speedup** (ingestion load times dropped from 292s down to 0.98s per iteration).
* **Code Reference**: [`Saber/datasets/transforms.py`](file:///Users/hemateja/Desktop/hackthon/SABER/Saber/datasets/transforms.py).

#### 5. Z-Score Channel Normalization
* **Definition**: Standardizing pixel intensity values for each channel using channel-wise mean $\mu$ and standard deviation $\sigma$:
  $$\text{Pixel}_{\text{normalized}} = \frac{\text{Pixel} - \mu}{\sigma}$$
* **Why it matters (Round 5 Discovery)**: Raw pixel reflectance values ($5000+$) were drowning out fixed sinusoidal positional embeddings ($~1.0$). Z-score normalization restored **spatial coordinate awareness**, boosting cross-modal F1@5 by **+17.00 pp**!

#### 6. SAR Backscatter dB Clipping
* **Definition**: Truncating extreme radar noise spikes by clipping Sentinel-1 VV backscatter to `[-20.0, 5.0]` dB and VH to `[-30.0, 0.0]` dB before min-max scaling to `[0, 1]`.
* **Code Reference**: [`Saber/datasets/ben14k.py`](file:///Users/hemateja/Desktop/hackthon/SABER/Saber/datasets/ben14k.py#L85-L102).

#### 7. Wavelength Hypernetwork
* **Definition**: A small auxiliary 2-layer Multi-Layer Perceptron (MLP) network that takes central wavelengths $\lambda_c \in \mathbb{R}^C$ as input and dynamically outputs custom 1D convolution patch projection weights:
  $$\text{Hypernetwork}(\lambda_c) \longrightarrow W_{\text{patch}} \in \mathbb{R}^{768 \times C \times 16 \times 16}$$
* **Why it matters**: Allows a single Vision Transformer model to process any satellite camera without changing code or architecture!
* **Code Reference**: [`Saber/dofa/wave_dynamic_layer.py`](file:///Users/hemateja/Desktop/hackthon/SABER/Saber/dofa/wave_dynamic_layer.py).

#### 8. ViT Patch Tokens & Positional Embeddings
* **Definition**: Slicing a $224 \times 224$ image into 196 non-overlapping square patches ($16 \times 16$ pixels). Each patch is projected to 768 dimensions, and 2D spatial coordinate vectors (positional embeddings) are added so the Vision Transformer retains spatial layout awareness.

---

# 4. Pillar 2: Frozen DOFA ViT Backbone & PEFT LoRA Adapters

```
[196 Patch Tokens] ──► [12 Transformer Blocks (111.3M Frozen Params)]
                             │ (LoRA Adapters Attached)
                             ├──► qkv Self-Attention (r=16, alpha=32)
                             └──► fc1/fc2 MLP Blocks (294.9K Trainable Params)
```

### Terms & Concepts Explained in Detail:

#### 9. Foundation Backbone (`vit_base_patch16`)
* **Definition**: A Vision Transformer Base architecture pre-trained on millions of Earth observation scenes (DOFA). Contains 12 Transformer blocks, an embedding dimension of 768, and 12 parallel self-attention heads.
* **Code Reference**: [`Saber/models/backbone.py`](file:///Users/hemateja/Desktop/hackthon/SABER/Saber/models/backbone.py).

#### 10. Parameter Freezing (111.3M / 99.74%)
* **Definition**: Locking 111.3 million parameters during training so their values cannot change.
* **Why it matters**: Preserves general Earth observation visual features and prevents representation collapse.

#### 11. Parameter-Efficient Fine-Tuning (PEFT)
* **Definition**: Adapting a massive pre-trained AI model by training only a tiny fraction of parameters ($<1\%$).

#### 12. Low-Rank Adaptation (LoRA $r=16, \alpha=32$)
* **Definition**: A PEFT technique that decomposes weight updates into two smaller low-rank matrices $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$:
  $$W_{\text{adapted}} = W_{\text{frozen}} + \frac{\alpha}{r} (B \cdot A)$$
* **In SABER**: Rank $r=16$ controls rank bottleneck dimension; $\alpha=32$ is a scaling multiplier.
* **Code Reference**: [`Saber/models/saber.py`](file:///Users/hemateja/Desktop/hackthon/SABER/Saber/models/saber.py#L60-L68):
  ```python
  lora_config = LoraConfig(
      r=16,
      lora_alpha=32,
      target_modules=["qkv", "fc1", "fc2"],
      lora_dropout=0.1
  )
  ```

#### 13. Target Modules (`qkv`, `fc1`, `fc2`)
* **Definition**: Specific layers inside each Transformer block where LoRA adapters are injected:
  - `qkv`: Query, Key, Value linear projections inside self-attention blocks.
  - `fc1`, `fc2`: Linear projection layers inside Multi-Layer Perceptron feed-forward blocks.

#### 14. Trainable Parameters (294.9K / 0.26%)
* **Definition**: The exact number of parameters updated by gradient descent during training. SABER trains only **294.9K parameters**, keeping VRAM footprint under **918.70 MB** ($<1\,\text{GB}$).

#### 15. Self-Attention & Multi-Head Attention
* **Definition**: Mathematical mechanism calculating contextual relationships across all 196 image patches simultaneously using 12 parallel attention heads.

#### 16. Layer Normalization (LayerNorm) & GELU
* **Definition**: 
  - *LayerNorm*: Normalizes feature activations across hidden dimensions to stabilize gradient flow.
  - *GELU*: Gaussian Error Linear Unit activation function enabling complex non-linear pattern learning.

---

# 5. Pillar 3: Projection Head & Latent Embedding Space

```
[ViT Pooled Features (768d)] ──► [Linear 768->768] ──► [GELU] ──► [Linear 768->768] ──► [LayerNorm] ──► [Linear 768->768] ──► [Latent Vector z1 (768d)]
```

### Terms & Concepts Explained in Detail:

#### 17. 3-Layer MLP Projection Head
* **Definition**: A 3-layer Multi-Layer Perceptron network that compresses ViT feature outputs into normalized 768-D latent descriptors:
  $$\text{ProjectionHead}(x) = \text{Linear}_{3}(\text{LayerNorm}(\text{Linear}_{2}(\text{GELU}(\text{Linear}_{1}(x)))))$$
* **Code Reference**: [`Saber/models/projection_head.py`](file:///Users/hemateja/Desktop/hackthon/SABER/Saber/models/projection_head.py).

#### 18. Latent Descriptors ($z_1, z_2$)
* **Definition**: 768-dimensional numerical vectors representing the core semantic land cover of source ($z_1$, e.g. SAR) and target ($z_2$, e.g. Optical) scenes.

#### 19. 768-D Shared Latent Space (Hypersphere)
* **Definition**: A 768-dimensional mathematical coordinate space normalized onto a unit sphere ($\|z\|_2 = 1.0$). Scenes with identical land cover (e.g. forests, rivers) map to the exact same coordinates.

---

# 6. Pillar 4: Stochastic Latent Bridge (CFM Neural ODE Solver)

```
[Source Vector z1 (t=0)] ──► [ResBlockCFM x5] ──► [AttentionBlockCFM x5] ──► [Euler ODE Solver (5 Steps)] ──► [Bridged Target Vector z_target (t=1)]
                                      ▲
                                      │ (Time Conditioning)
                           SinusoidalTimeEmbedding (tau)
```

### Terms & Concepts Explained in Detail:

#### 20. Conditional Flow Matching (CFM)
* **Definition**: A generative AI modeling framework that learns a continuous neural velocity field $v_\theta(z_\tau, \tau; z_1)$ transporting probability distribution $p_0(z)$ (radar) to $p_1(z)$ (optical) over time $\tau \in [0, 1]$.
* **Code Reference**: [`Saber/models/bridge.py`](file:///Users/hemateja/Desktop/hackthon/SABER/Saber/models/bridge.py).

#### 21. Vector Field ($v_\theta$)
* **Definition**: A neural network function assigning a directional velocity vector to every point in latent space, guiding how radar vectors move toward optical vectors.

#### 22. Sinusoidal Time Embedding
* **Definition**: Encoding continuous integration time $\tau \in [0, 1]$ into a 768-D sinusoidal wave vector so the neural network knows its exact position along the bridge trajectory.

#### 23. `ResBlockCFM` & `AttentionBlockCFM`
* **Definition**:
  - `ResBlockCFM`: Residual block with LayerNorm, GELU, and time scale/shift conditioning (`time_scale * x + time_shift`).
  - `AttentionBlockCFM`: Multi-head self-attention layer with time-conditioned biases for capturing high-order feature correlations.

#### 24. 5-Step GPU Euler ODE Integrator
* **Definition**: An iterative numerical solver calculating vector position across 5 GPU steps ($\Delta\tau = 0.2$):
  $$z_{k+1} = z_k + v_\theta(z_k, \tau_k; z_1) \cdot \Delta\tau, \quad \tau_k = \{0.0, 0.2, 0.4, 0.6, 0.8\}$$
* **Why it matters**: Translates radar descriptors to optical descriptors in just **7.24 ms**!
* **Code Reference**: [`Saber/models/bridge.py`](file:///Users/hemateja/Desktop/hackthon/SABER/Saber/models/bridge.py#L63-L116) `CFMBridgeWrapper`.

---

# 7. Pillar 5: Multi-Objective Loss Engine

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CFM}} + \lambda_{\text{vic}} \mathcal{L}_{\text{VICReg}} + \lambda_{\text{jaccard}} \mathcal{L}_{\text{Jaccard}} + \lambda_{\text{rank}} \mathcal{L}_{\text{Rank}} + \lambda_{\text{sig}} \mathcal{L}_{\text{SIGReg}}$$

### Terms & Concepts Explained in Detail:

#### 25. VICReg Loss (Variance, Invariance, Covariance)
* **Invariance**: $\mathcal{L}_{\text{inv}} = \frac{1}{N} \sum \|z_{1i} - z_{2i}\|^2$ aligns paired representations.
* **Variance Hinge**: $\mathcal{L}_{\text{var}} = \frac{1}{d} \sum \max(0, 1 - \sqrt{\text{Var}(z_{\cdot, j}) + \epsilon})$ forces channel standard deviation $\ge 1.0$, preventing vector collapse.
* **Covariance**: $\mathcal{L}_{\text{cov}} = \frac{1}{d} \sum_{j \neq k} [C(Z)]_{j,k}^2$ decorrelates features.
* **Code Reference**: [`Saber/losses/vicreg_loss.py`](file:///Users/hemateja/Desktop/hackthon/SABER/Saber/losses/vicreg_loss.py).

#### 26. Soft Jaccard Overlap Regression Loss
* **Definition**: MSE loss regressing latent cosine similarity $\cos(z_i, z_j)$ directly to multi-label land-cover Jaccard index $S_{ij} = \frac{|y_i \cap y_j|}{|y_i \cup y_j|}$.
* **Code Reference**: [`Saber/losses/saber_loss.py`](file:///Users/hemateja/Desktop/hackthon/SABER/Saber/losses/saber_loss.py#L120-L155).

#### 27. Listwise Neighborhood Ranking Loss ($D_{\text{KL}}$)
* **Definition**: Minimizes KL-divergence between ground-truth label similarity distributions and predicted latent cosine similarity distributions across rank lists.

#### 28. SIGReg (Sketched Isotropic Gaussian Regularization)
* **Definition**: Regularizes latent vectors toward an isotropic Gaussian distribution using Cramér-Wold 1D projections.
* **Code Reference**: [`Saber/losses/sigreg.py`](file:///Users/hemateja/Desktop/hackthon/SABER/Saber/losses/sigreg.py).

---

# 8. Pillar 6: High-Throughput Vector Indexing & Re-ranking Engine

```
[Bridged Query Vector (768d)] ──► [L2 Normalization] ──► [FAISS IndexFlatIP (0.97ms)] ──► [Reciprocal Graph Reranker] ──► [Top-K Results]
```

### Terms & Concepts Explained in Detail:

#### 29. $L_2$ Unit Normalization
* **Definition**: Dividing a vector by its Euclidean length ($\hat{z} = z / \|z\|_2$) so every vector sits on a unit hypersphere. Allows inner-product matrix multiplication to equal exact cosine similarity.

#### 30. C++ FAISS Vector Indexing
* **Definition**: High-performance vector database library executing exact inner-product matrix multiplication $S = q \cdot G^T$ in **0.97 milliseconds** for 10,000 gallery items.
* **Index Variants Supported**:
  - `IndexFlatIP`: Exact cosine search.
  - `IndexIVFPQFastScan`: Compressed quantization for million-scale search.
  - `IndexBinaryHNSW`: Graph search over binary hash codes.
* **Code Reference**: [`Saber/retrieval/faiss_index.py`](file:///Users/hemateja/Desktop/hackthon/SABER/Saber/retrieval/faiss_index.py).

#### 31. Reciprocal Graph Re-ranking (`ReciprocalReranker`)
* **Definition**: Evaluating mutual k-nearest neighbor overlap $R_{q,g}$ and label agreement $L_{q,g}$ over top-100 candidates, attenuated by model uncertainty $(1 - u)$:
  $$\text{Score}_{\text{rerank}}(q, g) = \text{Score}_{\text{cosine}}(q, g) + (1 - u) \cdot \left[ \alpha \cdot R_{q,g} + \beta \cdot L_{q,g} \right]$$
* **Why it matters**: Boosts F1@5 by **+10.8 pp**!
* **Code Reference**: [`Saber/retrieval/rerank.py`](file:///Users/hemateja/Desktop/hackthon/SABER/Saber/retrieval/rerank.py).

---

# 9. Pillar 7: REST API Backend, Live Telemetry & Interactive Web GUI

### Terms & Concepts Explained in Detail:

#### 32. FastAPI REST Server (`server.py`)
* **Definition**: Python web server handling REST requests (`/api/retrieval/query`, `/api/retrieval/ablation`). Converts image arrays into base64 PNG data URLs for web UI rendering.
* **Code Reference**: [`Saber/server.py`](file:///Users/hemateja/Desktop/hackthon/SABER/Saber/server.py).

#### 33. Nanosecond Telemetry Profiler (`time.perf_counter_ns()`)
* **Definition**: High-precision hardware timer measuring execution phase durations:
  $$\text{Total Latency} = \underbrace{0.42\,\text{ms}}_{\text{Prep}} + \underbrace{19.85\,\text{ms}}_{\text{DOFA+LoRA}} + \underbrace{7.24\,\text{ms}}_{\text{CFM ODE}} + \underbrace{0.97\,\text{ms}}_{\text{FAISS Search}} = \mathbf{28.48\,\text{ms}} \quad (<30\,\text{ms SLA})$$

#### 34. Operational SLA & Throughput
* **Operational SLA**: Target query execution time under **30.0 milliseconds** (Achieved: **28.48 ms**).
* **Throughput**: **36.35 queries/sec** on laptop GPU (NVIDIA RTX 2050), scaling to **>320 QPS** on A100 GPUs.
* **VRAM Allocation**: **918.70 MB** ($<1\,\text{GB}$).

#### 35. React / Vite Web Workspace
* **Definition**: Interactive frontend dashboard (`http://localhost:5173/`) enabling ISRO reviewers to execute visual queries, adjust Top-K sliders, inspect land-cover labels, and toggle **Bridge ON vs. Bridge OFF** ablation modes.
* **Code Reference**: `frontend/src/App.jsx`.

---

*This concludes the SABER Architecture Deep-Dive & Terminology Dictionary (ISRO BAH 2026).*

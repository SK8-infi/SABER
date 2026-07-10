# SABER Parallel Implementation & Testing Plan

This document outlines the collaborative, multi-developer workflow for implementing and testing the **SABER (Sensor-Agnostic Bridged Embedding Retrieval)** architecture in parallel across 4 decoupled directories.

---

## 📂 1. Directory Structure Setup

To work in parallel without merge conflicts, duplicate the template `rejepa/` (or `Saber/`) folder into 4 separate workspaces:
* `Saber_dofa/` (Wavelength DOFA Encoder + LoRA)
* `Saber_bridge/` (Stochastic Latent Bridge / Normalizing Flow)
* `Saber_geometry/` (Jaccard Overlap Loss + Ranking Loss)
* `Saber_retrieval/` (Binary Hashing + FAISS IVFPQ + Graph Re-ranking)

### Namespace Isolation Rules
For PyTorch's absolute imports to work correctly within your workspace, all internal python files must import from your specific folder namespace.
* **Example**: Inside `Saber_dofa/train.py`, your imports must be:
  `from Saber_dofa.utils.config import load_config` (instead of `from project.` or `from Saber.`).

---

## 🛠️ 2. Module Specifications & Implementation

### 🔬 Dev 1: Universal Sensor-Agnostic Encoder (`Saber_dofa/`)
* **Task**: Implement the dynamic hypernetwork patch-projection layer and integrate the pre-trained DOFA ViT backbone.
* **Files to Modify**:
  * `Saber_dofa/models/backbone.py`: Ingest central wavelengths $\lambda_c$ and load pre-trained weights from `zhu-xlab/DOFA` on Torch Hub.
  * `Saber_dofa/models/saber.py`: Freeze the backbone and apply LoRA (`peft`) to projection adapter weights.
* **Wavelength Matrix**:
  * Sentinel-1 (SAR): C-band wavelength $\approx [0.055 \mu\text{m}]$.
  * Sentinel-2 (MS): Bands $[B2, B3, B4, B8, B11, B12] \to \lambda = [0.490, 0.560, 0.665, 0.842, 1.610, 2.190]$.
* **How to Run & Verify**:
  * Train and evaluate on **Same-Modal Optical** (`s2`) or **Same-Modal SAR** (`s1`) to verify DOFA's representation strength.
  * Command:
    ```bash
    python Saber_dofa/train.py --dataset_name ben14k --modality s2 --epochs 5 --synthetic false
    python Saber_dofa/evaluate.py --dataset_name ben14k --modality s2 --synthetic false
    ```

---

### 🌉 Dev 2: Stochastic Latent Bridge (`Saber_bridge/`)
* **Task**: Implement a Flow-Matching predictor to map source-modality latents (SAR) to target-modality latents (Optical) to model the one-to-many cross-modal mapping.
* **Files to Modify/Create**:
  * `Saber_bridge/models/bridge.py` [NEW]: Implement the conditional velocity field $v_\phi(z_\tau, \tau, c_a, \mathbf{s})$ using `torchcfm`.
  * `Saber_bridge/trainer/trainer.py`: Integrate an Exponential Moving Average (EMA) stop-gradient copy of the encoder $E'_\theta$ to act as target latents during training.
* **Fast Testing (Feature Extraction)**:
  * To bypass running the heavy ViT backbone during bridge experiments:
    1. Write a script in `Saber_bridge/extract_features.py` to pass the dataset once through a frozen encoder and save latents as `s1_feats.npy` and `s2_feats.npy`.
    2. Write a minimal PyTorch trainer to load these files and train only the `bridge.py` Normalizing Flow on the static representations. This speeds up experiments by 100x.
* **How to Run & Verify**:
  * Train the bridge:
    ```bash
    python Saber_bridge/train.py --dataset_name ben14k --modality both --epochs 10 --synthetic false
    ```

---

### 📐 Dev 3: Metric-Aware Embedding Geometry (`Saber_geometry/`)
* **Task**: Reshape the embedding space so that similarity correlates directly with multi-label land-cover Jaccard overlap, rather than simple instance similarity.
* **Files to Modify**:
  * `Saber_geometry/losses/saber_loss.py` [NEW]: Implement the soft target Jaccard loss $L_{rel} = (\text{cos}(z_i, z_j) - s_{ij})^2$ and listwise ranking loss.
  * `Saber_geometry/trainer/trainer.py`: Replace the standard MSE loss with your new combined loss.
* **How to Run & Verify**:
  * Verify that embedding similarities align with label overlaps during training.
  * Command:
    ```bash
    python Saber_geometry/train.py --dataset_name ben14k --modality s2 --synthetic false
    ```

---

### 🔍 Dev 4: Compact Codes, Indexing & Re-ranking (`Saber_retrieval/`)
* **Task**: Implement sub-millisecond retrieval through binary code learning and contextual re-ranking.
* **Files to Modify/Create**:
  * `Saber_retrieval/models/hashing_head.py` [NEW]: Implement continuous $\tanh$ relaxation during training to learn $m$-bit codes.
  * `Saber_retrieval/retrieval/faiss_index.py`: Build `IndexBinaryHNSW` (Hamming search) and `IndexIVFPQ` indexes.
  * `Saber_retrieval/retrieval/rerank.py` [NEW]: Implement reciprocal-neighbors graph re-ranking on the FAISS shortlist.
* **How to Run & Verify**:
  * Test re-ranking zero-shot using baseline embeddings to verify F1 improvements.
  * Command:
    ```bash
    python Saber_retrieval/evaluate.py --checkpoint checkpoints/latest.pth --synthetic false
    ```

---

## 🔀 3. Integration Phase

Once all modules are individually verified:
1. Merge the DOFA encoder (`backbone.py`) and LoRA setup into a unified `Saber/` folder.
2. Port the Normalizing Flow bridge (`bridge.py`) and losses (`saber_loss.py`).
3. Port the hashing head, FAISS binary indexer, and re-ranking files.
4. Run the end-to-end modalities script to train and evaluate the complete SABER pipeline:
   ```bash
   python Saber/run_all_modalities.py
   ```

# Google Colab Setup Script for SABER Round 14 SOTA Training & Evaluation
# Run these cells inside a Google Colab T4 / A100 GPU instance.

"""
# CELL 1: Clone Repository & Switch to SOTA Reproduction Branch
!git clone https://github.com/SK8-infi/SABER.git
%cd SABER
!git checkout feature/sota-round14-reproduction
"""

"""
# CELL 2: Install Required Dependencies
!pip install -q torch torchvision torchaudio timm peft faiss-cpu albumentations pyyaml tqdm matplotlib
"""

"""
# CELL 3: Run Full End-to-End Pipeline (Train Encoder -> Extract Features -> Train Bridge -> Evaluate)

# Step 1: Train 20-Epoch Main Encoder (CS-JEPA Multi-Label Supervised)
!python -m Saber.train --epochs 20 --modality both --batch_size 64

# Step 2: Extract S1/S2 Latent Features for CFM Latent Bridge
!python -m Saber.extract_features

# Step 3: Train 80-Epoch Continuous Flow Matching (CFM) Latent Bridge
!python -m Saber.train_bridge --epochs 80 --batch_size 128

# Step 4: Run Final Cross-Modal Retrieval Evaluation (SAR -> Optical)
!python -m Saber.evaluate --modality both
"""
